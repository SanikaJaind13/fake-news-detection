import json
import os
import tempfile
import unittest

import requests

from src import database
from src.config import Settings
from src.errors import ConfigError, InputError, RetrievalError
from src.models import CONTRADICTED, INSUFFICIENT, SUPPORTED, ClaimResult, Claim, Document
from src.pipeline import aggregate, analyze
from src.retriever import extract_text_from_html, fetch_page_text, retrieve_documents, search_web

HTML = """<html><head><title>Test Page</title></head><body>
<nav>Menu Home About</nav><script>var x = 1;</script>
<article><p>The Eiffel Tower in Paris was completed in 1889 for the World's Fair.</p>
<p>Short.</p><p>It was designed by the engineering company of Gustave Eiffel in France.</p></article>
<footer>Copyright junk footer text that should be removed from output</footer></body></html>"""


class Resp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code, self._body, self.text = status, body, text
        self.headers = headers or {"Content-Type": "text/html"}

    def json(self):
        if self._body is None:
            raise ValueError("bad json")
        return self._body


class Session:
    def __init__(self, post=None, get=None):
        self._post, self._get = post, get

    def post(self, *a, **k):
        if isinstance(self._post, Exception):
            raise self._post
        return self._post

    def get(self, *a, **k):
        if isinstance(self._get, Exception):
            raise self._get
        return self._get


S = Settings(groq_api_key="g", tavily_api_key="t")
TEST_USER_ID = 123


class TestRetriever(unittest.TestCase):
    def test_html_extraction_removes_junk(self):
        title, text = extract_text_from_html(HTML)
        self.assertEqual(title, "Test Page")
        self.assertIn("Eiffel Tower", text)
        self.assertNotIn("Menu", text)
        self.assertNotIn("Copyright junk", text)
        self.assertNotIn("Short.", text)

    def test_fetch_errors(self):
        with self.assertRaises(RetrievalError):
            fetch_page_text("https://x.com", S, Session(get=requests.ConnectionError("down")))
        with self.assertRaises(RetrievalError):
            fetch_page_text("https://x.com", S, Session(get=Resp(404)))
        with self.assertRaises(RetrievalError):
            fetch_page_text("https://x.com", S, Session(get=Resp(200, headers={"Content-Type": "application/pdf"})))

    def test_search_errors(self):
        with self.assertRaises(ConfigError):
            search_web("q", Settings(), Session())
        with self.assertRaises(ConfigError):
            search_web("q", S, Session(post=Resp(401)))
        with self.assertRaises(RetrievalError):
            search_web("q", S, Session(post=Resp(429)))
        with self.assertRaises(RetrievalError):
            search_web("q", S, Session(post=requests.Timeout()))

    def test_retrieve_documents_falls_back_to_snippet(self):
        results = {"results": [{"url": "https://www.reuters.com/a", "title": "T", "content": "Snippet about the claim."},
                               {"url": "https://www.reuters.com/a", "title": "dup", "content": "dup"}]}
        session = Session(post=Resp(200, results), get=Resp(500))  # page download fails
        docs = retrieve_documents("q", S, session)
        self.assertEqual(len(docs), 1)  # duplicate URL removed
        self.assertEqual(docs[0].text, "Snippet about the claim.")
        self.assertGreater(docs[0].credibility, 0.9)


def fake_search(query, settings):
    return [Document("https://reuters.com/x", "Eiffel", "reuters.com",
                     "The Eiffel Tower was completed in 1889 in Paris. It opened for the World's Fair. "
                     "Millions of people visit the tower every year.", 0.95, "Wire")]


def fake_llm_factory(verdict):
    def fn(messages, settings):
        key = "supporting" if verdict == "Supported" else "contradicting"
        return json.dumps({"verdict": verdict, "confidence": 0.9, "reasoning": "ok [E1]", key: ["E1"]})
    return fn


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(groq_api_key="g", tavily_api_key="t",
                                 db_path=os.path.join(self.tmp.name, "h.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_saves_history_and_reports_progress(self):
        events = []
        res = analyze("The Eiffel Tower was completed in 1889 in Paris.", self.settings,
                      progress=lambda f, m: events.append(f), search_fn=fake_search,
                      llm_fn=fake_llm_factory("Supported"), user_id=TEST_USER_ID)
        self.assertEqual(res.overall_verdict, SUPPORTED)
        self.assertIsNotNone(res.id)
        self.assertEqual(database.get_analysis(self.settings.db_path, res.id, TEST_USER_ID)["overall_verdict"], SUPPORTED)
        self.assertEqual(events[-1], 1.0)
        self.assertEqual(events, sorted(events))

    def test_contradicted(self):
        res = analyze("The Eiffel Tower was completed in 1950 in Paris.", self.settings,
                      search_fn=fake_search, llm_fn=fake_llm_factory("Contradicted"), save=False)
        self.assertEqual(res.overall_verdict, CONTRADICTED)
        self.assertIsNone(res.id)

    def test_input_and_config_validation(self):
        with self.assertRaises(InputError):
            analyze("hi", self.settings)
        with self.assertRaises(ConfigError):
            analyze("A long enough claim about something.", Settings())

    def test_partial_failure_is_isolated(self):
        calls = {"n": 0}

        def flaky(query, settings):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RetrievalError("search failed")
            return fake_search(query, settings)

        text = ("The Eiffel Tower was completed in 1889 in Paris. "
                "Gustave Eiffel designed the 330 metre tower in France.")
        res = analyze(text, self.settings, search_fn=flaky, llm_fn=fake_llm_factory("Supported"), save=False)
        self.assertEqual(len(res.claim_results), 2)
        self.assertIsNotNone(res.claim_results[0].error)
        self.assertEqual(res.claim_results[1].verdict, SUPPORTED)

    def test_all_failures_raise(self):
        def broken(q, s):
            raise RetrievalError("down")
        with self.assertRaises(RetrievalError):
            analyze("The Eiffel Tower was completed in 1889.", self.settings, search_fn=broken, save=False)

    def test_url_input_is_fetched(self):
        import src.pipeline as pl
        original = pl.fetch_page_text
        pl.fetch_page_text = lambda url, s: ("t", "The Eiffel Tower was completed in 1889 in Paris, France.")
        try:
            res = analyze("https://example.com/news", self.settings, search_fn=fake_search,
                          llm_fn=fake_llm_factory("Supported"), save=False)
        finally:
            pl.fetch_page_text = original
        self.assertIn("Eiffel", res.input_text)


class TestAggregate(unittest.TestCase):
    def cr(self, v, c=0.8):
        return ClaimResult(claim=Claim("C1", "x"), verdict=v, confidence=c)

    def test_rules(self):
        self.assertEqual(aggregate([self.cr(SUPPORTED), self.cr(SUPPORTED)])[0], SUPPORTED)
        self.assertEqual(aggregate([self.cr(SUPPORTED), self.cr(CONTRADICTED)])[0], CONTRADICTED)
        self.assertEqual(aggregate([self.cr(SUPPORTED), self.cr(INSUFFICIENT)])[0], INSUFFICIENT)
        self.assertIn("mixes", aggregate([self.cr(SUPPORTED), self.cr(CONTRADICTED)])[2])


if __name__ == "__main__":
    unittest.main()
