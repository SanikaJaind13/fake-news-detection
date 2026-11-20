import json
import unittest

from src.config import Settings
from src.errors import ConfigError, LLMError
from src.models import CONTRADICTED, INSUFFICIENT, SUPPORTED, Claim, Passage
from src.verdict import (build_messages, call_groq, judge_claim, normalise_verdict,
                         parse_llm_json)


def passages():
    return [Passage("E1", "Text one about the claim.", "https://a.com", "A", "a.com", 0.6, 0.9, "Wire"),
            Passage("E2", "Text two about the claim.", "https://b.com", "B", "b.com", 0.4, 0.5, "Unrated")]


CLAIM = Claim("C1", "Some claim to verify", 3.0)
SETTINGS = Settings(groq_api_key="k", tavily_api_key="t")


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, "err"

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def post(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


class TestParsing(unittest.TestCase):
    def test_parse_plain_fenced_and_wrapped(self):
        self.assertEqual(parse_llm_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_llm_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_llm_json('Here you go: {"a": 1} thanks'), {"a": 1})

    def test_parse_failure(self):
        with self.assertRaises(LLMError):
            parse_llm_json("no json at all")

    def test_normalise_verdict(self):
        self.assertEqual(normalise_verdict("supported"), SUPPORTED)
        self.assertEqual(normalise_verdict("CONTRADICTED"), CONTRADICTED)
        self.assertEqual(normalise_verdict("Insufficient Evidence"), INSUFFICIENT)
        self.assertEqual(normalise_verdict("¯\\_(ツ)_/¯"), INSUFFICIENT)


class TestJudge(unittest.TestCase):
    def test_no_passages_skips_llm(self):
        def boom(*a):
            raise AssertionError("LLM must not be called")
        r = judge_claim(CLAIM, [], SETTINGS, llm_fn=boom)
        self.assertEqual(r.verdict, INSUFFICIENT)

    def test_valid_supported(self):
        raw = json.dumps({"verdict": "Supported", "confidence": 0.9, "reasoning": "See [E1].",
                          "supporting": ["E1"], "contradicting": [], "key_points": ["x"]})
        r = judge_claim(CLAIM, passages(), SETTINGS, llm_fn=lambda m, s: raw)
        self.assertEqual(r.verdict, SUPPORTED)
        self.assertEqual([p.eid for p in r.supporting], ["E1"])
        self.assertEqual([p.eid for p in r.other], ["E2"])

    def test_hallucinated_citation_downgrades_verdict(self):
        raw = json.dumps({"verdict": "Contradicted", "confidence": 0.95, "reasoning": "See E9.",
                          "supporting": [], "contradicting": ["E9"]})
        r = judge_claim(CLAIM, passages(), SETTINGS, llm_fn=lambda m, s: raw)
        self.assertEqual(r.verdict, INSUFFICIENT)
        self.assertLessEqual(r.confidence, 0.4)
        self.assertEqual(r.contradicting, [])

    def test_confidence_clamped_and_bad_values(self):
        raw = json.dumps({"verdict": "Supported", "confidence": 7, "supporting": ["e1"]})
        r = judge_claim(CLAIM, passages(), SETTINGS, llm_fn=lambda m, s: raw)
        self.assertEqual(r.confidence, 1.0)
        raw = json.dumps({"verdict": "Supported", "confidence": "high", "supporting": ["E2"]})
        r = judge_claim(CLAIM, passages(), SETTINGS, llm_fn=lambda m, s: raw)
        self.assertEqual(r.confidence, 0.0)

    def test_prompt_contains_claim_and_evidence_ids(self):
        msgs = build_messages(CLAIM, passages())
        self.assertIn("Some claim to verify", msgs[1]["content"])
        self.assertIn("[E2]", msgs[1]["content"])


class TestGroqClient(unittest.TestCase):
    ok = {"choices": [{"message": {"content": "{}"}}]}

    def test_success(self):
        s = FakeSession([FakeResp(200, self.ok)])
        self.assertEqual(call_groq([], SETTINGS, session=s), "{}")

    def test_missing_or_rejected_key(self):
        with self.assertRaises(ConfigError):
            call_groq([], Settings(), session=FakeSession([]))
        with self.assertRaises(ConfigError):
            call_groq([], SETTINGS, session=FakeSession([FakeResp(401)]))

    def test_retries_on_rate_limit(self):
        s = FakeSession([FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, self.ok)])
        self.assertEqual(call_groq([], SETTINGS, session=s), "{}")
        self.assertEqual(s.calls, 2)

    def test_gives_up_after_retries(self):
        s = FakeSession([FakeResp(500, headers={"Retry-After": "0"})] * 3)
        with self.assertRaises(LLMError):
            call_groq([], SETTINGS, session=s)


if __name__ == "__main__":
    unittest.main()
