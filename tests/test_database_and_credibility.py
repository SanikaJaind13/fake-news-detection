import os
import tempfile
import unittest

from src import database
from src.credibility import rate_domain
from src.utils import extract_domain, is_url


def sample(verdict="Supported"):
    return {"created_at": "2025-01-01 10:00:00 UTC", "input_text": "Some claim text",
            "overall_verdict": verdict, "overall_confidence": 0.8,
            "summary": "1 claim(s) checked.", "claim_results": []}


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "nested", "h.db")
        self.user_id = 42

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_list_get_delete(self):
        a = database.save_analysis(self.db, sample("Supported"), self.user_id)
        b = database.save_analysis(self.db, sample("Contradicted"), self.user_id)
        rows = database.list_history(self.db, self.user_id)
        self.assertEqual([r["id"] for r in rows], [b, a])  # newest first
        self.assertEqual(database.get_analysis(self.db, a, self.user_id)["overall_verdict"], "Supported")
        self.assertTrue(database.delete_analysis(self.db, a, self.user_id))
        self.assertIsNone(database.get_analysis(self.db, a, self.user_id))
        self.assertFalse(database.delete_analysis(self.db, 999, self.user_id))

    def test_clear(self):
        database.save_analysis(self.db, sample(), self.user_id)
        database.save_analysis(self.db, sample(), self.user_id)
        self.assertEqual(database.clear_history(self.db, self.user_id), 2)
        self.assertEqual(database.list_history(self.db, self.user_id), [])

    def test_unicode_roundtrip(self):
        s = sample()
        s["input_text"] = "भारत ने चंद्रयान-3 उतारा ✔"
        i = database.save_analysis(self.db, s, self.user_id)
        self.assertEqual(database.get_analysis(self.db, i, self.user_id)["input_text"], s["input_text"])

    def test_user_isolation(self):
        """Test that users only see their own history."""
        user2 = 99
        database.save_analysis(self.db, sample("Supported"), self.user_id)
        database.save_analysis(self.db, sample("Contradicted"), user2)
        rows1 = database.list_history(self.db, self.user_id)
        rows2 = database.list_history(self.db, user2)
        self.assertEqual(len(rows1), 1)
        self.assertEqual(len(rows2), 1)
        self.assertEqual(rows1[0]["overall_verdict"], "Supported")
        self.assertEqual(rows2[0]["overall_verdict"], "Contradicted")
        # User 1 cannot access user 2's analysis
        self.assertIsNone(database.get_analysis(self.db, rows2[0]["id"], self.user_id))
        # User 1 cannot delete user 2's analysis
        self.assertFalse(database.delete_analysis(self.db, rows2[0]["id"], self.user_id))


class TestCredibilityAndUtils(unittest.TestCase):
    def test_ratings(self):
        self.assertGreater(rate_domain("reuters.com")[0], 0.9)
        self.assertGreaterEqual(rate_domain("data.cdc.gov")[0], 0.9)
        self.assertGreaterEqual(rate_domain("iitb.ac.in")[0], 0.9)
        self.assertEqual(rate_domain("bbc.co.uk")[0], 0.8)
        self.assertLess(rate_domain("old.reddit.com")[0], 0.5)
        self.assertEqual(rate_domain("randomblog.xyz")[0], 0.5)

    def test_url_helpers(self):
        self.assertEqual(extract_domain("https://www.BBC.com/news/x?y=1"), "bbc.com")
        self.assertTrue(is_url("https://example.com/a"))
        self.assertFalse(is_url("https://example.com is a site"))


if __name__ == "__main__":
    unittest.main()
