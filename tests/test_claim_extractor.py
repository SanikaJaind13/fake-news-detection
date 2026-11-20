import unittest

from src.claim_extractor import extract_claims, score_sentence, split_sentences


class TestSentenceSplitting(unittest.TestCase):
    def test_abbreviations_and_decimals_not_split(self):
        text = "Dr. Smith said inflation fell to 4.5 percent. The U.S. economy grew."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])


class TestClaimExtraction(unittest.TestCase):
    ARTICLE = (
        "The government announced on 12 March 2024 that petrol prices fell by 5 percent. "
        "Honestly, I think this is wonderful news for everyone. "
        "Is this really going to last? "
        "Finance Minister Anita Rao confirmed that 2 million households will receive a subsidy. "
        "We should all be happy."
    )

    def test_headline_returned_as_single_claim(self):
        claims = extract_claims("NASA confirms water found on the Moon in 2020")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].id, "C1")

    def test_picks_factual_sentences_and_drops_opinions_questions(self):
        texts = [c.text for c in extract_claims(self.ARTICLE, max_claims=3)]
        self.assertTrue(any("petrol prices fell" in t for t in texts))
        self.assertTrue(any("subsidy" in t for t in texts))
        self.assertFalse(any("Is this really" in t for t in texts))
        self.assertFalse(any("I think" in t for t in texts))

    def test_respects_max_claims_and_order(self):
        claims = extract_claims(self.ARTICLE, max_claims=1)
        self.assertEqual(len(claims), 1)
        claims = extract_claims(self.ARTICLE, max_claims=2)
        self.assertEqual([c.id for c in claims], ["C1", "C2"])
        self.assertLess(self.ARTICLE.index(claims[0].text), self.ARTICLE.index(claims[1].text))

    def test_always_returns_something_for_nonempty_text(self):
        self.assertEqual(len(extract_claims("Wow. Nice. Cool day here today.")), 1)

    def test_scoring_prefers_facts(self):
        factual = score_sentence("The WHO declared COVID-19 a pandemic in March 2020.")
        opinion = score_sentence("I think the weather should be beautiful tomorrow.")
        self.assertGreater(factual, opinion)


if __name__ == "__main__":
    unittest.main()
