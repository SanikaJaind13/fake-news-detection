import unittest

from src.models import Document
from src.similarity import key_terms, rank_passages, split_passages


def doc(url, text, domain="example.com"):
    return Document(url=url, title=url, domain=domain, text=text,
                    credibility=0.5, credibility_label="Unrated source")


class TestSimilarity(unittest.TestCase):
    def test_split_passages_overlap(self):
        text = " ".join(f"This is sentence number {i} of the sample article text." for i in range(7))
        passages = split_passages(text)
        self.assertGreaterEqual(len(passages), 3)
        self.assertTrue(all(len(p) >= 40 for p in passages))

    def test_key_terms(self):
        terms = key_terms("Chandrayaan-3 landed near the Moon south pole in 2023")
        self.assertIn("2023", terms)
        self.assertIn("moon", terms)

    def test_relevant_passage_ranks_first_and_ids_assigned(self):
        claim = "India landed Chandrayaan-3 near the Moon's south pole in August 2023"
        relevant = doc("https://a.com", "ISRO's Chandrayaan-3 landed near the lunar south pole "
                       "in August 2023. India became the fourth country to land on the Moon.")
        unrelated = doc("https://b.com", "The recipe needs two cups of flour and a pinch of salt. "
                        "Bake the bread for forty minutes until golden brown.")
        passages = rank_passages(claim, [unrelated, relevant], top_k=3)
        self.assertTrue(passages)
        self.assertEqual(passages[0].url, "https://a.com")
        self.assertEqual(passages[0].eid, "E1")
        scores = [p.similarity for p in passages]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_entity_mismatch_scores_lower(self):
        claim = "The company laid off 5000 workers in 2022"
        match = doc("https://m.com", "The company laid off 5000 workers in 2022 after losses.")
        mismatch = doc("https://n.com", "The company hired 300 workers in 2019 after growth.")
        passages = rank_passages(claim, [mismatch, match], top_k=2)
        self.assertEqual(passages[0].url, "https://m.com")

    def test_empty_inputs(self):
        self.assertEqual(rank_passages("some claim here", []), [])
        self.assertEqual(rank_passages("the of and", [doc("u", "Some text that is long enough to count here.")]), [])

    def test_max_two_passages_per_document(self):
        long_text = " ".join(f"Chandrayaan-3 landed on the Moon in 2023 fact {i}." for i in range(30))
        passages = rank_passages("Chandrayaan-3 landed on the Moon in 2023", [doc("https://x.com", long_text)], top_k=6)
        self.assertLessEqual(len(passages), 2)


if __name__ == "__main__":
    unittest.main()
