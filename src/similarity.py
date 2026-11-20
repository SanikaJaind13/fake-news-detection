"""Lightweight semantic matching with TF-IDF + cosine similarity (scikit-learn).

Pages are split into overlapping sentence windows ("passages"). Each passage
is scored against the claim with a hybrid score:

    score = 0.75 * cosine(TF-IDF(claim), TF-IDF(passage))
          + 0.25 * key-term coverage (numbers and named entities of the claim
                   that also appear in the passage)

The key-term component stops a passage about the same topic but with a
different number/name from scoring too highly.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .claim_extractor import split_sentences
from .models import Document, Passage
from .utils import truncate

COSINE_WEIGHT = 0.75
KEYTERM_WEIGHT = 0.25
MIN_SCORE = 0.05
MAX_PASSAGES_PER_DOC = 2


def split_passages(text: str, window: int = 3, stride: int = 2,
                   min_chars: int = 40, max_chars: int = 700) -> list[str]:
    """Overlapping windows of ``window`` sentences, moving ``stride`` at a time."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    passages = []
    for start in range(0, len(sentences), stride):
        chunk = " ".join(sentences[start:start + window]).strip()
        if len(chunk) >= min_chars:
            passages.append(truncate(chunk, max_chars))
        if start + window >= len(sentences):
            break
    return passages


def key_terms(claim: str) -> set[str]:
    """Numbers and capitalised words (likely entities) from the claim."""
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", claim)
    words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", claim)
    first = claim.split()[0] if claim.split() else ""
    terms = {n.replace(",", "") for n in numbers}
    terms |= {w.lower() for w in words if w != first or len(words) == 1}
    return terms


def _coverage(terms: set[str], passage: str) -> float:
    if not terms:
        return 0.0
    low = passage.lower().replace(",", "")
    return sum(1 for t in terms if t in low) / len(terms)


def rank_passages(claim: str, documents: list[Document], top_k: int = 6) -> list[Passage]:
    """Return the ``top_k`` passages most relevant to ``claim`` with E1..En ids."""
    candidates: list[tuple[Document, str]] = []
    for doc in documents:
        seen = set()
        for p in split_passages(doc.text):
            if p not in seen:
                seen.add(p)
                candidates.append((doc, p))
    if not candidates:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 sublinear_tf=True, strip_accents="unicode")
    try:
        matrix = vectorizer.fit_transform([claim] + [p for _, p in candidates])
    except ValueError:  # empty vocabulary (e.g. only stop-words)
        return []
    cosines = cosine_similarity(matrix[0:1], matrix[1:])[0]

    terms = key_terms(claim)
    scored = []
    for (doc, text), cos in zip(candidates, cosines):
        if terms:
            score = COSINE_WEIGHT * cos + KEYTERM_WEIGHT * _coverage(terms, text)
        else:
            score = float(cos)
        scored.append((float(score), doc, text))
    scored.sort(key=lambda t: t[0], reverse=True)

    per_doc: dict[str, int] = {}
    passages: list[Passage] = []
    for score, doc, text in scored:
        if score < MIN_SCORE or len(passages) >= top_k:
            break
        if per_doc.get(doc.url, 0) >= MAX_PASSAGES_PER_DOC:
            continue  # keep evidence diverse across sources
        per_doc[doc.url] = per_doc.get(doc.url, 0) + 1
        passages.append(Passage(
            eid=f"E{len(passages) + 1}", text=text, url=doc.url, title=doc.title,
            domain=doc.domain, similarity=round(score, 3),
            credibility=doc.credibility, credibility_label=doc.credibility_label))
    return passages
