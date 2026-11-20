"""Rule-based claim extraction (check-worthy sentence detection).

No spaCy / NLTK / transformers: sentences are split with regular expressions
and scored with transparent linguistic features. Sentences that contain
numbers, dates, named entities and factual/reporting verbs score high;
questions, opinions and hedged speculation score low.
"""
from __future__ import annotations

import re

from .models import Claim
from .utils import clean_text

ABBREVIATIONS = ("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
                 "e.g", "i.e", "u.s", "u.k", "u.n", "no", "inc", "ltd", "co", "gen", "rep", "sen")

FACTUAL_VERBS = {
    "announced", "confirmed", "said", "says", "reported", "claimed", "claims",
    "killed", "died", "dies", "arrested", "banned", "approved", "launched",
    "found", "discovered", "caused", "causes", "cures", "cure", "increased",
    "decreased", "rose", "fell", "won", "lost", "signed", "passed", "declared",
    "revealed", "released", "ordered", "shut", "closed", "opened", "built",
    "invented", "landed", "elected", "resigned", "appointed", "attacked",
    "spread", "infected", "contains", "prevents", "leads", "led", "will", "has",
    "have", "had", "is", "are", "was", "were", "created", "proved", "linked",
}
INTENSIFIERS = {"first", "only", "never", "always", "all", "every", "million",
                "billion", "thousand", "crore", "lakh", "most", "largest", "highest", "no"}
OPINION_PATTERNS = re.compile(
    r"\b(i think|i believe|i feel|in my opinion|we believe|should|must|hopefully|"
    r"perhaps|maybe|might|could be|seems to|arguably|beautiful|terrible|amazing|awful)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d")
_YEAR_OR_MONTH = re.compile(
    r"\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b", re.IGNORECASE)
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z0-9])")

CHECK_WORTHY_THRESHOLD = 2.0


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that protects abbreviations and decimals."""
    text = clean_text(text)
    if not text:
        return []
    for abbr in ABBREVIATIONS:
        text = re.sub(rf"\b({re.escape(abbr)})\.", r"\1<DOT>", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def named_entities(sentence: str) -> list[str]:
    """Capitalised word sequences; a lone capitalised first word is ignored."""
    entities = []
    for m in _ENTITY.finditer(sentence):
        if m.start() == 0 and " " not in m.group():
            continue
        entities.append(m.group())
    return entities


def score_sentence(sentence: str) -> float:
    """Heuristic check-worthiness score (higher = more factual/verifiable)."""
    words = sentence.split()
    n = len(words)
    score = 0.0
    if n < 5:
        score -= 2.5
    elif n <= 45:
        score += 0.5
    else:
        score -= 1.0
    if _NUMBER.search(sentence):
        score += 2.0
    if _YEAR_OR_MONTH.search(sentence):
        score += 1.5
    score += min(len(named_entities(sentence)), 3) * 1.0
    tokens = set(re.findall(r"[a-z']+", sentence.lower()))
    if tokens & FACTUAL_VERBS:
        score += 1.5
    if tokens & INTENSIFIERS:
        score += 0.5
    score -= 1.5 * min(len(OPINION_PATTERNS.findall(sentence)), 2)
    if sentence.rstrip().endswith("?"):
        score -= 3.0
    return round(score, 2)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def extract_claims(text: str, max_claims: int = 3) -> list[Claim]:
    """Return up to ``max_claims`` check-worthy claims, in original order."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:  # headline or a single claim: verify it as-is
        return [Claim(id="C1", text=sentences[0], score=score_sentence(sentences[0]))]

    scored = [(i, s, score_sentence(s)) for i, s in enumerate(sentences) if len(s.split()) >= 4]
    if not scored:
        scored = [(0, sentences[0], score_sentence(sentences[0]))]
    ranked = sorted(scored, key=lambda t: t[2], reverse=True)

    chosen: list[tuple[int, str, float]] = []
    for item in ranked:
        if item[2] < CHECK_WORTHY_THRESHOLD and chosen:
            break
        if any(_jaccard(item[1], c[1]) > 0.8 for c in chosen):
            continue
        chosen.append(item)
        if len(chosen) >= max(1, max_claims):
            break
    chosen.sort(key=lambda t: t[0])  # restore reading order
    return [Claim(id=f"C{n}", text=s, score=sc) for n, (_, s, sc) in enumerate(chosen, 1)]
