"""Rule-based source credibility prior (a transparent heuristic, not a model).

The score is NOT used to decide truth. It is (a) shown to the user and
(b) given to the LLM as context so it can weigh conflicting sources.
"""
from __future__ import annotations

FACT_CHECKERS = {
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
    "reuters.com", "apnews.com", "afp.com", "boomlive.in", "altnews.in",
    "factly.in", "vishvasnews.com", "leadstories.com",
}
INSTITUTIONS = {
    "who.int", "un.org", "nasa.gov", "cdc.gov", "nih.gov", "worldbank.org",
    "imf.org", "europa.eu", "isro.gov.in", "pib.gov.in", "nature.com",
    "science.org", "thelancet.com", "bmj.com", "britannica.com",
}
MAJOR_NEWS = {
    "bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "aljazeera.com", "npr.org", "bloomberg.com", "ft.com",
    "wsj.com", "economist.com", "thehindu.com", "indianexpress.com",
    "hindustantimes.com", "timesofindia.indiatimes.com", "ndtv.com", "dw.com",
    "cnn.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com",
    "scientificamerican.com", "livemint.com",
}
LOW_RELIABILITY = {
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "medium.com", "blogspot.com", "wordpress.com",
    "substack.com", "pinterest.com", "youtube.com", "t.me", "telegram.org",
}
_GOV_EDU_SUFFIXES = (".gov", ".gov.in", ".nic.in", ".edu", ".ac.in", ".ac.uk", ".edu.au")


def _matches(domain: str, group: set[str]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in group)


def rate_domain(domain: str) -> tuple[float, str]:
    """Return (score in [0,1], human-readable label) for a domain."""
    d = (domain or "").lower()
    if _matches(d, FACT_CHECKERS):
        return 0.95, "Fact-checker / wire service"
    if _matches(d, INSTITUTIONS) or d.endswith(_GOV_EDU_SUFFIXES):
        return 0.90, "Government / academic / institution"
    if _matches(d, MAJOR_NEWS):
        return 0.80, "Established news outlet"
    if _matches(d, {"wikipedia.org"}):
        return 0.60, "Encyclopedia (community edited)"
    if _matches(d, LOW_RELIABILITY):
        return 0.30, "Social media / user-generated"
    return 0.50, "Unrated source"
