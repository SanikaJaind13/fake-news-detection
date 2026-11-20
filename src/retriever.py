"""Evidence retrieval: Tavily web search + Requests/BeautifulSoup page extraction."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

from .config import Settings
from .credibility import rate_domain
from .errors import ConfigError, RetrievalError
from .models import Document
from .utils import clean_text, extract_domain, truncate

TAVILY_URL = "https://api.tavily.com/search"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 EvidenceCheckerBot/1.0"
)
_JUNK_TAGS = ["script", "style", "noscript", "nav", "footer", "header",
              "aside", "form", "svg", "iframe", "button"]
MIN_PAGE_CHARS = 200


def search_web(query: str, settings: Settings, session=requests) -> list[dict]:
    """Call the Tavily Search API and return its raw result dictionaries."""
    if not settings.tavily_api_key:
        raise ConfigError("TAVILY_API_KEY is missing.")
    payload = {
        "query": query[:390],  # Tavily limits query length
        "search_depth": "basic",
        "max_results": settings.results_per_claim,
        "include_answer": False,
    }
    headers = {"Authorization": f"Bearer {settings.tavily_api_key}"}
    try:
        resp = session.post(TAVILY_URL, json=payload, headers=headers,
                            timeout=settings.request_timeout)
    except requests.Timeout as exc:
        raise RetrievalError("Tavily search timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise RetrievalError(f"Could not reach Tavily: {exc}") from exc

    if resp.status_code in (401, 403):
        raise ConfigError("Tavily rejected the API key. Check TAVILY_API_KEY.")
    if resp.status_code == 429:
        raise RetrievalError("Tavily rate limit reached. Wait a moment and retry.")
    if resp.status_code != 200:
        raise RetrievalError(f"Tavily error {resp.status_code}: {resp.text[:150]}")
    try:
        return resp.json().get("results", [])
    except ValueError as exc:
        raise RetrievalError("Tavily returned invalid JSON.") from exc


def extract_text_from_html(html: str, max_chars: int = 20000) -> tuple[str, str]:
    """Return (title, main_text) from raw HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text()) if soup.title else ""
    for tag in soup(_JUNK_TAGS):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [clean_text(p.get_text(" ")) for p in container.find_all(["p", "li"])]
    paragraphs = [p for p in paragraphs if len(p) >= 40]
    text = " ".join(paragraphs) if paragraphs else clean_text(container.get_text(" "))
    return title, truncate(text, max_chars)


def fetch_page_text(url: str, settings: Settings, session=requests) -> tuple[str, str]:
    """Download a page and return (title, text). Raises RetrievalError."""
    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT},
                           timeout=settings.request_timeout)
    except requests.RequestException as exc:
        raise RetrievalError(f"Could not download {url}: {exc}") from exc
    if resp.status_code != 200:
        raise RetrievalError(f"{url} returned HTTP {resp.status_code}.")
    ctype = resp.headers.get("Content-Type", "text/html").lower()
    if "html" not in ctype and "text" not in ctype:
        raise RetrievalError(f"{url} is not an HTML page ({ctype}).")
    return extract_text_from_html(resp.text[:2_000_000], settings.max_page_chars)


def _build_document(result: dict, settings: Settings, session) -> Document | None:
    url = result.get("url", "")
    if not url:
        return None
    snippet = clean_text(result.get("content", ""))
    title = clean_text(result.get("title", "")) or url
    text = ""
    try:
        page_title, text = fetch_page_text(url, settings, session)
        title = title if result.get("title") else (page_title or title)
    except RetrievalError:
        pass  # paywall / bot-block / timeout: fall back to the Tavily snippet
    if len(text) < MIN_PAGE_CHARS:
        text = snippet or text
    if not text:
        return None
    domain = extract_domain(url)
    score, label = rate_domain(domain)
    # Tavily's snippet is query-focused, so keep it in front of the page text.
    if snippet and snippet not in text:
        text = f"{snippet} {text}"
    return Document(url=url, title=title, domain=domain, text=text,
                    credibility=score, credibility_label=label)


def retrieve_documents(query: str, settings: Settings, session=requests) -> list[Document]:
    """Search the web and download the text of the top results in parallel."""
    results = search_web(query, settings, session)
    if not results:
        return []
    with ThreadPoolExecutor(max_workers=min(5, len(results))) as pool:
        docs = list(pool.map(lambda r: _build_document(r, settings, session), results))
    seen, unique = set(), []
    for d in docs:
        if d and d.url not in seen:
            seen.add(d.url)
            unique.append(d)
    return unique
