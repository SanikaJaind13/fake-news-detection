"""Small helpers used across modules."""
from __future__ import annotations

import re
from urllib.parse import urlparse

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def is_url(text: str) -> bool:
    return bool(_URL_RE.match(text.strip()))


def extract_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    host = host.split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def clean_text(text: str) -> str:
    """Collapse whitespace and drop control characters."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
