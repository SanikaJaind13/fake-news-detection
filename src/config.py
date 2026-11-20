"""Central configuration. Values come from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    tavily_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    db_path: str = str(BASE_DIR / "data" / "history.db")
    max_claims: int = 3            # claims verified per submission
    results_per_claim: int = 5     # web pages retrieved per claim
    passages_per_claim: int = 6    # evidence passages sent to the LLM
    request_timeout: int = 12      # seconds, per HTTP request
    max_page_chars: int = 20000    # cap on text kept from each page

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing

    def with_overrides(self, **kwargs) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v not in (None, "")}
        return replace(self, **clean)


def load_settings(**overrides) -> Settings:
    """Read settings from the environment, then apply non-empty overrides."""
    base = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        or "openai/gpt-oss-120b",
        db_path=os.getenv("DB_PATH", str(BASE_DIR / "data" / "history.db")),
        max_claims=_int_env("MAX_CLAIMS", 3),
        results_per_claim=_int_env("RESULTS_PER_CLAIM", 5),
        passages_per_claim=_int_env("PASSAGES_PER_CLAIM", 6),
        request_timeout=_int_env("REQUEST_TIMEOUT", 12),
    )
    return base.with_overrides(**overrides)
