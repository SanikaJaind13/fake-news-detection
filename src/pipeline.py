"""End-to-end orchestration:

input -> claim extraction -> web retrieval -> TF-IDF evidence ranking
      -> Groq RAG verdict per claim -> aggregation -> SQLite history
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from . import database
from .claim_extractor import extract_claims
from .config import Settings
from .errors import ConfigError, FakeNewsError, InputError, RetrievalError
from .models import (CONTRADICTED, INSUFFICIENT, SUPPORTED, AnalysisResult, ClaimResult)
from .retriever import fetch_page_text, retrieve_documents
from .similarity import rank_passages
from .utils import clean_text, is_url, truncate
from .verdict import call_groq, judge_claim

MIN_INPUT_CHARS = 15
MAX_INPUT_CHARS = 15000
ProgressFn = Callable[[float, str], None]


def aggregate(results: list[ClaimResult]) -> tuple[str, float, str]:
    """Combine per-claim verdicts into an overall verdict, confidence, summary."""
    n = len(results)
    counts = {v: sum(r.verdict == v for r in results) for v in (SUPPORTED, CONTRADICTED, INSUFFICIENT)}
    if counts[CONTRADICTED]:
        overall = CONTRADICTED
    elif counts[SUPPORTED] == n:
        overall = SUPPORTED
    else:
        overall = INSUFFICIENT
    decisive = [r.confidence for r in results if r.verdict == overall] or [0.0]
    confidence = round(sum(decisive) / len(decisive), 2)
    summary = (f"{n} claim(s) checked: {counts[SUPPORTED]} supported, "
               f"{counts[CONTRADICTED]} contradicted, {counts[INSUFFICIENT]} with insufficient evidence.")
    if counts[CONTRADICTED] and counts[SUPPORTED]:
        summary += " The submission mixes accurate and inaccurate claims."
    return overall, confidence, summary


def analyze(text: str, settings: Settings, progress: ProgressFn | None = None,
            search_fn=retrieve_documents, llm_fn=call_groq, save: bool = True,
            user_id: int = 1) -> AnalysisResult:
    """Run the full fact-checking pipeline. ``search_fn``/``llm_fn`` can be
    swapped for fakes in tests."""
    report = progress or (lambda frac, msg: None)

    text = clean_text(text)
    if len(text) < MIN_INPUT_CHARS:
        raise InputError("Please enter a headline, claim or article (at least a full sentence).")
    missing = settings.missing_keys()
    if missing:
        raise ConfigError("Missing API key(s): " + ", ".join(missing)
                          + ". Add them to your .env file or the sidebar.")

    if is_url(text):
        report(0.03, "Downloading the article from the URL...")
        _, page_text = fetch_page_text(text, settings)
        if len(page_text) < MIN_INPUT_CHARS:
            raise InputError("Could not read any article text from that URL.")
        text = page_text
    text = truncate(text, MAX_INPUT_CHARS)

    report(0.08, "Extracting check-worthy claims...")
    claims = extract_claims(text, settings.max_claims)
    if not claims:
        raise InputError("No verifiable claims were found in the text.")

    results: list[ClaimResult] = []
    for idx, claim in enumerate(claims):
        base = 0.1 + 0.85 * idx / len(claims)
        step = 0.85 / len(claims)
        try:
            report(base, f"[{claim.id}] Searching the web for evidence...")
            docs = search_fn(claim.text, settings)
            report(base + step * 0.4, f"[{claim.id}] Ranking evidence (TF-IDF)...")
            passages = rank_passages(claim.text, docs, settings.passages_per_claim)
            report(base + step * 0.65, f"[{claim.id}] Asking the LLM to weigh the evidence...")
            results.append(judge_claim(claim, passages, settings, len(docs), llm_fn))
        except ConfigError:
            raise  # a bad API key affects every claim: stop and tell the user
        except FakeNewsError as exc:
            results.append(ClaimResult(
                claim=claim, verdict=INSUFFICIENT, confidence=0.0,
                reasoning="This claim could not be verified because of an error.",
                error=str(exc)))

    if all(r.error for r in results):
        raise RetrievalError("Every claim failed: " + results[0].error)

    overall, confidence, summary = aggregate(results)
    result = AnalysisResult(
        input_text=text, claim_results=results, overall_verdict=overall,
        overall_confidence=confidence, summary=summary,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    if save:
        try:
            result.id = database.save_analysis(settings.db_path, result.to_dict(), user_id)
        except Exception:  # history is a convenience; never lose the verdict
            result.id = None
    report(1.0, "Done.")
    return result
