"""Plain dataclasses shared by every module (easy to serialise to JSON)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SUPPORTED = "Supported"
CONTRADICTED = "Contradicted"
INSUFFICIENT = "Insufficient Evidence"
VERDICTS = (SUPPORTED, CONTRADICTED, INSUFFICIENT)


@dataclass
class Claim:
    id: str
    text: str
    score: float = 0.0  # check-worthiness score from the extractor


@dataclass
class Document:
    """A retrieved web page."""
    url: str
    title: str
    domain: str
    text: str
    credibility: float = 0.5
    credibility_label: str = "Unrated source"


@dataclass
class Passage:
    """A short chunk of a document, ranked against one claim."""
    eid: str                # evidence id used for citations, e.g. "E1"
    text: str
    url: str
    title: str
    domain: str
    similarity: float
    credibility: float
    credibility_label: str


@dataclass
class ClaimResult:
    claim: Claim
    verdict: str = INSUFFICIENT
    confidence: float = 0.0
    reasoning: str = ""
    supporting: list[Passage] = field(default_factory=list)
    contradicting: list[Passage] = field(default_factory=list)
    other: list[Passage] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    n_documents: int = 0
    error: str | None = None


@dataclass
class AnalysisResult:
    input_text: str
    claim_results: list[ClaimResult]
    overall_verdict: str
    overall_confidence: float
    summary: str
    created_at: str
    id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)
