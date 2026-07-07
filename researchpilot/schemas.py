"""Typed data structures passed between agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SubQuestion:
    id: int
    text: str
    rationale: str = ""


@dataclass
class Citation:
    uri: str
    title: str = ""


@dataclass
class Evidence:
    sub_question_id: int
    answer: str
    citations: list[Citation] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    attempt: int = 1

    @property
    def grounded(self) -> bool:
        return len(self.citations) > 0


Verdict = Literal["supported", "unsupported", "uncertain"]


@dataclass
class ClaimVerdict:
    claim: str
    verdict: Verdict
    reason: str = ""
    confidence: float = 0.0


@dataclass
class Critique:
    sub_question_id: int
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    overall_confidence: float = 0.0
    refined_query: str | None = None   # critic's suggested re-search query

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict == "unsupported"]

    @property
    def passed(self) -> bool:
        return not self.unsupported and self.overall_confidence > 0


@dataclass
class AgentEvent:
    """Emitted by the orchestrator so the CLI can render a live trace."""
    agent: str        # planner | researcher | critic | synthesizer | orchestrator
    action: str       # e.g. "plan", "search", "flag", "retry", "accept", "score"
    detail: str = ""
    data: dict | None = None


@dataclass
class Report:
    question: str
    summary: str
    groundedness_score: float
    sub_questions: list[SubQuestion] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    api_calls_used: int = 0
    corrective_passes: int = 0

    @property
    def all_citations(self) -> list[Citation]:
        seen, out = set(), []
        for ev in self.evidence:
            for c in ev.citations:
                if c.uri not in seen:
                    seen.add(c.uri)
                    out.append(c)
        return out
