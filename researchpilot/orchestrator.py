"""Orchestrator: the agentic control loop.

plan -> (research -> critique -> [refine & retry]) per sub-question
     -> synthesize -> score -> [corrective pass if groundedness too low]

All decisions (accept / retry / correct) are made here from structured critic
output — never from free-form model text.
"""
from __future__ import annotations

from typing import Callable

from .agents.critic import Critic
from .agents.planner import Planner
from .agents.researcher import Researcher
from .agents.synthesizer import Synthesizer
from .config import Settings
from .llm import BudgetExceeded
from .schemas import AgentEvent, Critique, Evidence, Report


class Orchestrator:
    def __init__(self, llm, settings: Settings | None = None,
                 on_event: Callable[[AgentEvent], None] | None = None):
        self.s = settings or Settings()
        self.llm = llm
        self.planner = Planner(llm, self.s.max_sub_questions)
        self.researcher = Researcher(llm)
        self.critic = Critic(llm)
        self.synthesizer = Synthesizer(llm)
        self._emit = on_event or (lambda e: None)

    # ------------------------------------------------------------------ run
    def run(self, question: str) -> Report:
        self._emit(AgentEvent("orchestrator", "start", question))

        sub_qs = self.planner.plan(question)
        self._emit(AgentEvent("planner", "plan",
                              f"{len(sub_qs)} sub-questions",
                              {"sub_questions": [s.text for s in sub_qs]}))

        evidence: list[Evidence] = []
        critiques: list[Critique] = []
        for sq in sub_qs:
            try:
                ev, cr = self._research_until_accepted(sq)
            except BudgetExceeded:
                self._emit(AgentEvent("orchestrator", "budget",
                                      "API budget exhausted — stopping research"))
                break
            evidence.append(ev)
            critiques.append(cr)

        report = self._synthesize_and_score(question, sub_qs, evidence, critiques)

        # Report-level self-correction: one corrective pass on the weakest link.
        if (report.groundedness_score < self.s.groundedness_threshold
                and evidence and self._budget_left(4)):
            report = self._corrective_pass(report, sub_qs, evidence, critiques)

        report.api_calls_used = self.llm.calls_made
        self._emit(AgentEvent("orchestrator", "done",
                              f"groundedness={report.groundedness_score} "
                              f"api_calls={report.api_calls_used}"))
        return report

    # ------------------------------------------------- per-sub-question loop
    def _research_until_accepted(self, sq) -> tuple[Evidence, Critique]:
        refined = None
        best: tuple[Evidence, Critique] | None = None
        for attempt in range(1, self.s.max_research_retries + 2):
            self._emit(AgentEvent("researcher", "search",
                                  f"[SQ{sq.id}] attempt {attempt}: "
                                  f"{refined or sq.text}"))
            ev = self.researcher.research(sq, refined_query=refined, attempt=attempt)
            self._emit(AgentEvent("researcher", "evidence",
                                  f"[SQ{sq.id}] {len(ev.citations)} citations, "
                                  f"{len(ev.search_queries)} searches"))
            cr = self.critic.review(sq, ev)
            for v in cr.unsupported:
                self._emit(AgentEvent("critic", "flag",
                                      f"[SQ{sq.id}] unsupported: {v.claim[:80]}"))
            self._emit(AgentEvent("critic", "verdict",
                                  f"[SQ{sq.id}] confidence={cr.overall_confidence:.2f}"))

            if best is None or cr.overall_confidence > best[1].overall_confidence:
                best = (ev, cr)
            if cr.overall_confidence >= self.s.confidence_threshold and not cr.unsupported:
                self._emit(AgentEvent("orchestrator", "accept",
                                      f"[SQ{sq.id}] accepted on attempt {attempt}"))
                return ev, cr
            if attempt <= self.s.max_research_retries and self._budget_left(2):
                refined = cr.refined_query or f"{sq.text} authoritative source statistics"
                self._emit(AgentEvent("orchestrator", "retry",
                                      f"[SQ{sq.id}] below threshold — re-researching "
                                      f"with: {refined[:80]}"))
            else:
                break
        self._emit(AgentEvent("orchestrator", "accept",
                              f"[SQ{sq.id}] keeping best attempt "
                              f"(confidence={best[1].overall_confidence:.2f})"))
        return best

    # ------------------------------------------------------------ synthesis
    def _synthesize_and_score(self, question, sub_qs, evidence, critiques) -> Report:
        self._emit(AgentEvent("synthesizer", "write", "drafting final answer"))
        try:
            summary = self.synthesizer.synthesize(question, sub_qs, evidence, critiques)
        except BudgetExceeded:
            self._emit(AgentEvent("synthesizer", "budget",
                                  "budget exhausted — falling back to raw evidence"))
            summary = "\n".join(e.answer for e in evidence) or \
                      "Budget exhausted before any evidence was gathered."
        score = self.synthesizer.groundedness(evidence, critiques)
        self._emit(AgentEvent("synthesizer", "score", f"groundedness={score}"))
        caveats = [f"{v.claim} — {v.reason}"
                   for c in critiques for v in c.verdicts]
        return Report(question=question, summary=summary, groundedness_score=score,
                      sub_questions=sub_qs, evidence=evidence,
                      critiques=critiques, caveats=caveats)

    def _corrective_pass(self, report, sub_qs, evidence, critiques) -> Report:
        weakest = min(critiques, key=lambda c: c.overall_confidence)
        sq = next(s for s in sub_qs if s.id == weakest.sub_question_id)
        self._emit(AgentEvent("orchestrator", "correct",
                              f"groundedness {report.groundedness_score} < "
                              f"{self.s.groundedness_threshold} — re-researching SQ{sq.id}"))
        try:
            ev, cr = self._research_until_accepted(sq)
        except BudgetExceeded:
            return report
        idx = next(i for i, e in enumerate(evidence)
                   if e.sub_question_id == sq.id)
        if cr.overall_confidence > weakest.overall_confidence:
            evidence[idx], critiques[critiques.index(weakest)] = ev, cr
        new_report = self._synthesize_and_score(report.question, sub_qs,
                                                evidence, critiques)
        new_report.corrective_passes = report.corrective_passes + 1
        return new_report

    def _budget_left(self, calls_needed: int) -> bool:
        return self.llm.max_api_calls - self.llm.calls_made >= calls_needed
