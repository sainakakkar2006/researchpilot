"""Synthesizer agent: writes the final report and computes groundedness."""
from __future__ import annotations

from ..schemas import Critique, Evidence, SubQuestion

PROMPT = """You are the synthesis agent of a research system. Write a clear,
5-8 sentence answer to the main question using ONLY the verified evidence
below. Treat flagged claims as caveats: either omit them or present them
with explicit uncertainty ("evidence is mixed on...").

Main question: {question}

Verified evidence per sub-question:
{evidence_block}

Claims flagged by the fact-checker (do NOT state these as fact):
{flags_block}

Write the answer now. No headers, no bullet lists, plain prose."""


class Synthesizer:
    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, question: str, sub_qs: list[SubQuestion],
                   evidence: list[Evidence], critiques: list[Critique]) -> str:
        by_id = {s.id: s.text for s in sub_qs}
        ev_lines = []
        for ev in evidence:
            cites = "; ".join(c.title or c.uri for c in ev.citations[:5]) or "no citations"
            ev_lines.append(f"- Q: {by_id.get(ev.sub_question_id, '?')}\n"
                            f"  A: {ev.answer}\n  Sources: {cites}")
        flags = [f"- \"{v.claim}\" ({v.verdict}: {v.reason})"
                 for c in critiques for v in c.verdicts]
        return self.llm.generate_text(PROMPT.format(
            question=question,
            evidence_block="\n".join(ev_lines) or "(none)",
            flags_block="\n".join(flags) or "(none)",
        )).strip()

    @staticmethod
    def groundedness(evidence: list[Evidence], critiques: list[Critique]) -> float:
        """Deterministic score in [0, 1] — computed, not model-reported.

        Blends (a) share of sub-answers that carry real citations with
        (b) mean critic confidence, then subtracts a penalty per unsupported claim.
        """
        if not evidence:
            return 0.0
        grounded_share = sum(1 for e in evidence if e.grounded) / len(evidence)
        if critiques:
            mean_conf = sum(c.overall_confidence for c in critiques) / len(critiques)
            n_unsupported = sum(len(c.unsupported) for c in critiques)
        else:
            mean_conf, n_unsupported = 0.0, 0
        score = 0.5 * grounded_share + 0.5 * mean_conf - 0.05 * n_unsupported
        return round(max(0.0, min(1.0, score)), 3)
