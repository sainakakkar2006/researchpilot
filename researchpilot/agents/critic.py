"""Critic agent: fact-checks evidence via Gemini function calling.

The model is forced to act as a strict reviewer: instead of free text, it must
report through function calls, which we parse into structured verdicts.
"""
from __future__ import annotations

from ..schemas import ClaimVerdict, Critique, Evidence, SubQuestion

FUNCTIONS = [
    {
        "name": "flag_claim",
        "description": "Flag one claim from the answer as unsupported or uncertain "
                       "given the cited evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The exact claim being flagged"},
                "verdict": {"type": "string", "enum": ["unsupported", "uncertain"]},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "description": "0-1, confidence in this flag"},
            },
            "required": ["claim", "verdict", "reason"],
        },
    },
    {
        "name": "finish_review",
        "description": "Conclude the review. MUST be called exactly once, last.",
        "parameters": {
            "type": "object",
            "properties": {
                "overall_confidence": {
                    "type": "number",
                    "description": "0-1: how well the answer is supported by its citations",
                },
                "refined_query": {
                    "type": "string",
                    "description": "If the answer needs re-research, a better search query; "
                                   "empty string otherwise",
                },
            },
            "required": ["overall_confidence"],
        },
    },
]

PROMPT = """You are a strict fact-checking agent. Review this answer to a research
question. The answer cites {n_cites} web sources ({grounded}).

Question: {question}

Answer to review:
{answer}

Sources cited: {sources}

Rules:
- Call flag_claim once for EACH claim that the cited sources cannot plausibly
  support (statistics, dates, superlatives, and causal claims deserve scrutiny).
- Then call finish_review exactly once with your overall confidence (0-1) and,
  if confidence < 0.7, a refined_query that would find better evidence.
- If the answer has no citations at all, confidence must be at most 0.3.
- Respond ONLY with function calls."""


class Critic:
    def __init__(self, llm):
        self.llm = llm

    def review(self, sub_q: SubQuestion, evidence: Evidence) -> Critique:
        sources = ", ".join(c.title or c.uri for c in evidence.citations[:8]) or "NONE"
        prompt = PROMPT.format(
            n_cites=len(evidence.citations),
            grounded="grounded" if evidence.grounded else "NOT grounded",
            question=sub_q.text,
            answer=evidence.answer,
            sources=sources,
        )
        calls = self.llm.generate_with_tools(prompt, FUNCTIONS)
        return self._parse(sub_q.id, evidence, calls)

    def _parse(self, sub_q_id: int, evidence: Evidence,
               calls: list[tuple[str, dict]]) -> Critique:
        critique = Critique(sub_question_id=sub_q_id)
        for name, args in calls:
            if name == "flag_claim":
                verdict = args.get("verdict", "uncertain")
                if verdict not in ("unsupported", "uncertain"):
                    verdict = "uncertain"
                critique.verdicts.append(ClaimVerdict(
                    claim=str(args.get("claim", ""))[:300],
                    verdict=verdict,
                    reason=str(args.get("reason", "")),
                    confidence=float(args.get("confidence", 0.5) or 0.5),
                ))
            elif name == "finish_review":
                critique.overall_confidence = max(0.0, min(1.0,
                    float(args.get("overall_confidence", 0.0) or 0.0)))
                rq = (args.get("refined_query") or "").strip()
                critique.refined_query = rq or None

        # Model never called finish_review -> distrust the whole review.
        if critique.overall_confidence == 0.0 and not critique.verdicts:
            critique.overall_confidence = 0.3 if not evidence.grounded else 0.5
        # Ungrounded evidence is capped regardless of what the model said.
        if not evidence.grounded:
            critique.overall_confidence = min(critique.overall_confidence, 0.3)
        return critique
