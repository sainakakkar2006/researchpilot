"""Planner agent: decomposes a research question into sub-questions."""
from __future__ import annotations

from ..schemas import SubQuestion

PROMPT = """You are the planning agent of a research system.
Decompose the research question below into {n} or fewer focused sub-questions
that can each be answered with a web search. Prefer fewer, sharper questions.

Research question: {question}

Respond with ONLY a JSON array of objects:
[{{"text": "<sub-question>", "rationale": "<why this is needed>"}}]"""


class Planner:
    def __init__(self, llm, max_sub_questions: int = 4):
        self.llm = llm
        self.max_sub_questions = max_sub_questions

    def plan(self, question: str) -> list[SubQuestion]:
        prompt = PROMPT.format(n=self.max_sub_questions, question=question)
        last_err = None
        for _ in range(2):  # one retry on malformed output
            try:
                raw = self.llm.generate_json(prompt)
                subs = self._parse(raw)
                if subs:
                    return subs
            except (ValueError, KeyError, TypeError) as e:
                last_err = e
        # graceful degradation: research the question directly
        if last_err:
            return [SubQuestion(id=1, text=question, rationale="planner fallback")]
        return [SubQuestion(id=1, text=question, rationale="planner fallback")]

    def _parse(self, raw) -> list[SubQuestion]:
        if isinstance(raw, dict):  # model wrapped the array in an object
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
        if not isinstance(raw, list):
            raise ValueError("planner output is not a list")
        subs = []
        for i, item in enumerate(raw[: self.max_sub_questions], start=1):
            text = (item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
            if text:
                rationale = item.get("rationale", "") if isinstance(item, dict) else ""
                subs.append(SubQuestion(id=i, text=text, rationale=rationale))
        return subs
