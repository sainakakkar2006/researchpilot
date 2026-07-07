"""Researcher agent: answers a sub-question using Google Search grounding."""
from __future__ import annotations

from ..schemas import Evidence, SubQuestion

PROMPT = """Answer the question below using Google Search. Be factual and
concise (under 200 words). Every claim must come from search results.
If the evidence is thin or conflicting, say so explicitly.

Question: {question}{hint}"""


class Researcher:
    def __init__(self, llm):
        self.llm = llm

    def research(self, sub_q: SubQuestion, refined_query: str | None = None,
                 attempt: int = 1) -> Evidence:
        hint = ""
        if refined_query:
            hint = (f"\n\nA previous answer to this question failed fact-checking. "
                    f"Search specifically for: {refined_query}")
        text, citations, queries = self.llm.generate_grounded(
            PROMPT.format(question=sub_q.text, hint=hint)
        )
        return Evidence(
            sub_question_id=sub_q.id,
            answer=text.strip(),
            citations=citations,
            search_queries=queries,
            attempt=attempt,
        )
