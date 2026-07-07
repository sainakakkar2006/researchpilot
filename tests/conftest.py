"""Shared fakes: a scriptable stand-in for GeminiClient (no SDK, no network)."""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from researchpilot.llm import BudgetExceeded          # noqa: E402
from researchpilot.schemas import Citation            # noqa: E402


class FakeLLM:
    """Implements the GeminiClient interface with scripted responses.

    Each generate_* method pops the next scripted item from its queue.
    """

    def __init__(self, json_q=None, grounded_q=None, tools_q=None, text_q=None,
                 max_api_calls: int = 25):
        self.model = "fake-model"
        self.json_q = list(json_q or [])
        self.grounded_q = list(grounded_q or [])
        self.tools_q = list(tools_q or [])
        self.text_q = list(text_q or [])
        self.calls_made = 0
        self.max_api_calls = max_api_calls

    def _spend(self):
        if self.calls_made >= self.max_api_calls:
            raise BudgetExceeded("budget")
        self.calls_made += 1

    def generate_json(self, prompt):
        self._spend()
        item = self.json_q.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def generate_grounded(self, prompt):
        self._spend()
        return self.grounded_q.pop(0)

    def generate_with_tools(self, prompt, function_declarations):
        self._spend()
        return self.tools_q.pop(0)

    def generate_text(self, prompt, temperature=0.4):
        self._spend()
        return self.text_q.pop(0) if self.text_q else "Synthesized answer."


def cite(uri="https://example.org/a", title="Example"):
    return Citation(uri=uri, title=title)


GOOD_REVIEW = [("finish_review", {"overall_confidence": 0.9, "refined_query": ""})]
BAD_REVIEW = [
    ("flag_claim", {"claim": "GDP grew 900%", "verdict": "unsupported",
                    "reason": "no source supports this", "confidence": 0.9}),
    ("finish_review", {"overall_confidence": 0.3,
                       "refined_query": "official GDP statistics 2026"}),
]
