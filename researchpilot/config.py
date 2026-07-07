"""Central configuration for ResearchPilot."""
from dataclasses import dataclass, field


# Tried in order; the first one available on the user's API key wins.
# On a 429 quota error at runtime, the client falls back down this chain —
# each model has its own free-tier quota.
MODEL_PREFERENCE = [
    "gemini-3.5-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


@dataclass
class Settings:
    model: str | None = None          # None -> auto-resolve from MODEL_PREFERENCE
    max_sub_questions: int = 3        # keeps a full run inside free-tier quotas
    max_research_retries: int = 2      # self-correction loops per sub-question
    confidence_threshold: float = 0.7  # critic confidence needed to accept evidence
    groundedness_threshold: float = 0.6  # report-level floor before corrective pass
    max_api_calls: int = 25            # hard budget guard for the whole run
    model_preference: list[str] = field(default_factory=lambda: list(MODEL_PREFERENCE))
