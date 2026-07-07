"""Thin wrapper around the google-genai SDK.

Everything the agents need from Gemini goes through this class, so tests can
mock a single seam and the agents stay SDK-agnostic.

Resilience built in:
- request pacing (free-tier RPM limits)
- 429/quota handling: backoff-retry, then automatic fallback down a chain of
  models (each model has its own free-tier quota)
- hard API-call budget
"""
from __future__ import annotations

import json
import re
import sys
import time

from .schemas import Citation


class BudgetExceeded(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    pass


QUOTA_HELP = (
    "Gemini quota exhausted on all fallback models. This is a free-tier limit, "
    "not a bug: wait ~1 minute (per-minute rate limit) or up to 24h (daily "
    "grounded-search quota), or enable billing in AI Studio. "
    "Tip: re-run with --budget 10 to use fewer calls."
)


class GeminiClient:
    def __init__(self, api_key: str, model: str | None = None,
                 model_preference: list[str] | None = None,
                 max_api_calls: int = 25, pacing: float = 2.0):
        from google import genai  # imported lazily so tests never need the SDK
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.max_api_calls = max_api_calls
        self.calls_made = 0
        self.pacing = pacing
        self._last_call: float | None = None
        self._available: set[str] | None = None

        pref = list(model_preference or [])
        self.model = model or self._resolve_model(pref)
        rest = [m for m in pref if m != self.model]
        if self._available:
            rest = [m for m in rest if m in self._available]
        self._fallback = rest

    # ---------------------------------------------------------------- model
    def _resolve_model(self, preference: list[str]) -> str:
        """Pick the first preferred model this API key can actually use."""
        try:
            self._available = {m.name.split("/")[-1] for m in self._client.models.list()}
        except Exception:
            self._available = None
            return preference[0] if preference else "gemini-2.5-flash"
        for name in preference:
            if name in self._available:
                return name
        flash = sorted((m for m in self._available
                        if "flash" in m and "image" not in m),
                       reverse=True)
        if flash:
            return flash[0]
        raise RuntimeError(
            f"No usable Gemini model found. Available: {sorted(self._available)}")

    # ---------------------------------------------------------- resilience
    @staticmethod
    def _is_quota_error(e: Exception) -> bool:
        s = str(e)
        return "RESOURCE_EXHAUSTED" in s or "429" in s.split(".")[0] or " 429 " in s

    def _generate(self, **kwargs):
        """generate_content with pacing, quota backoff, and model fallback."""
        tries_this_model, total_quota_hits = 0, 0
        while True:
            if self.pacing and self._last_call is not None:
                wait = self.pacing - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
            try:
                self._last_call = time.time()
                return self._client.models.generate_content(model=self.model, **kwargs)
            except Exception as e:
                if not self._is_quota_error(e):
                    raise
                total_quota_hits += 1
                tries_this_model += 1
                if total_quota_hits >= 6:
                    raise QuotaExhausted(QUOTA_HELP) from e
                if tries_this_model >= 2:
                    if not self._fallback:
                        raise QuotaExhausted(QUOTA_HELP) from e
                    old, self.model = self.model, self._fallback.pop(0)
                    tries_this_model = 0
                    print(f"  [quota] {old} exhausted -> falling back to {self.model}",
                          file=sys.stderr)
                    continue
                print(f"  [quota] 429 on {self.model} -> waiting 15s and retrying",
                      file=sys.stderr)
                time.sleep(15)

    # -------------------------------------------------------------- helpers
    def _spend(self):
        if self.calls_made >= self.max_api_calls:
            raise BudgetExceeded(f"API call budget of {self.max_api_calls} exhausted")
        self.calls_made += 1

    @staticmethod
    def _extract_json(text: str):
        """Parse JSON, tolerating markdown fences and leading prose."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        m = re.search(r"```(?:json)?\s*(.+?)```", text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"(\[.*\]|\{.*\})", text or "", re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise ValueError(f"No JSON found in model response: {text[:200]!r}")

    # ---------------------------------------------------------------- calls
    def generate_json(self, prompt: str):
        """Structured-output call (planner). Returns parsed JSON."""
        self._spend()
        resp = self._generate(
            contents=prompt,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return self._extract_json(resp.text)

    def generate_grounded(self, prompt: str) -> tuple[str, list[Citation], list[str]]:
        """Google-Search-grounded call (researcher).

        Returns (answer_text, citations, search_queries_the_model_ran).
        """
        self._spend()
        resp = self._generate(
            contents=prompt,
            config=self._types.GenerateContentConfig(
                tools=[self._types.Tool(google_search=self._types.GoogleSearch())],
                temperature=0.3,
            ),
        )
        return self.parse_grounded_response(resp)

    @staticmethod
    def parse_grounded_response(resp) -> tuple[str, list[Citation], list[str]]:
        text = resp.text or ""
        citations: list[Citation] = []
        queries: list[str] = []
        cand = (getattr(resp, "candidates", None) or [None])[0]
        gm = getattr(cand, "grounding_metadata", None)
        if gm is not None:
            for chunk in getattr(gm, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web is not None and getattr(web, "uri", None):
                    citations.append(Citation(uri=web.uri, title=getattr(web, "title", "") or ""))
            queries = list(getattr(gm, "web_search_queries", None) or [])
        return text, citations, queries

    def generate_with_tools(self, prompt: str, function_declarations: list[dict]):
        """Function-calling call (critic). Returns list of (name, args) calls."""
        self._spend()
        resp = self._generate(
            contents=prompt,
            config=self._types.GenerateContentConfig(
                tools=[self._types.Tool(function_declarations=function_declarations)],
                temperature=0.1,
            ),
        )
        return self.parse_function_calls(resp)

    @staticmethod
    def parse_function_calls(resp) -> list[tuple[str, dict]]:
        calls = []
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    calls.append((fc.name, dict(getattr(fc, "args", None) or {})))
        return calls

    def generate_text(self, prompt: str, temperature: float = 0.4) -> str:
        """Plain generation (synthesizer)."""
        self._spend()
        resp = self._generate(
            contents=prompt,
            config=self._types.GenerateContentConfig(temperature=temperature),
        )
        return resp.text or ""
