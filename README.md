# ResearchPilot v2 — a self-correcting multi-agent research system on Gemini

ResearchPilot answers open research questions **without trusting the model**.
Four specialized Gemini agents cooperate, and every answer must survive an
adversarial fact-check before it reaches the report. Hallucinations aren't
just avoided — they are **detected, flagged, and repaired automatically**.

## Why it exists

Hallucination and factual reliability are the top blockers to enterprise LLM
adoption. The standard fix is grounding (make the model search); ResearchPilot
goes three steps further: ground, **verify**, **score**, and **self-correct**.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              ORCHESTRATOR                   │
                        │   (control loop, budget guard, retries)     │
                        └─────────────────────────────────────────────┘
  question ──► PLANNER ──► sub-questions
                              │
              ┌───────────────▼────────────────┐
              │  RESEARCHER                    │   Gemini + Google Search
              │  grounded answer + citations   │   grounding (real URLs)
              └───────────────┬────────────────┘
                              │ evidence
              ┌───────────────▼────────────────┐
              │  CRITIC (fact-checker)         │   Gemini function calling:
              │  flag_claim / finish_review    │   structured verdicts only
              └───────────────┬────────────────┘
                    passed?   │   failed?
                       │      └──────► refined query ──► back to RESEARCHER
                       ▼                                 (self-correction loop)
              ┌────────────────────────────────┐
              │  SYNTHESIZER                   │   report + deterministic
              │  final answer + groundedness   │   groundedness score
              └────────────────────────────────┘
                       │  score < threshold? ──► corrective pass on the
                       ▼                         weakest sub-question
                    REPORT (answer, caveats, citations, score)
```

### The four hallucination defenses

1. **Grounding** — the Researcher runs with Gemini's `google_search` tool;
   citations are pulled from `grounding_metadata.grounding_chunks` (real
   web URLs the model actually retrieved, not text it generated).
2. **Adversarial verification** — the Critic reviews each answer through
   *forced function calling* (`flag_claim`, `finish_review`), so verdicts are
   structured data, never prose that itself could hallucinate. Ungrounded
   answers are hard-capped at 0.3 confidence in code, regardless of what the
   model claims.
3. **Deterministic scoring** — groundedness is *computed* in Python
   (citation coverage + critic confidence − penalty per unsupported claim),
   not self-reported by the model.
4. **Self-correction loops** — two levels: per-sub-question (critic's
   refined query feeds back into a new search, up to N retries) and
   report-level (a low overall score triggers a corrective pass on the
   weakest sub-question). A hard API-call budget guard bounds the loops.

## Quick start

```bash
pip install -r requirements.txt

# live run (key is read from .env)
./run_live.sh "What is causing 2026 AI agent reliability concerns?"

# or directly:
python -m researchpilot.cli "your question" --json-out report.json

# offline demo — replays a scripted run incl. a self-correction, no key needed
python demo_offline.py

# tests (21, fully mocked, no key needed)
python -m pytest tests/ -q
```

The CLI auto-detects the newest Gemini model available on your key
(`gemini-3.5-flash` → `gemini-2.5-flash` → ...), streams a live trace of every
agent decision (searches, flags, retries, acceptances), and prints the final
report with groundedness score, caveats, and a citation table.

## Project layout

```
researchpilot/
  config.py         thresholds, budgets, model preference list
  schemas.py        typed dataclasses passed between agents
  llm.py            single Gemini seam (google-genai SDK) — mocked in tests
  orchestrator.py   the agentic control loop
  trace.py          rich live terminal trace
  cli.py            entry point
  agents/
    planner.py      question decomposition (structured JSON output)
    researcher.py   Google Search grounding + citation extraction
    critic.py       function-calling fact-checker
    synthesizer.py  final report + deterministic groundedness score
tests/              21 tests, FakeLLM scripted runs, zero network
demo_offline.py     scripted replay of a full run with a self-correction
```

## Resume bullets (pick your favorite)

- Built ResearchPilot, a self-correcting multi-agent research system on the
  Gemini API (Google AI Developer Program) that detects and repairs LLM
  hallucinations: Google Search grounding, an adversarial function-calling
  fact-checker, and deterministic groundedness scoring cut unsupported claims
  from final reports to zero across test runs.
- Designed a dual-loop self-correction architecture (per-claim retry with
  critic-refined queries + report-level corrective passes) with hard API
  budget guards, orchestrating 4 specialized Gemini agents end to end.
- Engineered for testability: 21 unit tests exercise the full agent loop —
  including retries, budget exhaustion, and corrective passes — against a
  scripted LLM fake, with zero API calls or network access.

## Notes

- `.env` holds `GEMINI_API_KEY` and is gitignored — never commit it.
- Free-tier keys work; grounded search queries are the main quota consumer.
  The `--budget` flag (default 25 calls) caps spend per run.
- **Quota resilience**: requests are paced (2s), 429s trigger a backoff retry,
  and repeated 429s automatically fall back down the model chain
  (`gemini-3.5-flash` → `gemini-2.5-flash` → ... → flash-lite), since each
  model has its own free-tier quota. If everything is exhausted, the CLI
  exits with a clear explanation instead of a traceback.
