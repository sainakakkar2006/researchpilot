<!--
  README.md
-->

<p align="center">
  <!-- BADGES:START -->
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.11-yellow"></a>
  <a href="#"><img alt="Gemini" src="https://img.shields.io/badge/gemini-google--genai-4285F4?logo=googlegemini&logoColor=white"></a>
  <a href="#"><img alt="Search Grounding" src="https://img.shields.io/badge/google%20search-grounding-34A853"></a>
  <a href="#"><img alt="Pytest" src="https://img.shields.io/badge/pytest-21%20tests%2C%20zero%20network-0A9EDC?logo=pytest&logoColor=white"></a>
  <!-- BADGES:END -->
</p>

# ResearchPilot v2

Author: Saina Kakkar

### Project Description
ResearchPilot is a self-correcting multi-agent research system built on
Gemini. It answers open research questions **without trusting the model**.
Four specialized Gemini agents cooperate, and every answer must survive an
adversarial fact-check before it reaches the report. When a hallucination is
detected, the system flags it and repairs it automatically.

Why build it this way? Hallucination is the top blocker to using LLMs for
research. The standard fix is grounding (make the model search). I wanted to
see how far you can go past that: ground, then **verify**, then **score**,
then **self-correct**.

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

### The Four Hallucination Defenses

1. **Grounding.** The Researcher runs with Gemini's `google_search` tool.
   Citations are pulled from `grounding_metadata.grounding_chunks`, which are
   real web URLs the model actually retrieved, not text it generated.
2. **Adversarial verification.** The Critic reviews each answer through
   forced function calling (`flag_claim`, `finish_review`), so verdicts
   arrive as structured data. Ungrounded answers are hard-capped at 0.3
   confidence in code, regardless of what the model claims about itself.
3. **Deterministic scoring.** Groundedness is computed in Python (citation
   coverage + critic confidence, minus a penalty per unsupported claim). The
   model does not get to grade its own homework.
4. **Self-correction loops.** Two levels. Per sub-question, the critic's
   refined query feeds back into a new search, up to N retries. Per report,
   a low overall score triggers a corrective pass on the weakest
   sub-question. A hard API-call budget bounds both loops.

## Quick Start

```bash
pip install -r requirements.txt

# live run (key is read from .env)
./run_live.sh "What is causing 2026 AI agent reliability concerns?"

# or directly:
python -m researchpilot.cli "your question" --json-out report.json
```

No API key handy? The offline demo replays a scripted run, including a
self-correction, and needs no key:

```bash
python demo_offline.py
```

The CLI auto-detects the newest Gemini model available on your key
(`gemini-3.5-flash`, then `gemini-2.5-flash`, and so on), streams a live
trace of every agent decision (searches, flags, retries, acceptances), and
prints the final report with the groundedness score, caveats, and a citation
table.

## Problems I Ran Into

1. **The fact-checker can hallucinate too.** My first Critic wrote a
   paragraph judging each answer. Sometimes that paragraph itself contained
   made-up reasoning, which defeats the whole purpose. The fix was forcing
   the Critic through function calls (`flag_claim` / `finish_review`), so a
   verdict is a data structure my code can act on, not prose I have to
   trust.

2. **Free-tier 429s in the middle of a run.** Grounded search queries burn
   quota quickly, and a long run would die halfway with rate-limit errors.
   Now requests are paced (2 seconds apart), a 429 triggers a backoff retry,
   and repeated 429s make the client fall down the model chain, since each
   model has its own free-tier quota. If everything is exhausted, the CLI
   exits with a clear explanation instead of a traceback.

3. **Loops that never stop.** A self-correcting system with a bad question
   can retry forever. The `--budget` flag (default 25 API calls) is a hard
   ceiling checked in the orchestrator. When the budget runs out, the report
   ships with its caveats section stating what could not be verified.

## Project Layout

```
researchpilot/
  config.py         thresholds, budgets, model preference list
  schemas.py        typed dataclasses passed between agents
  llm.py            single Gemini seam (google-genai SDK), mocked in tests
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

## Verify

```bash
python -m pytest tests/ -q   # 21 tests, fully mocked, no key needed
```

The tests exercise the full agent loop, including retries, budget
exhaustion, and corrective passes, against a scripted LLM fake. Zero API
calls, zero network access. Could've tested only the happy path, but the
retry and budget code is exactly where the bugs were.

## Notes

- `.env` holds `GEMINI_API_KEY` and is gitignored. Never commit it.
- Free-tier keys work. Grounded search queries are the main quota consumer.
