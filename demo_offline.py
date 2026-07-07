"""Offline demo: replays a scripted agent run (no API key, no network).

Shows the full live trace, a self-correction retry, and the final report —
useful for demos when you don't want to spend API quota.

    python demo_offline.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "tests"))

from conftest import FakeLLM, cite                       # noqa: E402
from researchpilot.orchestrator import Orchestrator      # noqa: E402
from researchpilot.trace import Trace                    # noqa: E402

QUESTION = "What is causing 2026 AI agent reliability concerns in enterprise adoption?"

llm = FakeLLM(
    json_q=[[
        {"text": "What reliability problems do enterprises report with AI agents in 2026?",
         "rationale": "establish the concrete failure modes"},
        {"text": "How large is the measured hallucination/error rate for agentic LLM systems?",
         "rationale": "quantify the problem"},
    ]],
    grounded_q=[
        # SQ1 attempt 1: ungrounded -> critic will reject it
        ("Agents fail 97% of the time according to everyone.", [], []),
        # SQ1 attempt 2: grounded
        ("Enterprises report cascading tool-call errors and hallucinated citations "
         "as the top agent-reliability failure modes in 2026 surveys.",
         [cite("https://example.org/survey-2026", "Enterprise AI Survey 2026"),
          cite("https://example.org/reliability", "Agent Reliability Report")],
         ["enterprise AI agent failure modes 2026"]),
        # SQ2 attempt 1: grounded
        ("Benchmarks measure double-digit hallucination rates for ungrounded agents, "
         "dropping sharply when search grounding and verification layers are added.",
         [cite("https://example.org/bench", "Agentic Benchmark 2026")],
         ["agent hallucination rate benchmark 2026"]),
    ],
    tools_q=[
        [("flag_claim", {"claim": "Agents fail 97% of the time",
                         "verdict": "unsupported",
                         "reason": "no citation supports this figure",
                         "confidence": 0.95}),
         ("finish_review", {"overall_confidence": 0.2,
                            "refined_query": "enterprise AI agent failure survey 2026"})],
        [("finish_review", {"overall_confidence": 0.9})],
        [("finish_review", {"overall_confidence": 0.85})],
    ],
    text_q=["Enterprise concern over AI-agent reliability in 2026 centers on two "
            "verified failure modes: cascading tool-call errors and hallucinated "
            "citations, per industry surveys. Benchmarks show meaningful "
            "hallucination rates for ungrounded agents, which fall sharply once "
            "search grounding and independent verification layers are added. "
            "One widely repeated claim — that agents \"fail 97% of the time\" — "
            "was flagged by the fact-checker as unsupported and is excluded."],
)

trace = Trace()
trace.console.print("[dim]model: offline replay (FakeLLM)[/dim]")
report = Orchestrator(llm, on_event=trace).run(QUESTION)
trace.render_report(report)
