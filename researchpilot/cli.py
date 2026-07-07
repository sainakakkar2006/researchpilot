"""ResearchPilot CLI.

    export GEMINI_API_KEY=...        # or put it in .env
    python -m researchpilot.cli "your research question"
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys

from .config import Settings
from .orchestrator import Orchestrator
from .trace import Trace


def _load_dotenv():
    """Tiny .env loader (no dependency): only reads GEMINI_API_KEY."""
    for candidate in (pathlib.Path(".env"),
                      pathlib.Path(__file__).resolve().parent.parent / ".env"):
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") and "GEMINI_API_KEY" not in os.environ:
                    os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip('"\'')
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="researchpilot",
                                     description="Multi-agent, hallucination-checked "
                                                 "research on Gemini + Google Search")
    parser.add_argument("question", help="research question to investigate")
    parser.add_argument("--model", default=None, help="override model id")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="self-correction retries per sub-question")
    parser.add_argument("--budget", type=int, default=25, help="max API calls")
    parser.add_argument("--json-out", metavar="PATH",
                        help="also write the full report as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress live trace")
    args = parser.parse_args(argv)

    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY (env var or .env file).", file=sys.stderr)
        return 2

    settings = Settings(model=args.model,
                        max_research_retries=args.max_retries,
                        max_api_calls=args.budget)

    from .llm import GeminiClient, QuotaExhausted  # deferred: keeps tests SDK-free
    trace = Trace(quiet=args.quiet)
    llm = GeminiClient(api_key=api_key, model=settings.model,
                       model_preference=settings.model_preference,
                       max_api_calls=settings.max_api_calls)
    trace.console.print(f"[dim]model: {llm.model} "
                        f"(auto-fallback on quota: {len(llm._fallback)} models)[/dim]")

    try:
        report = Orchestrator(llm, settings, on_event=trace).run(args.question)
    except QuotaExhausted as e:
        trace.console.print(f"\n[red bold]QUOTA[/red bold] {e}")
        return 3
    trace.render_report(report)

    if args.json_out:
        payload = dataclasses.asdict(report)
        pathlib.Path(args.json_out).write_text(json.dumps(payload, indent=2))
        trace.console.print(f"\n[dim]full report written to {args.json_out}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
