"""Live terminal trace of the agent loop, rendered with rich."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .schemas import AgentEvent, Report

AGENT_STYLE = {
    "planner": ("bold cyan", "PLAN"),
    "researcher": ("bold green", "SEARCH"),
    "critic": ("bold yellow", "CHECK"),
    "synthesizer": ("bold magenta", "WRITE"),
    "orchestrator": ("bold white", "AGENT"),
}


class Trace:
    def __init__(self, quiet: bool = False):
        self.console = Console()
        self.quiet = quiet

    def __call__(self, event: AgentEvent):
        if self.quiet:
            return
        style, tag = AGENT_STYLE.get(event.agent, ("white", event.agent.upper()))
        marker = {"flag": "[red]!ANOMALY[/red] ", "retry": "[red]RETRY[/red] ",
                  "correct": "[red]SELF-CORRECT[/red] ", "accept": "[green]OK[/green] ",
                  }.get(event.action, "")
        self.console.print(f"[{style}]{tag:>6}[/{style}] {marker}{event.detail}")
        if event.data and "sub_questions" in event.data:
            for i, sq in enumerate(event.data["sub_questions"], 1):
                self.console.print(f"       [dim]{i}. {sq}[/dim]")

    # ------------------------------------------------------------- report
    def render_report(self, report: Report):
        c = self.console
        c.print()
        c.print(Panel(report.summary, title=f"[bold]{report.question}[/bold]",
                      border_style="cyan"))

        score = report.groundedness_score
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        c.print(f"\n  Groundedness: [{color} bold]{score:.0%}[/{color} bold]"
                f"   API calls: {report.api_calls_used}"
                f"   Corrective passes: {report.corrective_passes}")

        if report.caveats:
            c.print("\n  [yellow bold]Caveats (claims the critic flagged):[/yellow bold]")
            for cav in report.caveats:
                c.print(f"   [yellow]- {cav}[/yellow]")

        cites = report.all_citations
        if cites:
            t = Table(title=f"Citations ({len(cites)})", show_lines=False,
                      title_justify="left")
            t.add_column("#", style="dim", width=3)
            t.add_column("Source")
            t.add_column("URI", style="blue", overflow="fold")
            for i, cit in enumerate(cites, 1):
                t.add_row(str(i), cit.title or "—", cit.uri)
            c.print()
            c.print(t)
