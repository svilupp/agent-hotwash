"""Rich terminal renderer.

Three tables: a per-run summary, per-agent aggregate stats, and a findings
leaderboard (finding id -> count across all runs). Rendering goes through a
:class:`rich.console.Console`; on a non-TTY (piped/captured) rich drops color
and box-drawing on its own, so the same code degrades gracefully. ``render_table``
returns the plain string for files/tests; ``print_table`` writes to a live
console (stderr-friendly for the human view).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from agent_hotwash.aggregate import GroupStats
    from agent_hotwash.report.model import Report


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _money(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def _per_run_table(report: Report) -> Table:
    t = Table(title="Per-run analysis", show_lines=False, expand=False)
    t.add_column("trace", overflow="fold", max_width=40)
    t.add_column("agent")
    t.add_column("model", overflow="fold", max_width=24)
    t.add_column("outcome")
    t.add_column("resolved")
    t.add_column("events", justify="right")
    t.add_column("tools", justify="right")
    t.add_column("err", justify="right")
    t.add_column("tokens", justify="right")
    t.add_column("cost", justify="right")
    t.add_column("findings", justify="right")
    for run in report.runs:
        a = run.analysis
        m = a.root
        t.add_row(
            a.instance_id or a.trace_id,
            a.agent.value,
            a.model or "-",
            a.outcome.label,
            _fmt(a.resolved),
            _fmt(m.event_count),
            _fmt(m.tool_calls_total),
            _fmt(m.tool_error_count),
            _fmt(a.total_tokens.total),
            _money(a.cost),
            _fmt(len(run.findings)),
        )
    return t


def _group_table(title: str, groups: dict[str, GroupStats]) -> Table:
    t = Table(title=title, expand=False)
    t.add_column("group")
    t.add_column("n", justify="right")
    t.add_column("skipped", justify="right")
    t.add_column("success", justify="right")
    t.add_column("truth", justify="right")
    t.add_column("cost/task", justify="right")
    t.add_column("p50 len", justify="right")
    t.add_column("p95 len", justify="right")
    for name, g in groups.items():
        t.add_row(
            name,
            _fmt(g.n),
            _fmt(g.skipped),
            _pct(g.success_rate),
            _pct(g.ground_truth_success_rate),
            _money(g.cost_per_task),
            _fmt(g.p50_trace_length),
            _fmt(g.p95_trace_length),
        )
    return t


def _findings_table(report: Report) -> Table:
    t = Table(title="Findings leaderboard", expand=False)
    t.add_column("finding")
    t.add_column("count", justify="right")
    t.add_column("severities")
    if not report.finding_histogram:
        t.add_row("(none)", "0", "-")
        return t
    for fid, count in report.finding_histogram.items():
        sev = report.finding_severity.get(fid, {})
        sev_str = ", ".join(f"{k}:{v}" for k, v in sev.items()) or "-"
        t.add_row(fid, str(count), sev_str)
    return t


def _renderables(report: Report) -> list[Table]:
    out = [_per_run_table(report)]
    if report.aggregate.by_agent:
        out.append(_group_table("By agent", report.aggregate.by_agent))
    if report.aggregate.by_experiment:
        out.append(_group_table("By experiment", report.aggregate.by_experiment))
    out.append(_findings_table(report))
    return out


def render_table(report: Report, *, width: int = 120) -> str:
    """Render all tables to a plain string (non-TTY safe)."""
    # Render into a discarded buffer; the text is recovered via export_text so
    # the width/theme are controlled and the caller gets a plain (non-TTY) string.
    console = Console(record=True, width=width, file=io.StringIO())
    for r in _renderables(report):
        console.print(r)
    return console.export_text()


def print_table(report: Report, console: Console | None = None) -> None:
    """Print all tables to a live console (defaults to stdout)."""
    console = console or Console()
    for r in _renderables(report):
        console.print(r)


__all__ = ["print_table", "render_table"]
