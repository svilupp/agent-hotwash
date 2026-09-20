"""Flat per-run CSV writer.

One row per analyzed trace with the load-bearing metrics plus a total finding
count and a column per finding id that appears anywhere in the report (so the
header is stable across the whole batch). Missing metrics render as empty cells,
never a misleading ``0``.
"""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hotwash.report.model import Report, RunResult

# Fixed leading columns (stable schema); finding-id columns are appended after.
_BASE_COLUMNS = [
    "trace_id",
    "agent",
    "model",
    "experiment",
    "instance_id",
    "outcome",
    "resolved",
    "event_count",
    "user_turns",
    "assistant_turns",
    "tool_calls_total",
    "tool_error_count",
    "tool_error_rate",
    "total_tokens",
    "cost",
    "cost_source",
    "cost_estimated",
    "duration_seconds",
    "subagent_count",
    "findings_total",
]


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _base_row(run: RunResult) -> dict[str, object]:
    a = run.analysis
    m = a.root
    return {
        "trace_id": a.trace_id,
        "agent": a.agent.value,
        "model": a.model,
        "experiment": a.experiment,
        "instance_id": a.instance_id,
        "outcome": a.outcome.label,
        "resolved": a.resolved,
        "event_count": m.event_count,
        "user_turns": m.user_turns,
        "assistant_turns": m.assistant_turns,
        "tool_calls_total": m.tool_calls_total,
        "tool_error_count": m.tool_error_count,
        "tool_error_rate": m.tool_error_rate,
        "total_tokens": a.total_tokens.total,
        "cost": a.cost,
        "cost_source": a.cost_source,
        "cost_estimated": a.cost_estimated,
        "duration_seconds": m.duration_seconds,
        "subagent_count": a.subagent_count,
        "findings_total": len(run.findings),
    }


def render_csv(report: Report) -> str:
    """Serialize the per-run rows to a CSV string."""
    finding_ids = sorted(report.finding_histogram)
    columns = [*_BASE_COLUMNS, *finding_ids]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for run in report.runs:
        row = _base_row(run)
        hist = run.finding_histogram
        for fid in finding_ids:
            row[fid] = hist.get(fid, 0)
        writer.writerow([_cell(row.get(c)) for c in columns])
    return buf.getvalue()


__all__ = ["_BASE_COLUMNS", "render_csv"]
