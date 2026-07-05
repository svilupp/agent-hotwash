"""Canonical machine output — the JSON writer.

Emits the whole :class:`Report` as pretty-printed JSON. This is the default
format when stdout is piped, so agents and CI parse a stable schema: top-level
``meta``, ``runs`` (each ``analysis`` + ``findings`` + ``finding_histogram``),
``aggregate``, ``finding_histogram`` and ``finding_severity``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_hotwash.report.model import Report


def report_to_dict(report: Report) -> dict[str, Any]:
    """The report as a plain JSON-able dict (per-run finding histograms inlined)."""
    data = report.model_dump(mode="json")
    for run, out in zip(report.runs, data["runs"], strict=True):
        out["finding_histogram"] = run.finding_histogram
    return data


def render_json(report: Report, *, indent: int = 2) -> str:
    """Serialize a Report to a JSON string."""
    return json.dumps(report_to_dict(report), indent=indent, default=str)


__all__ = ["render_json", "report_to_dict"]
