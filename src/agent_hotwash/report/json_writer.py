"""Canonical machine output — the JSON writer.

Emits the whole :class:`Report` as pretty-printed JSON. This is the default
format when stdout is piped, so agents and CI parse a stable schema: top-level
``meta``, ``runs`` (each ``analysis`` + ``findings`` + ``finding_histogram``),
``aggregate``, ``finding_histogram`` and ``finding_severity``. ``monthly`` and
per-run ``structure`` / ``features`` / ``capabilities`` / ``cost_views`` appear
only when semantic mode is not ``off``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_hotwash.report.model import Report


_OPTIONAL_RUN_KEYS = ("structure", "features", "capabilities", "cost_views")


def report_to_dict(report: Report) -> dict[str, Any]:
    """The report as a plain JSON-able dict (per-run finding histograms inlined).

    Semantic-only run keys are omitted when they are ``None`` so off-mode JSON
    does not grow new sections.
    """
    data = report.model_dump(mode="json")
    if data.get("monthly") is None:
        data.pop("monthly", None)
    for run, out in zip(report.runs, data["runs"], strict=True):
        out["finding_histogram"] = run.finding_histogram
        for key in _OPTIONAL_RUN_KEYS:
            if out.get(key) is None:
                out.pop(key, None)
    return data


def render_json(report: Report, *, indent: int = 2) -> str:
    """Serialize a Report to a JSON string."""
    return json.dumps(report_to_dict(report), indent=indent, default=str)


__all__ = ["render_json", "report_to_dict"]
