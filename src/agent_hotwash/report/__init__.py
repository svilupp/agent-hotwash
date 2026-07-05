"""Report assembly and pluggable writers (json/csv/table/html)."""

from __future__ import annotations

from agent_hotwash.report.csv_writer import render_csv
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import render_json, report_to_dict
from agent_hotwash.report.model import Report, ReportMeta, RunResult, findings_at_or_above
from agent_hotwash.report.table import print_table, render_table

__all__ = [
    "Report",
    "ReportMeta",
    "RunResult",
    "findings_at_or_above",
    "print_table",
    "render_csv",
    "render_html",
    "render_json",
    "render_table",
    "report_to_dict",
]
