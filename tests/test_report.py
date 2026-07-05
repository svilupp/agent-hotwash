"""Report model + writer tests.

Uses the shared ``tf`` synthetic factory to build Analysis/Finding-bearing runs
and asserts JSON schema stability, CSV row shape, and that the HTML is fully
self-contained (no external URLs).
"""

from __future__ import annotations

import json
import re

from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import Finding, Severity, SpanRef
from agent_hotwash.report.csv_writer import render_csv
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import render_json, report_to_dict
from agent_hotwash.report.model import Report, ReportMeta, RunResult, findings_at_or_above

CONFIG = load_config(None)


def _finding(fid: str, sev: Severity, session_id: str = "s0") -> Finding:
    return Finding(
        id=fid,
        kind="taxonomy",
        severity=sev,
        confidence="high",
        session_id=session_id,
        spans=[SpanRef(session_id=session_id, event_idx=0)],
        evidence={"count": 3, "why": "example"},
        message=f"{fid} fired",
    )


def _run(tf, *, findings=None, trace_id="t0", agent=None) -> RunResult:
    root = tf.session(
        [
            tf.user("please fix the bug"),
            tf.assistant("on it"),
            tf.tool("Read", call_id="c1", category=tf.ToolCategory.read, args={"file_path": "/a.py"}),
            tf.result(call_id="c1", ok=True),
        ],
        agent=agent or tf.AgentKind.claude,
    )
    trace = tf.trace(
        root, trace_id=trace_id, instance_id=trace_id, experiment="exp1", agent=agent or tf.AgentKind.claude
    )
    analysis = analyze(trace, CONFIG)
    return RunResult(analysis=analysis, findings=findings or [])


def _report(tf) -> Report:
    runs = [
        _run(
            tf, trace_id="t0", findings=[_finding("EDIT_THRASH", Severity.high), _finding("RETRY_STORM", Severity.low)]
        ),
        _run(tf, trace_id="t1", findings=[_finding("EDIT_THRASH", Severity.medium)]),
    ]
    meta = ReportMeta(tool_version="9.9.9", inputs=["/x"])
    return Report.build(runs, meta)


def test_report_build_histograms(tf) -> None:
    report = _report(tf)
    assert report.finding_histogram == {"EDIT_THRASH": 2, "RETRY_STORM": 1}
    assert report.finding_severity["EDIT_THRASH"] == {"high": 1, "medium": 1}
    assert report.total_findings() == 3
    assert report.max_severity() is Severity.high
    assert report.aggregate.total_traces == 2


def test_runresult_finding_histogram(tf) -> None:
    run = _run(tf, findings=[_finding("A", Severity.low), _finding("A", Severity.low), _finding("B", Severity.info)])
    assert run.finding_histogram == {"A": 2, "B": 1}
    assert run.max_severity() is Severity.low


def test_findings_at_or_above() -> None:
    fs = [_finding("A", Severity.info), _finding("B", Severity.high), _finding("C", Severity.low)]
    ids = {f.id for f in findings_at_or_above(fs, Severity.low)}
    assert ids == {"B", "C"}


def test_json_schema_stable(tf) -> None:
    report = _report(tf)
    data = json.loads(render_json(report))
    # Top-level keys are the stable contract agents/CI depend on.
    assert set(data) >= {"meta", "runs", "aggregate", "finding_histogram", "finding_severity"}
    assert data["meta"]["tool_version"] == "9.9.9"
    run0 = data["runs"][0]
    assert set(run0) >= {"analysis", "findings", "finding_histogram"}
    assert set(run0["analysis"]) >= {"trace_id", "agent", "root", "cost", "outcome", "total_tokens"}
    assert run0["finding_histogram"]["EDIT_THRASH"] == 1


def test_report_to_dict_matches_render(tf) -> None:
    report = _report(tf)
    assert report_to_dict(report) == json.loads(render_json(report))


def test_csv_row_shape(tf) -> None:
    report = _report(tf)
    csv_text = render_csv(report)
    lines = csv_text.strip().splitlines()
    header = lines[0].split(",")
    # Base columns + one column per finding id in the report.
    assert "trace_id" in header
    assert "findings_total" in header
    assert "EDIT_THRASH" in header
    assert "RETRY_STORM" in header
    # One data row per run; every row has the same column count as the header.
    assert len(lines) == 1 + len(report.runs)
    for row in lines[1:]:
        assert len(row.split(",")) == len(header)


def test_html_self_contained(tf) -> None:
    report = _report(tf)
    doc = render_html(report)
    assert doc.startswith("<!doctype html>")
    assert "EDIT_THRASH" in doc
    # No external references of any kind.
    assert "http://" not in doc
    assert "https://" not in doc
    assert not re.search(r'src\s*=\s*["\']https?:', doc)
    assert "<script" not in doc.lower()
    assert "EDIT_THRASH fired" in doc  # evidence/message rendered in the drill-down


def test_html_empty_report() -> None:
    report = Report.build([], ReportMeta(tool_version="0.0.0"))
    doc = render_html(report)
    assert "No findings." in doc
    assert doc.startswith("<!doctype html>")
