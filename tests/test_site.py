"""The default HTML export stays navigable as a multi-page local report."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import Finding, Severity
from agent_hotwash.diagnostics.expensive import ExpensiveRun
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.report.site import write_html_site
from agent_hotwash.runner import run_trace


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def _report(tf, ids: list[str]) -> Report:
    runs = []
    for trace_id in ids:
        trace = tf.trace(tf.session([tf.user("work")]), trace_id=trace_id)
        run = run_trace(trace, load_config(), semantic_mode="off")
        run.findings.append(
            Finding(
                id="EXAMPLE",
                kind="smell",
                severity=Severity.high,
                confidence="high",
                session_id="s0",
                message="Inspect this run",
            )
        )
        runs.append(run)
    return Report.build(runs, ReportMeta(tool_version="test"))


def test_site_pages_are_bounded_and_links_resolve(tf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_hotwash.report import site

    monkeypatch.setattr(site, "RUNS_PER_INDEX_PAGE", 1)
    target = tmp_path / "review notes.html"
    report = _report(tf, ["trace <one>", "trace/two"])
    assert write_html_site(report, target) == target

    front = target.read_text()
    assert front.index('<h2 id="actions">') < front.index('<h2 id="cost-work">')
    assert "Run explorer" not in front
    evidence = tmp_path / "review notes-evidence"
    assert 'href="review%20notes-evidence/runs/run-000001.html"' in front
    assert (evidence / "index.html").exists()
    assert (evidence / "page-2.html").exists()
    assert (evidence / "runs" / "run-000002.html").exists()
    assert "Back to report" in (evidence / "runs" / "run-000001.html").read_text()
    assert "trace &lt;one&gt;" in (evidence / "index.html").read_text()

    for page in tmp_path.rglob("*.html"):
        links = Links()
        links.feed(page.read_text())
        for href in links.hrefs:
            parsed = urlsplit(href)
            if not parsed.path:
                continue
            resolved = (page.parent / unquote(parsed.path)).resolve()
            assert resolved.is_relative_to(tmp_path)
            assert resolved.is_file(), (page, href)


def test_site_rebuild_removes_stale_pages_and_protects_other_directories(tf, tmp_path: Path) -> None:
    target = tmp_path / "report.html"
    write_html_site(_report(tf, ["first", "second"]), target)
    write_html_site(_report(tf, ["first"]), target)
    assert not (tmp_path / "report-evidence" / "runs" / "run-000002.html").exists()

    other = tmp_path / "other-evidence"
    other.mkdir()
    (other / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="refusing to replace"):
        write_html_site(_report(tf, ["first"]), tmp_path / "other.html")
    assert (other / "notes.txt").read_text() == "keep"


def test_duplicate_trace_ids_have_no_misleading_direct_link(tf, tmp_path: Path) -> None:
    write_html_site(_report(tf, ["duplicate", "duplicate"]), tmp_path / "report.html")
    front = (tmp_path / "report.html").read_text()
    assert "<code>duplicate</code>" in front
    assert 'href="report-evidence/runs/run-000001.html"' not in front
    assert 'href="report-evidence/index.html"' in front


def test_failed_site_replacement_keeps_previous_report(tf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_hotwash.report import site

    target = tmp_path / "report.html"
    write_html_site(_report(tf, ["first"]), target)
    old_front = target.read_bytes()
    old_run = (tmp_path / "report-evidence" / "runs" / "run-000001.html").read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(site.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        write_html_site(_report(tf, ["second"]), target)
    assert target.read_bytes() == old_front
    assert (tmp_path / "report-evidence" / "runs" / "run-000001.html").read_bytes() == old_run
    assert not list(tmp_path.glob(".report-evidence.*"))


def test_run_page_keeps_cohort_expense_assessment(tf, tmp_path: Path) -> None:
    report = _report(tf, ["reviewed"])
    report.expense_tail.runs = [
        ExpensiveRun(
            trace_id="reviewed",
            cost=42,
            basis="estimated",
            cohort_n=1,
            rank=1,
            cutoff=42,
            cost_share=1,
            outcome="unknown",
            assessment="verified_return_scope_unclear",
        )
    ]
    report.expense_tail.cohort_sizes = {"estimated": 1}
    report.expense_tail.selected_cost_share = {"estimated": 1}
    write_html_site(report, tmp_path / "report.html")
    page = (tmp_path / "report-evidence" / "runs" / "run-000001.html").read_text()
    assert "verified return scope unclear" in page
