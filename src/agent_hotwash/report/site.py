"""Write a bounded HTML report with one evidence page per run."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import uuid
from collections import Counter
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

from agent_hotwash.report.highlights import build_highlights_data, render_highlights
from agent_hotwash.report.html import render_html
from agent_hotwash.report.model import Report, RunResult

RUNS_PER_INDEX_PAGE = 100
_MARKER = ".agent-hotwash-site"


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        value = dict(attrs).get("id")
        if value:
            self.ids.add(value)


def _href(name: str) -> str:
    return quote(name, safe="")


def _index_name(page: int) -> str:
    return "index.html" if page == 1 else f"page-{page}.html"


def _cost(run: RunResult) -> str:
    value = run.analysis.cost
    return f"${value:,.2f}" if value is not None and math.isfinite(value) else "unknown"


def _run_index(report: Report, rows: list[tuple[RunResult, str]], page: int, pages: int, summary: str) -> str:
    start = (page - 1) * RUNS_PER_INDEX_PAGE
    chunk = rows[start : start + RUNS_PER_INDEX_PAGE]
    links = [f'<a href="../{_href(summary)}">Report summary</a>']
    if page > 1:
        links.append(f'<a href="{_index_name(page - 1)}">Previous</a>')
    if page < pages:
        links.append(f'<a href="{_index_name(page + 1)}">Next</a>')
    body = "".join(
        "<tr>"
        f'<td><a href="runs/{filename}">{escape(run.analysis.trace_id)}</a></td>'
        f"<td>{escape(run.analysis.agent.value)}</td>"
        f"<td>{escape(run.analysis.model or 'unknown')}</td>"
        f"<td>{_cost(run)}</td>"
        f"<td>{len(run.findings):,}</td>"
        "</tr>"
        for run, filename in chunk
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>agent-hotwash · run evidence</title>"
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 20px}"
        "nav{display:flex;gap:18px}table{border-collapse:collapse;width:100%;margin-top:20px}"
        "th,td{text-align:left;padding:9px;border-bottom:1px solid #dce3e0}td:first-child{overflow-wrap:anywhere}"
        "a{color:#155c50}a:focus-visible{outline:3px solid #bd7737}</style></head><body>"
        f"<nav aria-label='Evidence pages'>{' '.join(links)}</nav>"
        f"<h1>Run evidence</h1><p>{len(report.runs):,} runs. Page {page} of {pages}. "
        "Rows are ordered by recorded cost; unknown costs follow priced runs. "
        "Each page contains one run and its source-linked evidence.</p>"
        "<table><thead><tr><th>Trace</th><th>Agent</th><th>Model</th><th>Cost</th><th>Findings</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
        f"<nav aria-label='Evidence pages'>{' '.join(links)}</nav></body></html>"
    )


def write_html_site(report: Report, out: Path) -> Path:
    """Write a linked summary and paged evidence; replace prior owned output safely."""
    target = out / "report.html" if out.is_dir() else out
    target.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir = target.parent / f"{target.stem}-evidence"
    if evidence_dir.exists() and (not evidence_dir.is_dir() or not (evidence_dir / _MARKER).is_file()):
        raise ValueError(f"refusing to replace non-agent-hotwash directory: {evidence_dir}")

    stage = Path(tempfile.mkdtemp(prefix=f".{evidence_dir.name}.stage-", dir=target.parent))
    front_tmp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    backup = target.parent / f".{evidence_dir.name}.backup-{uuid.uuid4().hex}"
    old_moved = False
    new_installed = False
    try:
        runs_dir = stage / "runs"
        runs_dir.mkdir()
        numbered = [(run, f"run-{index:06d}.html") for index, run in enumerate(report.runs, start=1)]
        counts = Counter(run.analysis.trace_id for run in report.runs)
        page_ids: dict[str, set[str]] = {}
        for run, filename in numbered:
            single = Report.build([run], report.meta)
            selected = (
                [row for row in report.expense_tail.runs if row.trace_id == run.analysis.trace_id]
                if counts[run.analysis.trace_id] == 1
                else []
            )
            single.expense_tail = report.expense_tail.model_copy(
                update={
                    "runs": selected,
                    "reviewed_runs": sum(row.assessment != "not_reviewed" for row in selected),
                }
            )
            page = render_html(
                single,
                back_href=f"../../{_href(target.name)}",
                title=f"Run evidence: {run.analysis.trace_id}",
            )
            (runs_dir / filename).write_text(page, encoding="utf-8")
            if counts[run.analysis.trace_id] == 1:
                parser = _Ids()
                parser.feed(page)
                page_ids[run.analysis.trace_id] = parser.ids

        ordered = sorted(
            numbered,
            key=lambda item: (
                item[0].analysis.cost is None or not math.isfinite(item[0].analysis.cost),
                -(item[0].analysis.cost or 0) if item[0].analysis.cost is not None else 0,
                item[0].analysis.trace_id,
            ),
        )
        page_count = max(1, math.ceil(len(ordered) / RUNS_PER_INDEX_PAGE))
        for page in range(1, page_count + 1):
            (stage / _index_name(page)).write_text(
                _run_index(report, ordered, page, page_count, target.name), encoding="utf-8"
            )

        run_links = {
            run.analysis.trace_id: f"{_href(evidence_dir.name)}/runs/{filename}"
            for run, filename in numbered
            if counts[run.analysis.trace_id] == 1
        }
        example_links: dict[tuple[str, str], str] = {}
        for priority in report.priorities:
            for example in priority.examples:
                href = run_links.get(example.trace_id)
                if href and example.target_id in page_ids[example.trace_id]:
                    example_links.setdefault((priority.id, example.trace_id), f"{href}#{_href(example.target_id)}")
        summary = render_highlights(
            build_highlights_data(report),
            run_links=run_links,
            example_links=example_links,
            runs_index_href=f"{_href(evidence_dir.name)}/index.html",
        )
        front_tmp.write_text(summary, encoding="utf-8")
        (stage / _MARKER).write_text("agent-hotwash HTML site\n", encoding="utf-8")

        if evidence_dir.exists():
            evidence_dir.rename(backup)
            old_moved = True
        stage.rename(evidence_dir)
        new_installed = True
        os.replace(front_tmp, target)
    except Exception:
        if new_installed:
            shutil.rmtree(evidence_dir)
        if old_moved:
            backup.rename(evidence_dir)
        raise
    else:
        if old_moved:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        front_tmp.unlink(missing_ok=True)
    return target


__all__ = ["RUNS_PER_INDEX_PAGE", "write_html_site"]
