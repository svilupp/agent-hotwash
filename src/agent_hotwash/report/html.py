"""Self-contained single-file HTML report.

Everything is inline (CSS in a ``<style>`` block, no scripts, no external
assets or URLs) so the file opens anywhere and passes an "no external requests"
check. Layout: summary cards, per-agent aggregate table, finding histogram, and
a per-run table where each row's findings expand (native ``<details>``) to show
evidence. Colors are theme-aware via ``prefers-color-scheme``.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hotwash.aggregate import GroupStats
    from agent_hotwash.report.model import Report, RunResult

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --card: #f5f5f7;
  --border: #e0e0e0; --accent: #2563eb;
  --sev-high: #dc2626; --sev-medium: #d97706; --sev-low: #ca8a04; --sev-info: #6b7280;
  --pos: #16a34a; --neg: #dc2626; --unk: #6b7280;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e6e6; --muted: #9aa0a6; --card: #1a1d23;
    --border: #2a2e37; --accent: #60a5fa;
    --sev-high: #f87171; --sev-medium: #fbbf24; --sev-low: #facc15; --sev-info: #9ca3af;
    --pos: #4ade80; --neg: #f87171; --unk: #9ca3af;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem; background: var(--bg); color: var(--fg);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .75rem; }
.meta { color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }
.meta code { background: var(--card); padding: .1rem .3rem; border-radius: 4px; }
.cards { display: flex; flex-wrap: wrap; gap: 1rem; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem 1.25rem; min-width: 130px; }
.card .value { font-size: 1.6rem; font-weight: 600; }
.card .label { color: var(--muted); font-size: .8rem; text-transform: uppercase;
  letter-spacing: .03em; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background: var(--card); }
.pill { display: inline-block; padding: .05rem .5rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; }
.outcome-positive { color: var(--pos); }
.outcome-negative { color: var(--neg); }
.outcome-unknown { color: var(--unk); }
.sev-high { color: var(--sev-high); } .sev-medium { color: var(--sev-medium); }
.sev-low { color: var(--sev-low); } .sev-info { color: var(--sev-info); }
details { margin: .2rem 0; }
summary { cursor: pointer; }
.evidence { margin: .4rem 0 .8rem 1.2rem; }
.finding { margin: .3rem 0; padding: .4rem .6rem; background: var(--card);
  border-left: 3px solid var(--border); border-radius: 4px; }
.finding .msg { font-weight: 500; }
pre { background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  padding: .5rem .7rem; overflow-x: auto; font-size: .8rem; margin: .3rem 0 0; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem; }
"""


def _esc(value: object) -> str:
    return html.escape("-" if value is None else str(value))


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _money(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def _card(value: object, label: str) -> str:
    return f'<div class="card"><div class="value">{_esc(value)}</div><div class="label">{_esc(label)}</div></div>'


def _summary_cards(report: Report) -> str:
    agg = report.aggregate
    cards = [
        _card(agg.total_traces, "traces"),
        _card(agg.skipped, "skipped"),
        _card(_pct(agg.overall.success_rate), "success (proxy)"),
        _card(_pct(agg.overall.ground_truth_success_rate), "success (truth)"),
        _card(_money(agg.overall.cost_per_task), "cost / task"),
        _card(report.total_findings(), "findings"),
    ]
    return '<div class="cards">' + "".join(cards) + "</div>"


def _group_table(title: str, groups: dict[str, GroupStats]) -> str:
    if not groups:
        return ""
    rows = []
    for name, g in groups.items():
        rows.append(
            "<tr>"
            f"<td>{_esc(name)}</td>"
            f'<td class="num">{g.n}</td>'
            f'<td class="num">{g.skipped}</td>'
            f'<td class="num">{_pct(g.success_rate)}</td>'
            f'<td class="num">{_pct(g.ground_truth_success_rate)}</td>'
            f'<td class="num">{_money(g.cost_per_task)}</td>'
            f'<td class="num">{_esc(g.p50_trace_length)}</td>'
            f'<td class="num">{_esc(g.p95_trace_length)}</td>'
            "</tr>"
        )
    head = (
        "<tr><th>group</th><th class='num'>n</th><th class='num'>skipped</th>"
        "<th class='num'>success</th><th class='num'>truth</th><th class='num'>cost/task</th>"
        "<th class='num'>p50 len</th><th class='num'>p95 len</th></tr>"
    )
    return f"<h2>{_esc(title)}</h2><div class='scroll'><table>{head}{''.join(rows)}</table></div>"


def _finding_histogram(report: Report) -> str:
    if not report.finding_histogram:
        return "<h2>Findings</h2><p>No findings.</p>"
    rows = []
    for fid, count in report.finding_histogram.items():
        sev = report.finding_severity.get(fid, {})
        sev_str = ", ".join(f"{k}:{v}" for k, v in sev.items())
        rows.append(f"<tr><td>{_esc(fid)}</td><td class='num'>{count}</td><td>{_esc(sev_str)}</td></tr>")
    head = "<tr><th>finding</th><th class='num'>count</th><th>severities</th></tr>"
    return f"<h2>Findings histogram</h2><div class='scroll'><table>{head}{''.join(rows)}</table></div>"


def _finding_block(run: RunResult) -> str:
    if not run.findings:
        return "<span class='outcome-unknown'>none</span>"
    parts = [f"<details><summary>{len(run.findings)} finding(s)</summary><div class='evidence'>"]
    for f in run.findings:
        ev = json.dumps(f.evidence, indent=2, default=str) if f.evidence else ""
        pre = f"<pre>{_esc(ev)}</pre>" if ev else ""
        parts.append(
            f"<div class='finding'>"
            f"<span class='sev-{_esc(f.severity.value)}'>[{_esc(f.severity.value)}]</span> "
            f"<span class='msg'>{_esc(f.id)}</span> "
            f"<span class='outcome-unknown'>({_esc(f.confidence)})</span>"
            f"<div>{_esc(f.message)}</div>{pre}</div>"
        )
    parts.append("</div></details>")
    return "".join(parts)


def _per_run_table(report: Report) -> str:
    rows = []
    for run in report.runs:
        a = run.analysis
        m = a.root
        rows.append(
            "<tr>"
            f"<td>{_esc(a.instance_id or a.trace_id)}</td>"
            f"<td>{_esc(a.agent.value)}</td>"
            f"<td>{_esc(a.model)}</td>"
            f"<td class='outcome-{_esc(a.outcome.label)}'>{_esc(a.outcome.label)}</td>"
            f"<td>{_esc(a.resolved)}</td>"
            f"<td class='num'>{m.event_count}</td>"
            f"<td class='num'>{m.tool_calls_total}</td>"
            f"<td class='num'>{m.tool_error_count}</td>"
            f"<td class='num'>{_esc(a.total_tokens.total)}</td>"
            f"<td class='num'>{_money(a.cost)}</td>"
            f"<td>{_finding_block(run)}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>trace</th><th>agent</th><th>model</th><th>outcome</th><th>resolved</th>"
        "<th class='num'>events</th><th class='num'>tools</th><th class='num'>err</th>"
        "<th class='num'>tokens</th><th class='num'>cost</th><th>findings</th></tr>"
    )
    return f"<h2>Per-run detail</h2><div class='scroll'><table>{head}{''.join(rows)}</table></div>"


def render_html(report: Report) -> str:
    """Render the full self-contained HTML document as a string."""
    meta = report.meta
    meta_line = (
        f"agent-hotwash <code>{_esc(meta.tool_version)}</code> &middot; "
        f"generated {_esc(meta.generated_at)} &middot; "
        f"config <code>{_esc(meta.config_path or 'defaults')}</code> &middot; "
        f"detectors {'on' if meta.detectors_enabled else 'off'}"
    )
    inputs = ", ".join(_esc(p) for p in meta.inputs) or "-"
    body = "".join(
        [
            "<h1>agent-hotwash report</h1>",
            f"<div class='meta'>{meta_line}<br>inputs: {inputs}</div>",
            _summary_cards(report),
            _group_table("By agent", report.aggregate.by_agent),
            _group_table("By model", report.aggregate.by_model),
            _group_table("By experiment", report.aggregate.by_experiment),
            _finding_histogram(report),
            _per_run_table(report),
            "<footer>Self-contained report. No external assets.</footer>",
        ]
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>agent-hotwash report</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


__all__ = ["render_html"]
