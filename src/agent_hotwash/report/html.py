# ruff: noqa: E501
"""Self-contained light-theme HTML dashboard.

The renderer deliberately consumes only the stable :class:`Report` model. It
does not re-run analytics, fetch assets, or make claims where the report has
no evidence. This keeps the output useful for native Claude, Codex, pi, and
code-bench batches while remaining a single file that can be opened offline.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from typing import TYPE_CHECKING

from agent_hotwash.report.cards import actual_label, diagnosis_label, intent_label, shape_label, trajectory_label

if TYPE_CHECKING:
    from agent_hotwash.aggregate import GroupStats, MonthlyRollup
    from agent_hotwash.analytics import ErrorExample
    from agent_hotwash.report.model import Report, RunResult


_CSS = """
:root {
  color-scheme: light;
  --ink: #14213d;
  --ink-soft: #354765;
  --muted: #71809a;
  --canvas: #f4f8fc;
  --paper: #ffffff;
  --paper-tint: #f8fbff;
  --line: #dbe5f0;
  --blue: #2f6fed;
  --blue-soft: #e9f1ff;
  --cyan: #0e9fbb;
  --green: #16835b;
  --green-soft: #e6f7ef;
  --amber: #aa6b00;
  --amber-soft: #fff4d7;
  --red: #c53b52;
  --red-soft: #ffebee;
  --violet: #7055c6;
  --shadow: 0 12px 35px rgba(39, 75, 118, .08);
}
* { box-sizing: border-box; }
html { background: var(--canvas); }
body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 8% 0%, rgba(84, 172, 234, .16), transparent 31rem),
    linear-gradient(155deg, #f8fbff 0%, var(--canvas) 55%, #eef5fc 100%);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.shell { max-width: 1480px; margin: 0 auto; padding: 2.5rem clamp(1rem, 3vw, 3.5rem) 4rem; }
.eyebrow { color: var(--blue); font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
h1 { max-width: 820px; margin: .35rem 0 .6rem; font-size: clamp(2rem, 4vw, 3.6rem); line-height: 1.03; letter-spacing: -.045em; }
h2 { margin: 0; font-size: 1.15rem; letter-spacing: -.015em; }
h3 { margin: 0; font-size: .96rem; }
p { margin: 0; }
.lede { max-width: 820px; color: var(--ink-soft); font-size: 1rem; }
.meta { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.2rem; color: var(--muted); font-size: .8rem; }
.meta span { padding: .3rem .6rem; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.65); }
.meta code, code { font: .9em ui-monospace, SFMono-Regular, Menlo, monospace; }
.section { margin-top: 2.6rem; }
.section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-bottom: .8rem; }
.section-note { color: var(--muted); font-size: .8rem; }
.cards { display: grid; grid-template-columns: repeat(6, minmax(135px, 1fr)); gap: .8rem; margin-top: 2rem; }
.card, .panel, .insight { border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.9); box-shadow: var(--shadow); }
.card { min-height: 112px; padding: 1rem 1.1rem; }
.card .value { color: var(--ink); font-size: 1.65rem; font-weight: 800; letter-spacing: -.035em; font-variant-numeric: tabular-nums; }
.card .label { margin-top: .32rem; color: var(--muted); font-size: .72rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
.card .hint { margin-top: .35rem; color: var(--ink-soft); font-size: .72rem; }
.insights { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .8rem; }
.insight { position: relative; overflow: hidden; padding: 1.1rem 1.2rem 1.15rem 1.35rem; }
.insight::before { position: absolute; inset: 0 auto 0 0; width: 4px; content: ""; background: var(--blue); }
.insight.warn::before { background: var(--amber); }
.insight.good::before { background: var(--green); }
.insight.alert::before { background: var(--red); }
.insight-kicker { color: var(--blue); font-size: .68rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
.insight.warn .insight-kicker { color: var(--amber); }
.insight.good .insight-kicker { color: var(--green); }
.insight.alert .insight-kicker { color: var(--red); }
.insight h3 { margin: .25rem 0 .35rem; }
.insight p { color: var(--ink-soft); font-size: .88rem; }
.evidence-tag { display: inline-block; margin-top: .65rem; padding: .18rem .48rem; border-radius: 999px; background: var(--blue-soft); color: #275bb9; font-size: .7rem; font-weight: 700; }
.panel { padding: 1.1rem 1.2rem 1.25rem; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .84rem; }
th, td { padding: .65rem .55rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: .69rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: #f7fbff; }
.group-name { color: var(--ink); font-weight: 750; }
.subtle { color: var(--muted); }
.pill { display: inline-block; padding: .18rem .5rem; border-radius: 999px; font-size: .72rem; font-weight: 750; white-space: nowrap; }
.pill-positive, .pill-pass { color: var(--green); background: var(--green-soft); }
.pill-negative, .pill-fail { color: var(--red); background: var(--red-soft); }
.pill-unknown, .pill-na { color: var(--muted); background: #eef2f7; }
.pill-warn { color: var(--amber); background: var(--amber-soft); }
.sev-high { color: var(--red); }
.sev-medium { color: var(--amber); }
.sev-low { color: #977000; }
.sev-info { color: var(--muted); }
.barline { display: flex; align-items: center; gap: .55rem; min-width: 150px; }
.bar { width: 100px; height: 7px; overflow: hidden; border-radius: 99px; background: #e7eef7; }
.bar span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--cyan), var(--blue)); }
.barline strong { min-width: 2.5rem; color: var(--ink-soft); font-size: .78rem; font-variant-numeric: tabular-nums; }
.run-id { max-width: 180px; overflow-wrap: anywhere; color: var(--ink); font-weight: 700; }
.run-meta { margin-top: .2rem; color: var(--muted); font-size: .71rem; }
.quality { color: var(--muted); font-size: .72rem; }
details { margin-top: .3rem; }
summary { cursor: pointer; color: var(--blue); font-size: .75rem; font-weight: 750; }
.finding { margin: .6rem 0 0; padding: .7rem .8rem; border: 1px solid var(--line); border-left: 3px solid var(--blue); border-radius: 9px; background: var(--paper-tint); }
.finding .msg { margin: .2rem 0; color: var(--ink); font-weight: 700; }
.finding p { color: var(--ink-soft); font-size: .8rem; }
.error-kind { font-weight: 800; }
.owner { display: inline-block; padding: .18rem .45rem; border-radius: 999px; font-size: .69rem; font-weight: 800; white-space: nowrap; }
.owner-agent { color: #275bb9; background: var(--blue-soft); }
.owner-shared { color: var(--amber); background: var(--amber-soft); }
.owner-harness { color: #6c4aa1; background: #f1eaff; }
.owner-benign { color: var(--muted); background: #eef2f7; }
.error-example { margin-top: .35rem; color: var(--ink-soft); font-size: .76rem; }
.error-example code { display: inline-block; max-width: 520px; overflow-wrap: anywhere; color: #31445e; }
pre { max-width: 680px; margin: .45rem 0 0; padding: .6rem .7rem; overflow-x: auto; border-radius: 8px; background: #edf3fa; color: #31445e; font: .72rem/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }
.empty { padding: 1rem; border: 1px dashed var(--line); border-radius: 12px; color: var(--muted); background: rgba(255,255,255,.55); }
.footnote { margin-top: 1rem; color: var(--muted); font-size: .75rem; }
.overspend { margin: 0.75rem 0 1.5rem; font-weight: 700; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .76rem; }
@media (max-width: 1050px) { .cards { grid-template-columns: repeat(3, minmax(135px, 1fr)); } }
@media (max-width: 700px) {
  .shell { padding-top: 1.5rem; }
  .cards, .insights { grid-template-columns: 1fr; }
  .section-heading { align-items: flex-start; flex-direction: column; gap: .25rem; }
}
"""


def _esc(value: object) -> str:
    """Escape both text and attribute content produced by the renderer."""
    return html.escape("-" if value is None else str(value), quote=True)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.2f}" if abs(value) >= 1 else f"${value:.4f}"


def _number(value: object) -> str:
    """Format aggregate percentiles without leaking floating-point noise."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.1f}" if value % 1 else str(int(value))
    return _esc(value)


def _duration(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 60:
        return f"{value:.0f}s"
    minutes, seconds = divmod(round(value), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _card(value: object, label: str, hint: str | None = None) -> str:
    detail = f'<div class="hint">{_esc(hint)}</div>' if hint else ""
    return (
        f'<div class="card"><div class="value">{_esc(value)}</div><div class="label">{_esc(label)}</div>{detail}</div>'
    )


def _truth_for(run: RunResult) -> bool | None:
    outcome_truth = run.analysis.outcome.ground_truth_resolved
    return outcome_truth if outcome_truth is not None else run.analysis.resolved


def _truth_coverage(report: Report) -> tuple[int, int]:
    known = sum(_truth_for(run) is not None for run in report.runs)
    return known, len(report.runs)


def _outcome_pill(label: str, text: str | None = None) -> str:
    value = text or label
    return f'<span class="pill pill-{_esc(label)}">{_esc(value)}</span>'


def _truth_pill(value: bool | None) -> str:
    if value is None:
        return _outcome_pill("na", "not recorded")
    return _outcome_pill("pass" if value else "fail", "pass" if value else "fail")


def _section(title: str, content: str, note: str | None = None) -> str:
    note_html = f'<span class="section-note">{_esc(note)}</span>' if note else ""
    return f'<section class="section"><div class="section-heading"><h2>{_esc(title)}</h2>{note_html}</div>{content}</section>'


def _summary(report: Report) -> str:
    overall = report.aggregate.overall
    truth_known, total = _truth_coverage(report)
    cost_hint = (
        f"{report.aggregate.overall.n} analyzed runs"
        if report.aggregate.overall.total_cost is None
        else "reported or estimated"
    )
    return (
        '<div class="cards">'
        + _card(
            report.aggregate.total_traces, "traces analyzed", f"{len(report.aggregate.by_agent)} harnesses represented"
        )
        + _card(_pct(overall.success_rate), "proxy success", "directional end-of-session signal")
        + _card(
            _pct(overall.ground_truth_success_rate), "truth-backed success", f"{truth_known}/{total} runs carry truth"
        )
        + _card(_money(overall.total_cost), "total spend", cost_hint)
        + _card(report.total_findings(), "detector findings", f"{len(report.finding_histogram)} distinct signals")
        + _card(_number(overall.p95_trace_length), "p95 events", "root-session length")
        + "</div>"
    )


def _insight(kicker: str, title: str, body: str, evidence: str, tone: str = "info") -> str:
    return (
        f'<article class="insight {_esc(tone)}">'
        f'<div class="insight-kicker">{_esc(kicker)}</div>'
        f"<h3>{_esc(title)}</h3>"
        f"<p>{_esc(body)}</p>"
        f'<span class="evidence-tag">Evidence · {_esc(evidence)}</span>'
        "</article>"
    )


def _rate(g: GroupStats, *, prefer_truth: bool = False) -> tuple[float | None, str]:
    if prefer_truth and g.ground_truth_success_rate is not None:
        return g.ground_truth_success_rate, "truth"
    return g.success_rate, "proxy"


def _behavior_metrics(report: Report) -> dict[str, dict[str, float | int | None]]:
    """Aggregate behavior counters by harness without inventing denominators."""
    buckets: dict[str, list[RunResult]] = {}
    for run in report.runs:
        buckets.setdefault(run.analysis.agent.value, []).append(run)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for agent, runs in buckets.items():
        error_count = sum(run.analysis.root.tool_error_count for run in runs)
        result_count = sum(run.analysis.root.tool_results_total for run in runs)
        n = len(runs)
        metrics[agent] = {
            "runs": n,
            "error_rate": error_count / result_count if result_count else None,
            "error_count": error_count,
            "result_count": result_count,
            "retry_per_run": sum(run.analysis.root.retry_after_error for run in runs) / n if n else None,
            "compactions_per_run": sum(run.analysis.root.compaction_count for run in runs) / n if n else None,
            "cycles_per_run": sum(run.analysis.root.edit_test_cycles for run in runs) / n if n else None,
            "subagents_per_run": sum(run.analysis.subagent_count for run in runs) / n if n else None,
        }
    return metrics


def _metric(metrics: dict[str, float | int | None], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _behavior_table(report: Report) -> str:
    metrics = _behavior_metrics(report)
    if not metrics:
        return '<div class="empty">No harness behavior counters were recorded.</div>'
    rows: list[str] = []
    for agent, values in metrics.items():
        error_rate = _metric(values, "error_rate")
        error_count = _metric(values, "error_count")
        result_count = _metric(values, "result_count")
        error_pair = "-" if error_count is None or result_count is None else f"{int(error_count)} / {int(result_count)}"
        rows.append(
            "<tr>"
            f'<td class="group-name">{_esc(agent)}</td>'
            f'<td class="num">{_esc(values["runs"])}</td>'
            f'<td class="num">{_pct(error_rate)}</td>'
            f'<td class="num">{_esc(error_pair)}</td>'
            f'<td class="num">{_number(values["retry_per_run"])}</td>'
            f'<td class="num">{_number(values["compactions_per_run"])}</td>'
            f'<td class="num">{_number(values["cycles_per_run"])}</td>'
            f'<td class="num">{_number(values["subagents_per_run"])}</td>'
            "</tr>"
        )
    head = (
        "<thead><tr><th>harness</th><th class='num'>runs</th><th class='num'>error exposure</th>"
        "<th class='num'>errors / results</th><th class='num'>retries / run</th>"
        "<th class='num'>compactions / run</th><th class='num'>edit/test / run</th>"
        "<th class='num'>subagents / run</th></tr></thead>"
    )
    return f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div></div>'


def _harness_insight(report: Report) -> str | None:
    metrics = _behavior_metrics(report)
    ranked: list[tuple[str, dict[str, float | int | None], float]] = []
    for agent, values in metrics.items():
        error_rate = _metric(values, "error_rate")
        if error_rate is not None:
            ranked.append((agent, values, error_rate))
    if len(ranked) < 2:
        return None
    high = max(ranked, key=lambda row: row[2])
    low = min(ranked, key=lambda row: row[2])
    high_rate = high[2]
    low_rate = low[2]
    if high_rate <= low_rate:
        return None
    high_retries = _number(high[1]["retry_per_run"])
    high_compactions = _number(high[1]["compactions_per_run"])
    return _insight(
        "Harness difference",
        f"{high[0]} exposes more tool errors than {low[0]}",
        f"{high[0]} has {_pct(high_rate)} failed tool results versus {_pct(low_rate)} for {low[0]}. Its runs average {high_retries} retries after error and {high_compactions} compactions; inspect recovery policy alongside outcome signals.",
        f"{_number(high[1]['runs'])} vs {_number(low[1]['runs'])} runs · {_pct(high_rate)} vs {_pct(low_rate)} error exposure",
        "alert",
    )


def _insights(report: Report) -> str:
    overall = report.aggregate.overall
    truth_known, total = _truth_coverage(report)
    cards: list[str] = []

    if truth_known:
        agreement = overall.proxy_truth_agreement
        if agreement is None:
            cards.append(
                _insight(
                    "Outcome evidence",
                    "Truth is present, but agreement is unavailable",
                    "Use the per-run truth and proxy columns to inspect this batch.",
                    f"{truth_known}/{total} runs have truth",
                    "warn",
                )
            )
        elif agreement < 1:
            mismatch = truth_known - round(agreement * truth_known)
            cards.append(
                _insight(
                    "Outcome evidence",
                    "The success proxy disagrees with harness truth",
                    f"The deterministic proxy differs on {mismatch} of {truth_known} truth-backed runs. Review the outcome reasons before using proxy success for model decisions.",
                    f"agreement {_pct(agreement)} · {truth_known} truth-backed runs",
                    "warn",
                )
            )
        else:
            cards.append(
                _insight(
                    "Outcome evidence",
                    "The success proxy matches recorded truth",
                    "This batch has no observed proxy/truth disagreement; keep the truth-backed rate as the stronger benchmark.",
                    f"agreement {_pct(agreement)} · {truth_known} truth-backed runs",
                    "good",
                )
            )
    else:
        cards.append(
            _insight(
                "Outcome evidence",
                "No harness truth was recorded",
                "Proxy success is directional only here. Add resolved outcomes in the source harness before treating it as a benchmark.",
                f"0/{total} runs have truth",
                "warn",
            )
        )

    if report.finding_histogram:
        finding_id, count = next(iter(report.finding_histogram.items()))
        severity = report.finding_severity.get(finding_id, {})
        highest = next((name for name in ("high", "medium", "low", "info") if severity.get(name)), "info")
        run_count = sum(any(f.id == finding_id for f in run.findings) for run in report.runs)
        tone = "alert" if highest == "high" else "warn" if highest == "medium" else "info"
        cards.append(
            _insight(
                "Detector hotspot",
                f"{finding_id} is the most frequent signal",
                f"It appears {count} time(s) across {run_count} run(s), with {highest} as the highest observed severity. Start review with the expanded evidence on those runs.",
                f"{count} occurrences · {highest} highest severity",
                tone,
            )
        )
    elif report.aggregate.total_traces:
        cards.append(
            _insight(
                "Detector coverage",
                "No findings were emitted",
                "The current batch has no detector signals. This is not proof of clean behavior; it may also reflect detector configuration or limited trace detail.",
                f"0 findings across {report.aggregate.total_traces} traces",
                "good",
            )
        )

    tools = overall.most_error_prone_tools
    if tools:
        tool, errors = tools[0]
        total_errors = sum(count for _, count in tools)
        share = errors / total_errors if total_errors else 0
        cards.append(
            _insight(
                "Failure loop",
                f"{tool} is the leading error source",
                f"It accounts for {_pct(share)} of the recorded tool errors in the aggregate leaderboard. Inspect whether the repeated calls are recoverable retries or an interface mismatch.",
                f"{errors}/{total_errors} leaderboard errors",
                "alert" if share >= 0.5 else "warn",
            )
        )

    harness_insight = _harness_insight(report)
    if harness_insight:
        cards.append(harness_insight)

    model_groups = [(name, group) for name, group in report.aggregate.by_model.items() if group.n]
    truth_groups = [(name, group) for name, group in model_groups if group.ground_truth_success_rate is not None]
    comparison_groups = truth_groups if len(truth_groups) >= 2 else model_groups
    if len(comparison_groups) >= 2:
        prefer_truth = len(truth_groups) >= 2
        ranked = []
        for name, group in comparison_groups:
            rate, _ = _rate(group, prefer_truth=prefer_truth)
            if rate is not None:
                ranked.append((name, group, rate))
        if len(ranked) >= 2:
            best = max(ranked, key=lambda row: row[2])
            worst = min(ranked, key=lambda row: row[2])
            spread = best[2] - worst[2]
            metric = "truth-backed success" if prefer_truth else "proxy success"
            cards.append(
                _insight(
                    "Model comparison",
                    f"{best[0]} leads {metric}",
                    f"Its observed rate is {_pct(best[2])}, versus {_pct(worst[2])} for {worst[0]} — a {_pct(spread)} spread. Treat this as a lead for controlled follow-up, not a causal model verdict.",
                    f"{len(comparison_groups)} model groups · {_pct(spread)} spread",
                    "good" if spread >= 0 else "info",
                )
            )

    if not cards:
        cards.append(
            _insight(
                "Batch status",
                "No analyzable trace data",
                "The dashboard has no run-level evidence to summarize yet.",
                "empty report",
                "warn",
            )
        )
    return '<div class="insights">' + "".join(cards[:6]) + "</div>"


def _group_table(title: str, groups: dict[str, GroupStats]) -> str:
    if not groups:
        return '<div class="empty">No grouping metadata was recorded for these traces.</div>'
    rows: list[str] = []
    for name, group in groups.items():
        rows.append(
            "<tr>"
            f'<td class="group-name">{_esc(name)}</td>'
            f'<td class="num">{group.n}</td>'
            f'<td class="num">{group.skipped}</td>'
            f'<td class="num">{_pct(group.success_rate)}</td>'
            f'<td class="num">{_pct(group.ground_truth_success_rate)}</td>'
            f'<td class="num">{_pct(group.proxy_truth_agreement)}</td>'
            f'<td class="num">{_money(group.cost_per_task)}</td>'
            f'<td class="num">{_number(group.p50_trace_length)}</td>'
            f'<td class="num">{_number(group.p95_trace_length)}</td>'
            "</tr>"
        )
    head = (
        "<thead><tr><th>group</th><th class='num'>runs</th><th class='num'>skipped</th>"
        "<th class='num'>proxy</th><th class='num'>truth</th><th class='num'>agreement</th>"
        "<th class='num'>cost / run</th><th class='num'>p50 events</th><th class='num'>p95 events</th></tr></thead>"
    )
    return f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div></div>'


def _finding_histogram(report: Report) -> str:
    if not report.finding_histogram:
        return '<div class="empty">No findings.</div>'
    max_count = max(report.finding_histogram.values(), default=1)
    rows: list[str] = []
    for finding_id, count in report.finding_histogram.items():
        severity = report.finding_severity.get(finding_id, {})
        severity_text = ", ".join(f"{key}: {value}" for key, value in severity.items()) or "-"
        run_count = sum(any(f.id == finding_id for f in run.findings) for run in report.runs)
        width = max(0, min(100, round(count / max_count * 100)))
        rows.append(
            "<tr>"
            f'<td><span class="group-name">{_esc(finding_id)}</span><div class="subtle">{run_count} run(s)</div></td>'
            f'<td class="num"><div class="barline"><div class="bar"><span style="width:{width}%"></span></div><strong>{count}</strong></div></td>'
            f'<td class="subtle">{_esc(severity_text)}</td>'
            "</tr>"
        )
    head = "<thead><tr><th>detector signal</th><th class='num'>occurrences</th><th>severity mix</th></tr></thead>"
    return f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div></div>'


_ERROR_GUIDANCE: dict[str, tuple[str, str, str, str]] = {
    "agent_syntax_error": (
        "agent",
        "The model sent invalid tool arguments or violated a tool schema.",
        "Constrain tool inputs; show valid examples; retry with corrected arguments.",
        "high",
    ),
    "edit_mismatch": (
        "agent",
        "An edit did not match the file state, or the file had not been read first.",
        "Require read-before-edit and re-read after failed patches; avoid stale multi-file edits.",
        "high",
    ),
    "file_not_found": (
        "agent",
        "A command or edit targeted a path that was not present.",
        "Check paths before mutation; use repository discovery before assuming filenames.",
        "high",
    ),
    "command_not_found": (
        "shared",
        "The requested executable is missing from the environment.",
        "Pin/setup the toolchain, or teach the agent to verify availability first.",
        "high",
    ),
    "build_test_fail": (
        "shared",
        "A build or test command returned a failure.",
        "Separate product failures from agent mistakes; surface the failing assertion and require a repair loop.",
        "medium",
    ),
    "permission": (
        "harness",
        "The OS, sandbox, or repository denied the operation.",
        "Change permissions/policy or choose an allowed path; the model cannot fix this reliably.",
        "high",
    ),
    "timeout": (
        "shared",
        "A command or integration exceeded its time budget.",
        "Tune timeout/budget and split long commands; do not count timeout retries as progress.",
        "medium",
    ),
    "rate_limit": (
        "harness",
        "The provider rejected work because of load or rate limits.",
        "Use backoff, concurrency limits, or a fallback model; this is not an agent reasoning defect.",
        "medium",
    ),
    "mcp_transport": (
        "harness",
        "An MCP connection closed or returned a transport error.",
        "Fix/restart the integration and add health checks; changing the prompt will not repair transport.",
        "high",
    ),
    "sandbox_egress": (
        "harness",
        "The sandbox blocked network or filesystem egress.",
        "Adjust the sandbox policy or provide an approved proxy/tool.",
        "medium",
    ),
    "network": (
        "harness",
        "The network lookup or connection failed.",
        "Retry with bounded backoff or repair the dependency; do not blame the model without evidence.",
        "medium",
    ),
    "cancelled": (
        "shared",
        "The operation was interrupted before completion.",
        "Inspect cancellation/interrupt policy and make resume state explicit.",
        "low",
    ),
    "harness_blocked": (
        "harness",
        "The harness blocked an otherwise requested operation.",
        "Change the harness policy or provide a supported operation.",
        "high",
    ),
    "no_match_probe": (
        "benign",
        "A search returned no match; this is often an intentional probe, not a defect.",
        "Do not spend repair budget unless the missing match should have existed.",
        "low",
    ),
    "other": (
        "shared",
        "The parser could not classify the failure more specifically.",
        "Inspect the example text and improve classification before taking action.",
        "medium",
    ),
}


def _error_ledger(report: Report) -> str:
    counts: Counter[str] = Counter()
    examples: dict[str, list[ErrorExample]] = {}
    for run in report.runs:
        for metrics in [run.analysis.root, *run.analysis.subagents]:
            counts.update(metrics.error_categories)
        for example in run.analysis.error_examples:
            examples.setdefault(example.category, []).append(example)
    if not counts:
        return '<div class="empty">No actual tool errors were recorded in this slice.</div>'

    rows: list[str] = []
    for category, count in counts.most_common():
        owner, meaning, action, _severity = _ERROR_GUIDANCE.get(category, _ERROR_GUIDANCE["other"])
        owner_class = {"agent": "agent", "shared": "shared", "harness": "harness", "benign": "benign"}.get(
            owner, "shared"
        )
        sample_parts: list[str] = []
        for example in examples.get(category, [])[:2]:
            text = _esc(example.message)
            command = f" <code>{_esc(example.command)}</code>" if example.command else ""
            sample_parts.append(f'<div class="error-example">{text}{command}</div>')
        samples = "".join(sample_parts) or '<span class="subtle">No message captured.</span>'
        rows.append(
            "<tr>"
            f'<td><span class="error-kind">{_esc(category)}</span>{samples}</td>'
            f'<td><span class="owner owner-{owner_class}">{_esc(owner)}</span><div class="run-meta">{_esc(meaning)}</div></td>'
            f'<td class="num">{count}</td>'
            f"<td>{_esc(action)}</td>"
            "</tr>"
        )
    head = "<thead><tr><th>what actually failed</th><th>who can influence it</th><th class='num'>count</th><th>practical response</th></tr></thead>"
    return f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div><p class="footnote">Counts include root and subagent tool results. Samples are truncated and representative, not a complete error log.</p></div>'


def _finding_block(run: RunResult) -> str:
    if not run.findings:
        return '<span class="subtle">none</span>'
    parts = [f"<details><summary>{len(run.findings)} finding(s) · expand evidence</summary>"]
    for finding in run.findings:
        evidence = json.dumps(finding.evidence, indent=2, default=str) if finding.evidence else ""
        evidence_html = f"<pre>{_esc(evidence)}</pre>" if evidence else ""
        parts.append(
            f'<div class="finding"><div><span class="sev-{_esc(finding.severity.value)}">{_esc(finding.severity.value)}</span>'
            f' <span class="subtle">confidence {_esc(finding.confidence)}</span></div>'
            f'<div class="msg">{_esc(finding.id)}</div><p>{_esc(finding.message)}</p>{evidence_html}</div>'
        )
    parts.append("</details>")
    return "".join(parts)


def _quality(run: RunResult) -> str:
    analysis = run.analysis
    flags: list[str] = []
    if analysis.degraded:
        flags.extend(analysis.degraded)
    if not analysis.root.has_timestamps:
        flags.append("timing unavailable")
    if not analysis.root.usage_reliable:
        flags.append("usage fallback")
    if not flags:
        return '<span class="quality">complete</span>'
    unique = list(dict.fromkeys(flags))
    return f'<span class="quality">{_esc(", ".join(unique))}</span>'


def _run_detail(run: RunResult) -> str:
    analysis = run.analysis
    root = analysis.root
    truth = _truth_for(run)
    label = analysis.outcome.label
    identity = analysis.instance_id or analysis.trace_id
    experiment = analysis.experiment or "no experiment"
    model = analysis.model or root.model or "model unknown"
    tests = f"{root.test_pass_count} pass / {root.test_fail_count} fail" if root.test_run_count else "no test runs"
    tokens = _esc(analysis.total_tokens.total)
    cost = _money(analysis.cost)
    outcomes = ", ".join(analysis.outcome.reasons) or "no outcome reason recorded"
    return (
        "<tr>"
        f'<td><div class="run-id">{_esc(identity)}</div><div class="run-meta">trace {_esc(analysis.trace_id)}</div></td>'
        f"<td>{_esc(analysis.agent.value)}</td>"
        f'<td><div class="group-name">{_esc(model)}</div><div class="run-meta">{_esc(experiment)}</div></td>'
        f'<td>{_outcome_pill(label)}<div class="run-meta">truth {_truth_pill(truth)}</div></td>'
        f'<td class="num">{root.event_count}<div class="run-meta">{root.tool_calls_total} tools · {root.tool_error_count} err</div></td>'
        f'<td class="num">{root.unique_files_touched}<div class="run-meta">{root.edit_test_cycles} edit/test</div></td>'
        f'<td class="num">{tokens}<div class="run-meta">{cost} · {_esc(analysis.cost_source or "cost unavailable")}</div></td>'
        f'<td>{_finding_block(run)}<div class="run-meta">{_quality(run)}</div></td>'
        f'<td><details><summary>run notes</summary><div class="finding"><p>{_esc(outcomes)}</p>'
        f"<p>Duration {_esc(_duration(root.duration_seconds))} · test signal {_esc(tests)}</p>"
        f"<p>Subagents {_esc(analysis.subagent_count)} · cache hit {_esc(_pct(root.cache_hit_ratio))}</p></div></details></td>"
        "</tr>"
    )


def _per_run_table(report: Report) -> str:
    if not report.runs:
        return '<div class="empty">No runs to display.</div>'
    rows = "".join(_run_detail(run) for run in report.runs)
    head = (
        "<thead><tr><th>run / instance</th><th>harness</th><th>model / experiment</th><th>outcome / truth</th>"
        "<th class='num'>events / tools</th><th class='num'>files / cycles</th><th class='num'>tokens / cost</th>"
        "<th>findings</th><th>run notes</th></tr></thead>"
    )
    return f'<div class="panel"><div class="scroll"><table>{head}<tbody>{rows}</tbody></table></div></div>'


def _meta(report: Report) -> str:
    meta = report.meta
    inputs = ", ".join(_esc(path) for path in meta.inputs) or "no input paths recorded"
    detector_state = "enabled" if meta.detectors_enabled else "disabled"
    filters = " · ".join(f"{key}: {value}" for key, value in meta.filters.items())
    return (
        '<div class="meta">'
        f"<span>agent-hotwash <code>{_esc(meta.tool_version)}</code></span>"
        f"<span>generated {_esc(meta.generated_at)}</span>"
        f"<span>detectors {_esc(detector_state)}</span>"
        f"<span>config <code>{_esc(meta.config_path or 'defaults')}</code></span>"
        f'<span title="{_esc(inputs)}">{_esc(len(meta.inputs))} input path(s)</span>'
        f"{f'<span>filter {_esc(filters)}</span>' if filters else ''}"
        "</div>"
    )


def _task_cards(report: Report) -> str:
    blocks: list[str] = []
    for run in report.runs:
        if run.structure is None or not run.structure.tasks:
            continue
        rows = []
        for task in run.structure.tasks:
            rows.append(
                "<tr>"
                f"<td>{_esc(task.task_id)}</td>"
                f"<td>{_esc(intent_label(run, task.task_id))}</td>"
                f"<td>{_esc(shape_label(run, task.task_id))}</td>"
                f"<td>{_esc(actual_label(task))}</td>"
                f"<td>{_esc(trajectory_label(run, task.task_id))}</td>"
                f"<td>{_esc(diagnosis_label(run, task.task_id))}</td>"
                "</tr>"
            )
        head = (
            "<thead><tr><th>task</th><th>Intent</th><th>Shape</th>"
            "<th>Actual</th><th>Trajectory</th><th>Diagnosis</th></tr></thead>"
        )
        title = f"Task card — {_esc(run.analysis.trace_id)}"
        blocks.append(
            f"<h3>{title}</h3>"
            f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div></div>'
        )
    return "".join(blocks)


def _monthly(monthly: MonthlyRollup | None) -> str:
    if monthly is None:
        return ""
    rows = []
    for cell in monthly.cells:
        rows.append(
            "<tr>"
            f"<td>{_esc(cell.month)}</td>"
            f"<td>{_esc(cell.model_class)}</td>"
            f"<td>{_esc(cell.effort)}</td>"
            f'<td class="num">{cell.n_tasks}</td>'
            f'<td class="num">{_money(cell.invoice_total)}</td>'
            f"<td>{_esc(cell.pricing_status)}</td>"
            f'<td class="num">{_pct(cell.semantic_coverage)}</td>'
            "</tr>"
        )
    head = (
        "<thead><tr><th>month</th><th>model</th><th>effort</th><th class='num'>n tasks</th>"
        "<th class='num'>invoice</th><th>pricing</th><th class='num'>coverage</th></tr></thead>"
    )
    ranked = (
        "<p>ranked by task count: "
        f"{_esc(', '.join(monthly.ranked_by_task_count) or '-')} &middot; "
        "invoice: "
        f"{_esc(', '.join(monthly.ranked_by_invoice) or '-')} &middot; "
        "counterfactual: "
        f"{_esc(', '.join(monthly.ranked_by_counterfactual) or '-')}</p>"
    )
    statement = f'<p class="overspend">{_esc(monthly.overspend_statement)}</p>' if monthly.overspend_statement else ""
    table = (
        f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div></div>'
        if rows
        else '<div class="empty">No monthly cells.</div>'
    )
    return f"{statement}{ranked}{table}"


def render_html(report: Report) -> str:
    """Render the full self-contained HTML dashboard as a string."""
    model_count = len(report.aggregate.by_model)
    experiment_count = len(report.aggregate.by_experiment)
    task_html = _task_cards(report)
    monthly_html = _monthly(report.monthly)
    body = (
        '<main class="shell">'
        '<header><div class="eyebrow">Trace intelligence · cross-harness review</div>'
        "<h1>What actually broke — and who can fix it?</h1>"
        '<p class="lede">A focused evidence ledger for coding-agent runs. It separates agent mistakes from harness, provider, and environment failures, then shows the failed commands and practical response for each category.</p>'
        f"{_meta(report)}</header>"
        + _summary(report)
        + _section("Signals worth acting on", _insights(report), "Every callout is derived from recorded report fields")
        + _section(
            "Cross-harness comparison",
            _group_table("By agent", report.aggregate.by_agent),
            "Compare behavior by harness; truth is only shown where recorded",
        )
        + _section(
            "Behavioral fingerprints",
            _behavior_table(report),
            "Computed from root-session counters; error exposure is failed results / recorded results",
        )
        + _section(
            "Actual errors and levers",
            _error_ledger(report),
            "Start here: detector names describe patterns; this table shows the underlying failure class and your available lever",
        )
        + _section(
            "Model comparison", _group_table("By model", report.aggregate.by_model), f"{model_count} model group(s)"
        )
        + _section(
            "Experiment comparison",
            _group_table("By experiment", report.aggregate.by_experiment),
            f"{experiment_count} experiment group(s)",
        )
        + _section(
            "Detector signals", _finding_histogram(report), "Counts are occurrences; expand runs below for evidence"
        )
        + _section(
            "Run explorer",
            _per_run_table(report),
            "Root-session metrics shown; subagent totals are called out in run notes",
        )
        + (_section("Task cards", task_html, "Present when semantic mode produced tasks") if task_html else "")
        + (
            _section(
                "Monthly root-task rollup",
                monthly_html,
                "Root tasks grouped by month, model, and effort",
            )
            if monthly_html
            else ""
        )
        + "<footer>Self-contained HTML · no external assets · proxy outcomes are directional unless backed by recorded harness truth.</footer>"
        + "</main>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>agent-hotwash · trace intelligence</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


__all__ = ["render_html"]
