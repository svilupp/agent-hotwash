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

from agent_hotwash.diagnostics.tails import TailIncident
from agent_hotwash.report.cards import actual_label, diagnosis_label, intent_label, shape_label, trajectory_label
from agent_hotwash.report.priorities import metric as display_metric

if TYPE_CHECKING:
    from agent_hotwash.aggregate import GroupStats, MonthlyRollup
    from agent_hotwash.analytics import ErrorExample
    from agent_hotwash.report.model import Report, RunResult
    from agent_hotwash.semantic.results import FeatureSet


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
.report-nav { display: flex; flex-wrap: wrap; gap: .6rem 1rem; margin-top: 1.3rem; font-size: .84rem; }
a { color: var(--blue); }
.actions { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.action { padding: 1.2rem; overflow-wrap: anywhere; }
.action h3 { font-size: 1.05rem; margin: .45rem 0; }
.action p { margin-top: .55rem; font-size: .86rem; }
.action ul { padding-left: 1.1rem; font-size: .8rem; }
.action .scope { color: var(--muted); font-size: .78rem; }
.action .limit { color: var(--muted); font-size: .77rem; }
.evidence-facts { display: grid; grid-template-columns: minmax(100px, 1fr) 3fr; gap: .3rem .7rem; font-size: .8rem; }
.evidence-facts dt { color: var(--muted); }
.evidence-facts dd { margin: 0; overflow-wrap: anywhere; }
.tail-detail { border-bottom: 1px solid var(--line); padding: .5rem 0; }
.tail-detail summary { font-size: .83rem; overflow-wrap: anywhere; }
.tail-detail[open] { padding: .8rem; background: var(--paper); border-radius: 10px; }
:target { scroll-margin-top: 1rem; outline: 2px solid var(--blue); outline-offset: 4px; }
.overspend { margin: 0.75rem 0 1.5rem; font-weight: 700; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .76rem; }
@media (max-width: 1050px) { .cards { grid-template-columns: repeat(3, minmax(135px, 1fr)); } }
@media (max-width: 700px) {
  .shell { padding-top: 1.5rem; }
  .cards, .insights, .actions { grid-template-columns: 1fr; }
  .section-heading { align-items: flex-start; flex-direction: column; gap: .25rem; }
}
"""

# Static reports still work without JS; this only opens linked evidence and its
# enclosing disclosure sections. No trace text is interpolated into the script.
_LINK_SCRIPT = """
<script>
(() => {
  function revealEvidence() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target) return;
    for (let el = target; el; el = el.parentElement) {
      if (el.tagName === 'DETAILS') el.open = true;
    }
    target.scrollIntoView({block: 'start'});
  }
  window.addEventListener('hashchange', revealEvidence);
  revealEvidence();
})();
</script>
"""


def _esc(value: object) -> str:
    """Escape both text and attribute content produced by the renderer."""
    return html.escape("-" if value is None else str(value), quote=True)


def _known(value: object) -> str:
    """Show missing measurements explicitly while preserving zero and false."""
    return "unknown" if value is None else str(value)


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
    anchor = title.lower().replace(" ", "-").replace("%", "percent")
    return f'<section id="{_esc(anchor)}" class="section"><div class="section-heading"><h2>{_esc(title)}</h2>{note_html}</div>{content}</section>'


def _summary(report: Report) -> str:
    overall = report.aggregate.overall
    truth_known, total = _truth_coverage(report)
    supported_cases = sum(row.incidents for row in report.priorities if row.status == "supported")
    priced = [row for run in report.runs for row in run.tails.model_activity if row.estimated_cost is not None]
    charges = sum(row.estimated_cost or 0 for row in priced)
    coordination = sum(row.estimated_cost or 0 for row in priced if row.category == "coordination")
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
        + _card(supported_cases, "supported change cases", "trace evidence plus bounded JeV review")
        + _card(
            _pct(overall.ground_truth_success_rate), "truth-backed success", f"{truth_known}/{total} runs carry truth"
        )
        + _card(_money(overall.total_cost), "total spend", cost_hint)
        + _card(report.total_findings(), "detector findings", f"{len(report.finding_histogram)} distinct signals")
        + _card(
            _pct(coordination / charges if charges else None),
            "coordination charge share",
            "of priced, attributed model responses",
        )
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


def _priorities(report: Report) -> str:
    priorities = report.priorities
    coverage = report.classifier_coverage
    failed = sum(c.api_errors for c in coverage.values())
    coverage_note = (
        '<p class="notice"><strong>Classifier review is incomplete.</strong> '
        f"{failed} feature requests failed. These are missing evidence, not classifier uncertainty. "
        "Restore API access and rerun the affected reviews before drawing broader conclusions.</p>"
        if failed
        else ""
    )
    coverage_note += (
        '<div id="classifier-coverage"><strong>Classifier coverage</strong><ul>'
        + "".join(f"<li>{_esc(scope)}: {_esc(c.summary())}</li>" for scope, c in coverage.items())
        + '</ul><p class="section-note">Coverage refers to selected features only; it does not measure the share of all behavior reviewed.</p></div>'
        if coverage
        else '<p class="section-note">No classifier feature records are available. Measured investigations remain useful; this report cannot establish classifier-supported changes.</p>'
    )
    if not priorities:
        return (
            coverage_note
            + '<div class="empty">No prioritized actions are supported by this slice. Check coverage and the retained observations below; absence of an action is not proof of efficient execution.</div>'
        )
    labels = {"supported": "Supported change", "investigate": "Investigate", "measure": "Improve visibility"}
    cards = []
    for row in priorities:
        tone = (
            "pill-pass" if row.status == "supported" else "pill-warn" if row.status == "investigate" else "pill-unknown"
        )
        charge = (
            f"<p><strong>{_money(row.observed_cost)}</strong> {_esc(row.cost_basis)} in these cases.</p>"
            if row.observed_cost is not None
            else ""
        )
        examples = "".join(
            f'<li><a href="#{_esc(example.target_id)}">{_esc(example.label)}</a>'
            f'<span class="run-meta"> · {_esc(example.trace_id)}</span></li>'
            for example in row.examples
        )
        cards.append(
            f'<li class="panel action" id="action-{_esc(row.id)}">'
            f'<span class="pill {tone}">{labels[row.status]}</span>'
            f'<h3>{_esc(row.title)}</h3><p class="scope">Suggested owner: {_esc(row.owner)} · '
            f"{row.affected_runs} workflows · {row.incidents} cases</p>"
            f"<p>{_esc(row.evidence)}</p>{charge}"
            f"<p><strong>Next step:</strong> {_esc(row.next_step)}</p>"
            f"<p><strong>Verify improvement:</strong> {_esc(row.verify)}</p>"
            f'<ul aria-label="Examples for {_esc(row.title)}">{examples}</ul>'
            f'<p class="limit">{_esc(row.limit)}</p></li>'
        )
    return (
        coverage_note
        + '<p class="section-note">Supported changes come first, followed by cost, recovery, and latency investigations, '
        "then visibility fixes. Counts refer to the selected cases below. Costs and time intervals can overlap; "
        "these cards do not estimate savings.</p>"
        '<ol class="actions">' + "".join(cards) + "</ol>"
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

    if report.expense_tail.selected_cost_share:
        basis = max(report.expense_tail.cohort_sizes, key=lambda name: report.expense_tail.cohort_sizes[name])
        share = report.expense_tail.selected_cost_share[basis]
        assessments = Counter(row.assessment for row in report.expense_tail.runs if row.basis == basis)
        needs_review = assessments["investigate_blocked_recovery"] + assessments["verified_return_scope_unclear"]
        cards.append(
            _insight(
                "Expense concentration",
                f"The top 5% accounts for {_pct(share)} of {basis} spend",
                f"Among {report.expense_tail.cohort_sizes[basis]} priced runs, {needs_review} selected runs have an incomplete or unclear scope assessment. Inspect their matched requests, returns, and child activity before judging necessity.",
                f"{len([row for row in report.expense_tail.runs if row.basis == basis])} selected runs · {basis}",
                "warn" if needs_review else "info",
            )
        )

    priced_activity = [row for run in report.runs for row in run.tails.model_activity if row.estimated_cost is not None]
    activity_charge: Counter[str] = Counter()
    for row in priced_activity:
        activity_charge[row.category] += row.estimated_cost or 0
    attributed_charge = sum(activity_charge.values())
    priced_rounds = sum(row.rounds for row in priced_activity)
    if attributed_charge > 0 and priced_rounds >= 10:
        category, charge = activity_charge.most_common(1)[0]
        label = {
            "coordination": "Coordination requests",
            "inspection": "Inspection requests",
            "execution": "Execution and edit requests",
            "no_tools": "Records without tools",
            "mixed": "Mixed tool requests",
        }[category]
        components: Counter[str] = Counter()
        for row in priced_activity:
            if row.category == category:
                components.update(row.estimated_cost_components)
        component, amount = components.most_common(1)[0] if components else ("unknown", 0)
        cards.append(
            _insight(
                "Model work mix",
                f"{label} carry {_pct(charge / attributed_charge)} of attributed charges",
                f"The largest charge component is {component.replace('_', ' ')} ({_money(amount)}). "
                "Inspect the work-mix table and request-level tails to identify reusable context or avoidable round trips. "
                "These are observed price-table estimates, not savings.",
                f"{priced_rounds:,} priced model rounds · estimated child usage excluded",
                "info",
            )
        )

    tail_counts: Counter[str] = Counter()
    tail_runs: Counter[str] = Counter()
    for run in report.runs:
        crossed = [row.kind for row in run.tails.incidents if row.exceeds_threshold]
        tail_counts.update(crossed)
        tail_runs.update(set(crossed))
    if tail_runs:
        kind, affected = tail_runs.most_common(1)[0]
        sample = next(row for run in report.runs for row in run.tails.incidents if row.kind == kind)
        cards.append(
            _insight(
                "Execution tail",
                f"{kind.replace('_', ' ')} crossed its threshold in {affected} runs",
                f"There are {tail_counts[kind]} measured crossings at {sample.threshold:,.1f} {sample.unit}. Review the longest examples and their source coordinates in Execution extremes.",
                f"{affected}/{len(report.runs)} runs · {tail_counts[kind]} observations",
                "info",
            )
        )

    behavioral = [(key, count) for key, count in report.finding_histogram.items() if not key.startswith("TAIL_")]
    if behavioral:
        finding_id, count = behavioral[0]
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
    elif not report.finding_histogram and report.aggregate.total_traces:
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
    # Proxy outcomes across unmatched workloads do not establish a useful model
    # ranking. Keep the descriptive comparison table without a winner callout.
    comparison_groups = truth_groups
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
    return '<div class="insights">' + "".join(cards[:8]) + "</div>"


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
    failures = [failure for run in report.runs for failure in run.analysis.failures]
    if failures:
        counts: Counter[str] = Counter(failure.leaf for failure in failures)
        rows: list[str] = []
        for leaf, count in counts.most_common():
            leaf_rows = [failure for failure in failures if failure.leaf == leaf]
            distinct = []
            seen: set[str] = set()
            # Prefer diversity before filling remaining slots.
            diversity: set[tuple[str, str, str]] = set()
            for failure in leaf_rows:
                sid = failure.provenance.stable_id
                key = (failure.provenance.trace_id, failure.tool, failure.legacy_category)
                if sid in seen or (key in diversity and len(distinct) < min(5, len(leaf_rows))):
                    continue
                seen.add(sid)
                diversity.add(key)
                distinct.append(failure)
                if len(distinct) == 10:
                    break
            if len(distinct) < 10:
                for failure in leaf_rows:
                    if failure.provenance.stable_id in seen:
                        continue
                    seen.add(failure.provenance.stable_id)
                    distinct.append(failure)
                    if len(distinct) == 10:
                        break
            examples: list[str] = []
            for failure in distinct:
                p = failure.provenance
                command = f"<code>{_esc(failure.command)}</code>" if failure.command else "no command captured"
                retry = failure.recovery
                why = failure.abstention_reason or "ordered rule matched supported evidence"
                examples.append(
                    "<details class='error-example'><summary>"
                    f"{_esc(p.trace_id)} · {_esc(failure.tool)} · event {_esc(p.event_idx)}"
                    "</summary>"
                    f"<div>{command}</div><pre>{_esc(failure.diagnostic)}</pre>"
                    f"<div class='run-meta'>incident {_esc(failure.incident_id)} · owner {_esc(failure.owner)} · "
                    f"source record {_esc(p.record_index)} / ordinal {_esc(p.ordinal)} · "
                    f"retry {_esc(retry.outcome)} after {_esc(retry.attempts)} equivalent attempt(s) · why: {_esc(why)}</div>"
                    "</details>"
                )
            flags = leaf_rows[0].disposition
            owner = leaf_rows[0].owner
            raw = ", ".join(f"{name} {n}" for name, n in Counter(r.legacy_category for r in leaf_rows).most_common())
            incident_count = len({r.incident_id for r in leaf_rows})
            rows.append(
                "<tr>"
                f"<td><span class='error-kind'>{_esc(leaf)}</span><div class='run-meta'>{_esc(flags)} · "
                f"{incident_count} incident(s) · legacy: {_esc(raw)}</div>{''.join(examples)}</td>"
                f"<td>{_esc(owner)}</td><td class='num'>{count}</td><td>{_esc(leaf_rows[0].action)}</td>"
                "</tr>"
            )
        terminal = sum(not f.observation and f.leaf != "unresolved" for f in failures)
        observations = sum(f.observation for f in failures)
        unresolved = sum(f.leaf == "unresolved" for f in failures)
        head = "<thead><tr><th>primary leaf and distinct evidence</th><th>who can influence it</th><th class='num'>count</th><th>practical response</th></tr></thead>"
        reconcile = f"failed results {len(failures)} = terminal {terminal} + expected observations {observations} + unresolved {unresolved}"
        return f'<div class="panel"><div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div><p class="footnote">{_esc(reconcile)}. Raw tool status remains unchanged; examples are deduplicated by stable event identity.</p></div>'

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
        f'<tr id="run-{_esc(analysis.trace_id)}">'
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


def _handover_ledger(report: Report) -> str:
    rows = [row for run in report.runs for row in run.analysis.handovers]
    feature_sets = {fs.object_id: fs for run in report.runs for fs in (run.features or []) if fs.scope == "handover"}
    cache_waits = [wait for run in report.runs for wait in run.analysis.cache_waits]
    if not rows:
        return '<div class="empty">No observed delegation spawns in this slice.</div>'
    statuses = Counter(row.status for row in rows)
    n = len(rows)
    visible_request = sum(row.request_visibility == "plaintext" for row in rows)
    visible_reply = sum(row.reply_visibility == "plaintext" for row in rows)
    encrypted_requests = sum(row.request_visibility == "encrypted" for row in rows)
    ambiguous_links = sum(bool(row.link_candidates) for row in rows)
    steers = sum(row.steer_count for row in rows)
    continuations = sum(row.continuation_count for row in rows)
    timed = sum(row.spawn_to_final_seconds is not None for row in rows)
    status_text = ", ".join(f"{k} {v}" for k, v in statuses.items())
    stats = report.aggregate.overall
    steering_text = ", ".join(f"{k} steers: {v}" for k, v in sorted(stats.handover_steer_distribution.items()))
    role_text = ", ".join(f"{k} {v}" for k, v in stats.handover_roles.items())
    model_text = ", ".join(f"{k} {v}" for k, v in stats.handover_models.items())
    overview = (
        f'<p class="footnote">{n} observed spawns · {continuations} continuations · {steers} steers · '
        f"{visible_request}/{n} plaintext requests · {visible_reply}/{n} plaintext replies · "
        f"states: {_esc(status_text)}; encrypted requests {encrypted_requests}; "
        f"prefix-only link candidates {ambiguous_links}; unmatched child returns {stats.orphaned_returns}. "
        "Ciphertext and missing payloads are excluded from size measures. "
        f"Request chars p50/p95 {_esc(stats.handover_request_chars_p50)}/"
        f"{_esc(stats.handover_request_chars_p95)} (n={visible_request}); "
        f"reply chars p50/p95 {_esc(stats.handover_reply_chars_p50)}/"
        f"{_esc(stats.handover_reply_chars_p95)} (n={visible_reply}). "
        f"Runtime seconds p50/p95 {_esc(stats.handover_runtime_p50_seconds)}/"
        f"{_esc(stats.handover_runtime_p95_seconds)} (n={timed}). "
        f"Steering distribution: {_esc(steering_text or 'none')}. "
        f"Roles: {_esc(role_text)}. Observed models: {_esc(model_text)}.</p>"
    )
    buckets: dict[str, list] = {}
    for row in rows:
        buckets.setdefault(row.primary_intervention, []).append(row)
    bucket_rows = []
    for name, members in sorted(buckets.items(), key=lambda item: -len(item[1])):
        first = members[0]
        bucket_rows.append(
            f"<tr><td>{_esc(name)}</td><td class='num'>{len(members)}</td>"
            f"<td class='num'>{len({member.trace_id for member in members})}</td>"
            f"<td class='num'>{sum(member.spawn_to_final_seconds or 0 for member in members):.0f} "
            f"(n={sum(member.spawn_to_final_seconds is not None for member in members)})</td>"
            f"<td><a href='#handover-{_esc(first.id)}'>example {_esc(first.id)}</a></td></tr>"
        )
    bucket_table = (
        "<div class='scroll'><table><thead><tr><th>intervention</th><th>handovers</th>"
        "<th>traces</th><th>runtime seconds</th><th>example</th></tr></thead><tbody>"
        + "".join(bucket_rows)
        + "</tbody></table></div>"
    )
    # Keep every handover navigable in HTML, but embed the full structured record
    # only for the longest observed lifetimes. The JSON report retains all rows.
    full_ids = {row.id for row in sorted(rows, key=lambda item: -(item.spawn_to_final_seconds or 0))[:100]}
    cards = []
    for row in rows:
        feature_set = feature_sets.get(row.id)
        feature_text = ", ".join(
            f"{key}={value.reason or value.value}" for key, value in (feature_set.values.items() if feature_set else [])
        )
        timeline = " · ".join(
            f"{_esc(ev.kind)} [{_esc(ev.session_id)}:{_esc(ev.source.record_index if ev.source else ev.event_idx)}]"
            for ev in row.events
        )
        req = f"<pre>{_esc(row.request_excerpt)}</pre>" if row.request_excerpt else _esc(row.request_visibility)
        rep = f"<pre>{_esc(row.reply_excerpt)}</pre>" if row.reply_excerpt else _esc(row.reply_visibility)
        metrics = [
            (
                "Request",
                f"{row.request_chars} chars · {row.request_bytes} UTF-8 bytes"
                if row.request_chars is not None
                else "unknown",
            ),
            (
                "Reply",
                f"{row.reply_chars} chars · {row.reply_bytes} UTF-8 bytes"
                if row.reply_chars is not None
                else "unknown",
            ),
            (
                "Reply/request",
                f"{row.request_to_reply_chars:.2f}x" if row.request_to_reply_chars is not None else "unknown",
            ),
            (
                "Spawn → final",
                f"{row.spawn_to_final_seconds:.1f} s" if row.spawn_to_final_seconds is not None else "unknown",
            ),
            (
                "Final → consumption",
                f"{row.final_to_consumption_seconds:.1f} s"
                if row.final_to_consumption_seconds is not None
                else "unknown",
            ),
            ("Continuations / steers", f"{row.continuation_count} / {row.steer_count}"),
            ("Exact child transcript / estimated usage", f"{row.child_transcript_present} / {row.usage_estimated}"),
            (
                "Child tokens input / output / cache read / cache write",
                "/".join(_known(row.child_tokens.get(k)) for k in ("input", "output", "cache_read", "cache_write"))
                if row.child_tokens
                else "unknown",
            ),
            (
                "Return after parent final / late file overlap / possible overlap",
                f"{_known(row.return_after_parent_final)} / {_known(row.late_return_overlap)} / {row.possible_overlap}",
            ),
        ]
        metric_html = "".join(f"<tr><th>{_esc(label)}</th><td>{_esc(value)}</td></tr>" for label, value in metrics)
        steer_features = [
            (fs.object_id, fs) for fs in feature_sets.values() if fs.object_id.startswith(f"{row.id}:steer:")
        ]
        steer_text = "; ".join(
            f"{key}: " + ", ".join(f"{name}={value.reason or value.value}" for name, value in fs.values.items())
            for key, fs in steer_features
        )
        full_record_html = (
            f"<details><summary>Full handover record</summary><pre>{_esc(json.dumps(row.model_dump(mode='json'), indent=2))}</pre></details>"
            if row.id in full_ids
            else ""
        )
        cards.append(
            f'<details id="handover-{_esc(row.id)}" class="error-example"><summary>'
            f"{_esc(row.trace_id)} · {_esc(row.role or 'agent')} · {_esc(row.status)} · {_esc(row.id)}"
            "</summary>"
            f'<div class="run-meta">parent {_esc(row.parent_id)} · agent {_esc(row.agent_id or "unknown")} · '
            f"child {_esc(row.child_id or 'unlinked')} · candidates {_esc(len(row.link_candidates))} · "
            f"link {_esc(row.link_confidence)} · model {_esc(row.observed_model or row.requested_model or 'unknown')} · "
            f"action {_esc(row.primary_intervention)}</div>"
            f"<div>Request ({_esc(row.request_visibility)}): {req}</div>"
            f"<div>Reply ({_esc(row.reply_visibility)}): {rep}</div>"
            f'<div class="scroll"><table><tbody>{metric_html}</tbody></table></div>'
            f'<div class="run-meta">request files {_esc(", ".join(row.request_files) or "none")} · '
            f"reply files {_esc(', '.join(row.reply_files) or 'none')} · "
            f"child reads {_esc(', '.join(row.child_files_read) or 'unknown')} · "
            f"child writes {_esc(', '.join(row.child_files_written) or 'unknown')} · "
            f"parent reads after {_esc(', '.join(row.parent_files_read_after) or 'none observed')} · "
            f"model revisions {_esc(', '.join(row.observed_model_revisions) or 'unknown')} · "
            f"failure results {_esc(', '.join(row.failure_ids) or 'none')}</div>"
            f'<div class="run-meta">JeV: {_esc(feature_text or "unavailable")}</div>'
            f'<div class="run-meta">Steer JeV: {_esc(steer_text or "unavailable")}</div>'
            f'<div class="run-meta">{timeline}</div>'
            f"{full_record_html}"
            "</details>"
        )
    wait_rows = "".join(
        "<tr>"
        f"<td><a href='#handover-{_esc(wait.handover_id)}'>{_esc(wait.handover_id)}</a> · {_esc(wait.wait_event_idx)}</td>"
        f"<td>{f'{wait.wait_seconds:.1f} s' if wait.wait_seconds is not None else 'unknown'}</td>"
        f"<td>{_esc(wait.prior_model)} → {_esc(wait.next_model)}"
        f'<div class="run-meta">responses {_esc(wait.prior_response_id)} → {_esc(wait.next_response_id)}; '
        f"events {_esc(_known(wait.prior_event_start))} → {_esc(_known(wait.next_event_start))}</div></td>"
        f"<td>{_esc(_known(wait.prior_input))} / {_esc(_known(wait.prior_cache_read))}</td>"
        f"<td>{_esc(_known(wait.next_input))} / {_esc(_known(wait.next_cache_read))} / {_esc(_known(wait.next_cache_write))}</td>"
        f"<td>{_esc(wait.comparable)} / {_esc(_known(wait.cache_rewrite_after_wait))}</td>"
        f"<td>{_esc(wait.compaction_between)} / {_esc(wait.context_edit_between)} / "
        f"{_esc(wait.model_switch)} / {_esc(wait.new_instructions_between)}</td>"
        f"<td>{_money(wait.prior_input_cost)} → {_money(wait.next_input_cost)}"
        f"<div class='run-meta'>next cache write: {_money(wait.next_input_cost_components.get('cache_write'))}; "
        f"{_esc(wait.next_pricing_status)}</div></td>"
        "</tr>"
        for wait in cache_waits
    )
    wait_table = (
        "<h3>Parent cache use around child-result waits</h3>"
        f"<p>{len(cache_waits)} waits; {sum(w.comparable for w in cache_waits)} comparable. "
        "Values are observed tokens on the adjacent parent model calls; a rewrite flag does not establish that the child caused the transition or that tokens were wasted.</p>"
        "<div class='scroll'><table><thead><tr><th>handover / wait event</th><th>wait</th><th>model</th>"
        "<th>prior input / cache read</th><th>next input / cache read / cache write</th>"
        "<th>comparable / rewrite</th><th>compaction / context edit / model switch / new instructions</th>"
        "<th>estimated input charge</th></tr></thead>"
        f"<tbody>{wait_rows}</tbody></table></div>"
        if cache_waits
        else "<p>No child-result waits observed.</p>"
    )
    return (
        f'<div class="panel">{overview}{bucket_table}'
        f'<p class="footnote">All {n} handovers have navigable summaries. Full structured records for the '
        f"{min(100, n)} longest observed lifetimes appear here; every record is in JSON.</p>"
        f"{''.join(cards)}{_cache_wait_costs(report)}{wait_table}</div>"
    )


def _cache_wait_costs(report: Report) -> str:
    if not report.cache_wait_cohorts:
        return ""
    rows = []
    for group in report.cache_wait_cohorts:
        rate = f"{group.rewrites / group.comparable:.1%}" if group.comparable else "unknown"
        rows.append(
            f"<tr><td>{_esc(group.model or 'unknown')}</td><td>{_esc(group.wait_mode)}</td>"
            f"<td>{_esc(group.duration_band)}</td><td class='num'>{group.waits}</td>"
            f"<td class='num'>{group.comparable}</td><td class='num'>{group.rewrites} ({rate})</td>"
            f"<td class='num'>{_money(group.next_input_cost)}<div class='run-meta'>{group.priced_responses} responses priced</div></td>"
            f"<td class='num'>{_money(group.rewrite_write_cost)}</td></tr>"
        )
    return (
        "<h3>Cache waits by duration and model</h3>"
        "<p>Duration bands are configurable descriptions, not provider expiry rules. "
        "Rates use comparable waits only. Nonblocking probes are separated from explicit blocking waits. "
        "Charges deduplicate shared next responses within a row; the same response may occur in different rows, "
        "so do not sum row charges. Values overlap the model work mix and are not savings.</p>"
        "<div class='scroll'><table><thead><tr><th>model</th><th>wait mode</th><th>duration</th><th>waits</th>"
        "<th>comparable</th><th>rewrite matches</th><th>next input charge</th><th>matched rewrite charge</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _handover_unknowns(report: Report) -> str:
    rows = [row for run in report.runs for row in run.analysis.handovers]
    if not rows:
        return '<div class="empty">No handover visibility gaps in this slice.</div>'
    counts = {
        "encrypted requests": sum(row.request_visibility == "encrypted" for row in rows),
        "missing replies": sum(row.reply_visibility == "missing" for row in rows),
        "truncated replies": sum(row.reply_visibility == "truncated" for row in rows),
        "no exact child transcript join": sum(not row.child_transcript_present for row in rows),
        "prefix-only child candidates": sum(bool(row.link_candidates) for row in rows),
        "unknown child usage": sum(row.child_tokens is None for row in rows),
        "unknown runtime": sum(row.spawn_to_final_seconds is None for row in rows),
    }
    items = "".join(f"<li>{_esc(name)}: {count}/{len(rows)}</li>" for name, count in counts.items() if count)
    return f'<div class="panel"><ul>{items}</ul></div>' if items else '<div class="empty">No gaps recorded.</div>'


def _expense_tail(report: Report) -> str:
    tail = report.expense_tail
    if not tail.runs:
        return '<div class="empty">No positive priced runs available.</div>'
    assessed = Counter(row.assessment for row in tail.runs)
    parts = [
        "<p>Top 5% within each cost basis, including cutoff ties. "
        "Estimated costs are not invoices. A checked return does not establish that the spend was necessary.</p>",
        f"<p>Reviewed {tail.reviewed_runs}/{len(tail.runs)} selected runs; "
        f"{tail.missing_cost_runs} missing costs; {tail.zero_cost_runs} zero costs excluded. "
        + "; ".join(f"{_esc(name.replace('_', ' '))}: {count}" for name, count in sorted(assessed.items()))
        + "</p>",
    ]
    for basis, n in tail.cohort_sizes.items():
        parts.append(
            f"<p>{_esc(basis)}: {n} priced runs; selected share of spend {tail.selected_cost_share[basis]:.1%}.</p>"
        )
    parts.append(
        '<div class="scroll"><table><thead><tr><th>run</th><th>cost / basis</th>'
        "<th>workload</th><th>assessment</th><th>action and evidence</th></tr></thead><tbody>"
    )
    by_id = {run.analysis.trace_id: run for run in report.runs}
    for row in tail.runs:
        run = by_id.get(row.trace_id)
        features = next((fs for fs in (run.features or []) if fs.scope == "expense"), None) if run else None
        detail = row.model_dump(mode="json")
        if features:
            detail["semantic_features"] = features.model_dump(mode="json")
        workload = row.workload
        coverage = workload.get("reviewed_root_token_share")
        coverage_text = (
            f"{coverage:.0%} root token coverage" if isinstance(coverage, (int, float)) else "coverage unknown"
        )
        child_text = "estimated child usage" if row.estimated_child_usage else "child usage observed or absent"
        items = workload.get("reviewed_work_items")
        total_items = workload.get("total_work_items")
        item_text = f"{items}/{total_items} work items" if items is not None else "work items unreviewed"
        parts.append(
            f'<tr id="expense-{_esc(row.trace_id)}"><td><a href="#run-{_esc(row.trace_id)}">{_esc(row.trace_id)}</a></td>'
            f'<td class="num">${row.cost:,.2f}<div class="run-meta">{_esc(row.basis)} · '
            f"{row.cost_share:.1%} of cohort</div></td>"
            f'<td>{_esc(item_text)}<div class="run-meta">{_esc(coverage_text)} · '
            f"{_esc(child_text)} · {_esc(workload.get('subagent_count', 0))} children</div></td>"
            f"<td>{_esc(row.assessment.replace('_', ' '))}</td>"
            f"<td>{_esc(row.action)}<details><summary>Source coordinates and JeV evidence</summary>"
            f"<pre>{_esc(json.dumps(detail, indent=2))}</pre></details></td></tr>"
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _model_work_mix(report: Report) -> str:
    groups: dict[str, list] = {}
    for run in report.runs:
        for row in run.tails.model_activity:
            groups.setdefault(row.category, []).append(row)
    if not groups:
        return '<div class="empty">No model usage could be joined to its requested actions.</div>'
    total = sum(r.tails.coverage.get("model_usage_observations", 0) for r in report.runs)
    attributed = sum(row.rounds for group in groups.values() for row in group)
    labels = {
        "inspection": "Inspection requests",
        "coordination": "Coordination requests",
        "execution": "Execution and edits",
        "mixed": "Mixed and other tools",
        "no_tools": "No tools in record",
    }
    rows = []
    for category, members in sorted(groups.items(), key=lambda item: -sum(row.rounds for row in item[1])):
        rounds = sum(row.rounds for row in members)
        priced = [row for row in members if row.estimated_cost is not None]
        cost = _money(sum(row.estimated_cost or 0 for row in priced)) if priced else "unknown"
        components = (
            "<br>".join(
                f"{label}: {_money(sum(row.estimated_cost_components.get(field, 0) for row in priced))}"
                for field, label in (
                    ("input", "uncached input"),
                    ("cache_read", "cached reads"),
                    ("cache_write", "cache writes"),
                    ("output", "output"),
                )
            )
            if priced
            else ""
        )
        fields = {
            key: sum(getattr(row.usage, key) or 0 for row in members)
            for key in ("input", "cache_read", "cache_write", "output")
        }
        rows.append(
            f'<tr><td>{_esc(labels[category])}</td><td class="num">{rounds:,}'
            f'<div class="run-meta">{sum(row.rounds for row in members if row.session_kind == "child"):,} child rounds</div></td>'
            f'<td class="num">{sum(row.tool_calls for row in members):,}</td>'
            f'<td class="num">{fields["input"]:,} uncached<br>{fields["cache_read"]:,} cached'
            f"<br>{fields['cache_write']:,} cache write</td>"
            f'<td class="num">{fields["output"]:,}</td><td class="num">{cost}'
            f'<div class="run-meta">{sum(row.rounds for row in priced):,}/{rounds:,} rounds priced</div>'
            f"<details><summary>Charge components</summary>{components}</details></td></tr>"
        )
    selected = {r.trace_id for r in report.expense_tail.runs}
    comparison = []
    for label, cohort in (
        ("Top 5% cost selection", [r for r in report.runs if r.analysis.trace_id in selected]),
        ("Other runs", [r for r in report.runs if r.analysis.trace_id not in selected]),
    ):
        members = [row for run in cohort for row in run.tails.model_activity if row.estimated_cost is not None]
        charge = sum(row.estimated_cost or 0 for row in members)
        if not charge:
            continue
        cells = []
        for category in ("coordination", "inspection"):
            amount = sum(row.estimated_cost or 0 for row in members if row.category == category)
            cells.append(f'<td class="num">{_money(amount)} ({amount / charge:.1%})</td>')
        comparison.append(
            f'<tr><td>{label} ({len(cohort):,} runs)</td><td class="num">{_money(charge)}</td>'
            + "".join(cells)
            + "</tr>"
        )
    cost_cohorts = (
        (
            "<details><summary>Compare work mix in the most expensive runs</summary>"
            "<p>Shares use only priced, attributed model calls in each selection. "
            "Estimated child usage is excluded, so these totals differ from the full run cost ranking.</p>"
            '<div class="scroll"><table><thead><tr><th>run selection</th><th>attributed charge</th>'
            "<th>coordination</th><th>inspection</th></tr></thead><tbody>"
            + "".join(comparison)
            + "</tbody></table></div></details>"
        )
        if selected and comparison
        else ""
    )
    return (
        f"<p>{attributed:,}/{total:,} usage observations joined within the same source record or response ID. "
        "Missing joins stay outside this table. Categories describe the tools requested by a model response; "
        "inspection and coordination can be necessary. Charges are price-table estimates, not savings.</p>"
        '<div class="scroll"><table><thead><tr><th>model activity</th><th>rounds</th><th>tool calls</th>'
        "<th>input tokens</th><th>output tokens</th><th>estimated charge</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + _coordination_costs(report) + cost_cohorts
    )


def _coordination_costs(report: Report) -> str:
    groups: dict[str, list] = {}
    mixed_calls: Counter[str] = Counter()
    for run in report.runs:
        for row in run.tails.model_activity:
            if row.category == "coordination":
                groups.setdefault(row.coordination_kind or "unclassified", []).append(row)
            elif row.coordination_calls:
                mixed_calls.update(row.coordination_calls)
    if not groups:
        return ""
    rows = []
    for kind, members in sorted(groups.items(), key=lambda item: -sum(r.estimated_cost or 0 for r in item[1])):
        priced = [r for r in members if r.estimated_cost is not None]
        charge = sum(r.estimated_cost or 0 for r in priced) if priced else None
        calls: Counter[str] = Counter()
        for row in members:
            calls.update(row.coordination_calls)
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(calls.items()))
        rows.append(
            f"<tr><td>{_esc(kind)}<div class='run-meta'>{_esc(detail)}</div></td>"
            f"<td class='num'>{sum(r.rounds for r in members):,}</td>"
            f"<td class='num'>{sum(r.rounds for r in members if r.session_kind == 'child'):,}</td>"
            f"<td class='num'>{_money(charge)}<div class='run-meta'>{sum(r.rounds for r in priced):,} rounds priced</div></td>"
            f"<td class='num'>{_money(sum(r.estimated_cost_components.get('cache_write', 0) for r in priced) if priced else None)}</td>"
            f"<td class='num'>{_money(sum(r.estimated_cost_components.get('cache_read', 0) for r in priced) if priced else None)}</td>"
            "</tr>"
        )
    return (
        "<h3>Coordination charges by operation</h3>"
        "<p>Each model response is counted once. Responses mixing orchestration operations use the mixed row; "
        "their charge is not divided among individual tool calls. These rows sum to the coordination category above. "
        f"A further {sum(mixed_calls.values()):,} coordination calls share responses with other work and remain in the mixed-work category.</p>"
        "<div class='scroll'><table><thead><tr><th>operation</th><th>model rounds</th><th>child rounds</th>"
        "<th>estimated charge</th><th>cache-write charge</th><th>cache-read charge</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _tail_readable(row: TailIncident) -> str:
    events = ", ".join(map(str, row.event_indices[:8]))
    if len(row.event_indices) > 8:
        events += f" … ({len(row.event_indices)} events; full list below)"
    facts = [
        ("Session / events", f"{row.session_id}: {events}"),
        ("Alert threshold", display_metric(row.threshold, row.unit)),
    ]
    if row.median_ratio is not None:
        facts.append(
            (
                "Within-trace comparison",
                f"{row.median_ratio:.1f}x the median of {row.cohort_n} observations in {row.cohort}",
            )
        )
    evidence = row.evidence
    for key, label in {
        "tool": "Tool",
        "pending_results": "Pending results",
        "unchanged_pending_progress_results": "Repeated unchanged progress",
        "pending_only_model_rounds": "Model rounds requesting only pending probes",
        "failed_attempts": "Failed attempts",
        "intervening_edit_calls": "Intervening edits",
        "invocations": "Invocations",
        "explicit_zero_exit_invocations": "Explicit zero exits",
        "unknown_exit_invocations": "Unknown exit status",
        "cache_write_tokens": "Cache-write tokens",
        "child_transcript_present": "Child transcript available",
    }.items():
        if key in evidence:
            value = evidence[key]
            facts.append((label, f"{value:,}" if type(value) is int else str(value)))
    if "outcome" in evidence:
        facts.append(
            (
                "Tool result",
                "Error reported"
                if evidence["outcome"] is False
                else "No error reported"
                if evidence["outcome"] is True
                else "Unknown",
            )
        )
    parts = [
        '<dl class="evidence-facts">' + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in facts) + "</dl>"
    ]
    for key, label in (("request", "Invocation"), ("result", "Result excerpt")):
        if evidence.get(f"{key}_excerpt"):
            clipped = " (clipped)" if evidence.get(f"{key}_excerpt_clipped") else ""
            parts.append(f"<p><strong>{label}{clipped}</strong></p><pre>{_esc(evidence[key + '_excerpt'])}</pre>")
    if row.kind == "cache_rebuilds":
        transitions = sorted(
            evidence.get("transitions", []), key=lambda t: (-t.get("cache_write", 0), t["event_start"])
        )
        lines = []
        for item in transitions[:8]:
            gap = (
                display_metric(item["response_gap_seconds"], "seconds")
                if item.get("response_gap_seconds") is not None
                else "unknown"
            )
            lines.append(
                f"<tr><td>{_esc(item['prior_event_start'])} → {_esc(item['event_start'])}</td>"
                f"<td>{_esc(item['model'])}<div class='run-meta'>{_esc(item.get('coordination_kind') or 'other work')}</div></td>"
                f"<td>{_esc(gap)}</td><td class='num'>{item['prior_cache_read']:,} → 0 read / {item['cache_write']:,} write</td>"
                f"<td class='num'>{_money(item.get('cache_write_cost'))}<div class='run-meta'>rate match: {_esc(item.get('pricing_status', 'unknown'))}</div></td></tr>"
            )
        parts.append(
            f"<p>{len(transitions)} transitions; showing the {min(8, len(transitions))} largest writes. Gaps measure response timestamps, not cache age.</p>"
            "<div class='scroll'><table><thead><tr><th>prior → next event</th><th>model / operation</th>"
            "<th>response gap</th><th>cache tokens</th><th>estimated write charge</th></tr></thead><tbody>"
            + "".join(lines)
            + "</tbody></table></div>"
        )
    return "".join(parts)


def _tail_feature_checks(features: FeatureSet | None) -> str:
    if features is None or not features.values:
        return ""
    checks = []
    for value in features.values.values():
        if value.reason == "api_error":
            label = "Request failed"
        elif value.reason == "low_support" or value.abstains:
            label = "Uncertain"
        elif value.reason is not None or value.value is None:
            label = "Unavailable"
        elif isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
            positive = value.positive_threshold if value.positive_threshold is not None else 0.7
            negative = value.negative_threshold if value.negative_threshold is not None else 0.3
            label = (
                "Supported in excerpt"
                if value.value >= positive
                else "Evidence against in excerpt"
                if value.value <= negative
                else "Uncertain"
            )
        else:
            label = str(value.value)
        name = value.id.rsplit(".", 1)[-1].replace("_", " ").capitalize()
        checks.append(f"<dt>{_esc(name)}</dt><dd>{_esc(label)}</dd>")
    return (
        '<p><strong>Classifier checks</strong> · bounded trace evidence</p><dl class="evidence-facts">'
        + "".join(checks)
        + "</dl>"
    )


def _tail_ledger(report: Report) -> str:
    groups: dict[str, list[tuple[RunResult, TailIncident]]] = {}
    for run in report.runs:
        for row in run.tails.incidents:
            groups.setdefault(row.kind, []).append((run, row))
    if not groups:
        return '<div class="empty">No measurable tail observations.</div>'
    parts = [
        "<p>Largest observations by metric, including values below alert thresholds. "
        "Durations can overlap and are not additive savings. Open delegations are lower bounds, "
        "not proven hangs. Ratios use the same tool within one trace (at least five observations).</p>"
    ]
    linked = {example.target_id for priority in report.priorities for example in priority.examples}
    assessed = [row for members in groups.values() for _, row in members if row.assessment != "observation"]
    if assessed:
        counts = Counter(row.assessment for row in assessed)
        parts.append(
            "<p><strong>Repeated-work review:</strong> "
            f"{len(assessed)} assessed · {counts['supported_opportunity']} supported opportunities · "
            f"{counts['justified_repeat']} with a repeat reason · {counts['reuse_visible']} carrying prior findings · "
            f"{counts['unclear']} unclear. Unreviewed measurements remain observations. "
            "A context-reuse opportunity does not imply that the review can be skipped.</p>"
        )
    for kind, members in sorted(groups.items()):
        ranked = sorted(
            members, key=lambda pair: (pair[1].assessment != "supported_opportunity", -pair[1].value, pair[1].id)
        )
        visible = ranked[:20] + [pair for pair in ranked[20:] if f"tail-{pair[1].id}" in linked]
        crossings = sum(row.exceeds_threshold for _, row in ranked)
        eligible = sum(
            d.n
            for run in report.runs
            for d in run.tails.distributions
            if d.cohort == kind or d.cohort.startswith(kind + ":")
        )
        rate = f"{crossings / eligible:.1%}" if eligible else "unknown"
        threshold = ranked[0][1].threshold
        unit = ranked[0][1].unit
        parts.append(
            f"<h3>{_esc(kind.replace('_', ' ').capitalize())} · {crossings}/{eligible} measured cases cross the threshold ({rate})</h3>"
            f"<p>Alert at {_esc(display_metric(threshold, unit))}. Showing {len(visible)} of {len(ranked)} retained cases, including every linked action example. Supported changes appear first; full ledger in JSON.</p>"
        )
        if kind == "cache_rebuilds":
            parts.append(
                "<p>Adjacent responses in the same request and model changed from a cache hit to zero cache read "
                "and a large cache write. Missing usage, compaction, and recorded context edits break the sequence. "
                "Response gaps are not cache ages. Charges overlap the work mix; matching prompt prefixes, "
                "provider expiry, and avoidable spend are unproven.</p>"
            )
        for run, row in visible:
            ratio = f" · {row.median_ratio:.1f}x median" if row.median_ratio is not None else ""
            charge = (
                f" · {_money(row.evidence.get('observed_cache_write_cost'))} cache-write charge "
                f"({row.evidence.get('priced_transitions', 0)} transitions priced)"
                if kind == "cache_rebuilds"
                else ""
            )
            detail = {"trace_id": run.analysis.trace_id, **row.model_dump(mode="json")}
            features = next((fs for fs in (run.features or []) if fs.scope == "tail" and fs.object_id == row.id), None)
            if features is not None:
                detail["semantic_features"] = features.model_dump(mode="json")
            errors = sum(v.reason == "api_error" for v in features.values.values()) if features else 0
            review_gap = (
                f'<p class="notice"><strong>Review incomplete:</strong> {errors} classifier requests failed. '
                "The measurement is retained; its interpretation is incomplete.</p>"
                if errors
                else ""
            )
            parts.append(
                f'<details class="tail-detail" id="tail-{_esc(row.id)}"><summary>{_esc(row.label)}: '
                f"{_esc(display_metric(row.value, row.unit))}{_esc(ratio)}{charge} · "
                f"{_esc(row.session_id)}:{row.event_indices[0]}</summary>"
                f"<p><strong>{_esc(row.assessment.replace('_', ' '))}</strong> · {_esc(row.action)}</p>"
                f"{review_gap}{_tail_readable(row)}{_tail_feature_checks(features)}<details><summary>Full source record and classifier evidence</summary>"
                f"<pre>{_esc(json.dumps(detail, indent=2))}</pre></details></details>"
            )
    parts.append(
        "<details><summary>Timing coverage and per-trace distributions</summary><pre>"
        + _esc(
            json.dumps(
                [
                    {
                        "trace_id": run.analysis.trace_id,
                        "coverage": run.tails.coverage,
                        "distributions": [d.model_dump() for d in run.tails.distributions],
                    }
                    for run in report.runs
                ],
                indent=2,
            )
        )
        + "</pre></details>"
    )
    return "".join(parts)


def render_html(report: Report, *, back_href: str | None = None, title: str | None = None) -> str:
    """Render the full self-contained HTML dashboard as a string."""
    model_count = len(report.aggregate.by_model)
    experiment_count = len(report.aggregate.by_experiment)
    task_html = _task_cards(report)
    monthly_html = _monthly(report.monthly)
    body = (
        '<main class="shell">'
        '<header><div class="eyebrow">Trace intelligence · cross-harness review</div>'
        + (f'<p><a href="{_esc(back_href)}">Back to report</a></p>' if back_href else "")
        + f"<h1>{_esc(title or 'Where to reduce cost and delay')}</h1>"
        '<p class="lede">Start with supported changes and the largest measured burdens. Each action names the evidence, a next step, and how to check improvement while preserving task quality.</p>'
        f"{_meta(report)}</header>"
        '<nav class="report-nav" aria-label="Report sections"><a href="#signals-worth-acting-on">Actions</a>'
        '<a href="#model-work-mix">Cost breakdown</a><a href="#execution-extremes">Execution extremes</a>'
        '<a href="#most-expensive-5percent">Expensive runs</a><a href="#delegation-handovers">Delegation and cache waits</a>'
        '<a href="#run-explorer">Run explorer</a></nav>'
        + _summary(report)
        + _section("Signals worth acting on", _priorities(report), "Evidence → next step → verification")
        + '<details class="section"><summary>Batch context and coverage</summary>'
        + _insights(report)
        + "</details>"
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
        + _section("Most expensive 5%", _expense_tail(report))
        + _section("Model work mix", _model_work_mix(report))
        + _section("Execution extremes", _tail_ledger(report))
        + _section("Delegation handovers", _handover_ledger(report), "One row per observed new spawn")
        + _section(
            "Handover unknowns", _handover_unknowns(report), "Counts excluded from plaintext and timing denominators"
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
        f"<style>{_CSS}</style></head><body>{body}{_LINK_SCRIPT}</body></html>"
    )


__all__ = ["render_html"]
