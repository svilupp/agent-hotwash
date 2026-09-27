"""Small, selective HTML view over the complete report data model."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent_hotwash.report.error_actions import ErrorActionAccumulator, ErrorActionGroup, ErrorActionSummary
from agent_hotwash.report.overview import SpendOverview, build_spend_overview
from agent_hotwash.report.ranking import RankedTheme, build_themes

if TYPE_CHECKING:
    from agent_hotwash.report.model import Report


_CSS = """body{font:16px/1.55 system-ui,sans-serif;color:#19252a;background:#f4f5f2;margin:0}
main{max-width:1040px;margin:auto;padding:36px 24px 80px}h1{font-size:2.5rem;line-height:1.1}
h2{margin-top:48px}p{max-width:78ch}.muted,small{color:#56646a}.metrics,.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.metric,article,details,.note{
background:white;border:1px solid #d8dfd9;border-radius:12px;padding:18px}.metric strong{display:block;
font-size:1.8rem}article{margin:12px 0}article h3{margin:4px 0}.tag{font-size:.76rem;
font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#356f65}.score{float:right;
font-weight:700;color:#155c50}dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px}
dt{font-weight:700}dd{margin:0}table{border-collapse:collapse;width:100%;background:white}td,th{
padding:8px;border-bottom:1px solid #e1e5e0;text-align:left}th:last-child,td:last-child{text-align:right}
.scroll{overflow:auto}.note{border-left:4px solid #bd7737}code{font-size:.85em;overflow-wrap:anywhere}
nav a{display:inline-block;margin:0 16px 8px 0}a{color:#155c50}a:focus-visible,summary:focus-visible{
outline:3px solid #bd7737;outline-offset:3px}.skip{position:absolute;left:-10000px}.skip:focus{left:12px;
top:12px;background:white;padding:8px;z-index:1}article details{padding:10px 14px;margin-top:12px}
summary{cursor:pointer;font-weight:600}table caption{text-align:left;font-weight:700;padding:8px}
"""


_DECISION_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Narrow costly work",
        "Inspect the largest tasks and accepted scope before setting a smaller budget or splitting work.",
        ("investigate-expense-estimated", "detector-overlong_trace", "detector-KITCHEN_SINK"),
    ),
    (
        "Check delegation and supervisor use",
        "Compare child outcomes, handoff burden, and parent waiting before changing supervisor or delegation use.",
        ("supported-status_probes", "investigate-parent_wait", "measure-delegation_open"),
    ),
    (
        "Improve context and cache use",
        "Measure repeated prefix writes and rereads before changing prompts, skills, or cache policy.",
        ("investigate-cache_rebuilds", "detector-TAIL_CONTEXT_REPLAY", "detector-COMPACTION_AMNESIA"),
    ),
    (
        "Fix agent tool and edit behavior",
        "Inspect rejected calls, repeated retries, and edits that missed current file state.",
        (
            "investigate-retry_attempts",
            "detector-EDIT_WITHOUT_READ",
            "detector-TOOL_ARG_MALFORMED",
            "detector-family-recovery_loops",
        ),
    ),
)


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _brief(value: str, limit: int = 360) -> str:
    """Bound free-text fields before embedding them in the compact page."""
    return _e(value if len(value) <= limit else value[: limit - 1].rstrip() + "…")


def _dollars(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "unpriced"


class HighlightsData(BaseModel):
    """Portable presentation state: no run payloads or model calls."""

    traces: int
    total_cost: float | None
    total_findings: int
    detector_signals: int
    harness_runs: dict[str, int] = Field(default_factory=dict)
    truth_known: int = 0
    truth_successes: int = 0
    themes: list[RankedTheme] = Field(default_factory=list)
    highlight_action_ids: list[str] = Field(default_factory=list)
    review_rescued_ids: list[str] = Field(default_factory=list)
    classifier_coverage: dict[str, str] = Field(default_factory=dict)
    spend_overview: SpendOverview | None = None
    error_actions: ErrorActionSummary | None = None


def build_highlights_data(report: Report) -> HighlightsData:
    from pathlib import Path

    from agent_hotwash.config import load_config

    error_acc = ErrorActionAccumulator()
    for run in report.runs:
        error_acc.add_run(run)
    pricing = load_config(Path(report.meta.config_path) if report.meta.config_path else None)
    return HighlightsData(
        traces=len(report.runs),
        total_cost=report.aggregate.overall.total_cost,
        total_findings=report.total_findings(),
        detector_signals=len(report.finding_histogram),
        harness_runs={str(name): group.n for name, group in report.aggregate.by_agent.items()},
        truth_known=sum(
            run.analysis.outcome.ground_truth_resolved is not None or run.analysis.resolved is not None
            for run in report.runs
        ),
        truth_successes=sum(
            (run.analysis.outcome.ground_truth_resolved is True)
            or (run.analysis.outcome.ground_truth_resolved is None and run.analysis.resolved is True)
            for run in report.runs
        ),
        themes=build_themes(report),
        classifier_coverage={scope: row.summary() for scope, row in report.classifier_coverage.items()},
        spend_overview=build_spend_overview(report.runs, pricing=pricing),
        error_actions=error_acc.summary(),
    )


def _trace_examples(
    trace_ids: list[str],
    run_links: Mapping[str, str],
    *,
    theme_id: str | None = None,
    example_links: Mapping[tuple[str, str], str] | None = None,
) -> str:
    examples = []
    for trace_id in trace_ids:
        label = f"<code>{_brief(trace_id, 120)}</code>"
        href = example_links.get((theme_id, trace_id)) if theme_id and example_links else None
        href = href or run_links.get(trace_id)
        examples.append(f'<a href="{_e(href)}">{label}</a>' if href else label)
    return ", ".join(examples) or "none recorded"


def _card(
    theme: RankedTheme,
    *,
    discovery: bool = False,
    run_links: Mapping[str, str] | None = None,
    example_links: Mapping[tuple[str, str], str] | None = None,
) -> str:
    discovery_note = (
        '<p class="note">JeV discovery candidate: review this measurement or trial before rollout.</p>'
        if discovery
        else ""
    )
    charge = (
        f"<p><strong>${theme.observed_cost:,.2f}</strong> {_e(theme.cost_basis or 'observed charge')}</p>"
        if theme.observed_cost is not None
        else ""
    )
    traces = _trace_examples(
        theme.example_trace_ids[:3], run_links or {}, theme_id=theme.id, example_links=example_links
    )
    missing_charge = (
        '<p class="muted">Observed charge unknown; spend adds no score points.</p>'
        if theme.kind == "action" and theme.observed_cost is None and theme.components
        else ""
    )
    score_parts = (
        "<details><summary>Score breakdown</summary><dl>"
        + "".join(
            f"<dt>{_e(name.replace('_', ' '))}</dt><dd>{value:g} points</dd>"
            for name, value in theme.components.items()
        )
        + f"</dl>{missing_charge}</details>"
        if theme.components
        else ""
    )
    return (
        f'<article id="{_e(theme.id)}"><span class="score" '
        f'aria-label="{_e(theme.kind)} attention score">{theme.score:.0f}/100</span>'
        f'<div class="tag">{_e(theme.status)} · {_e(theme.kind)}</div><h3>{_e(theme.title)}</h3>'
        f"{discovery_note}"
        f"<p>{theme.affected_runs:,} runs · {theme.incidents:,} "
        f"{'detector occurrences' if theme.kind == 'detector' else 'cases'}"
        f"{' · max severity ' + _e(theme.severity) if theme.severity else ''}</p>"
        f"<p>{_brief(theme.evidence)}</p>{charge}<p><strong>Next:</strong> {_brief(theme.next_step)}</p>"
        f"<p><strong>Check:</strong> {_brief(theme.verify)}</p>"
        f'<p class="muted"><strong>Limit:</strong> {_brief(theme.limit)}</p>'
        f"<small>Example trace IDs: {traces}</small>"
        f"{score_parts}</article>"
    )


def _error_card(group: ErrorActionGroup, run_links: Mapping[str, str] | None = None) -> str:
    leaves = ", ".join(
        f"{leaf} ({count:,})" for leaf, count in sorted(group.leaf_counts.items(), key=lambda item: -item[1])[:3]
    )
    traces = _trace_examples([example.trace_id for example in group.examples[:2]], run_links or {})
    return (
        f"<article><div class='tag'>{group.count:,} failed results · {_e(group.id.replace('_', ' '))}</div>"
        f"<h3>{_e(group.title)}</h3><p>{_e(group.next_step)}</p>"
        f"<p class='muted'>Common leaves: {_e(leaves)}</p>"
        f"<small>Example trace IDs: {traces}</small></article>"
    )


def _decision_map(themes: list[RankedTheme]) -> str:
    by_id = {theme.id: theme for theme in themes}
    cards = []
    for title, decision, member_ids in _DECISION_GROUPS:
        members = [by_id[theme_id] for theme_id in member_ids if theme_id in by_id]
        if not members:
            continue
        links = "".join(
            f"<li><a href='#{_e(theme.id if theme.kind == 'action' else 'index-' + theme.id)}'>"
            f"{_e(theme.title)}</a> "
            f"<small>({theme.affected_runs:,} runs; {_e(theme.status)})</small></li>"
            for theme in members
        )
        cards.append(f"<article><h3>{_e(title)}</h3><p>{_e(decision)}</p><ul>{links}</ul></article>")
    return (
        '<h2 id="decision-map">Choose the next investigation</h2>'
        "<p>Related themes share a decision to make; their counts can overlap and do not imply one cause.</p>"
        "<article><h3>Choose a model for a task</h3><p>Use the model and task spend tables to find "
        "matched work, then compare completion and repeated tool failures. Spend alone cannot identify a better model. "
        "<a href='#cost-work'>Inspect model and task spend</a>.</p></article>" + "".join(cards)
    )


def render_highlights(
    report: Report | HighlightsData,
    *,
    run_links: Mapping[str, str] | None = None,
    example_links: Mapping[tuple[str, str], str] | None = None,
    runs_index_href: str | None = None,
) -> str:
    """Show the top themes and quantitative map without embedding trace records."""
    data = build_highlights_data(report) if not isinstance(report, HighlightsData) else report
    run_links = run_links or {}
    themes = data.themes
    actions = [theme for theme in themes if theme.kind == "action"]
    if data.highlight_action_ids:
        actions_by_id = {theme.id: theme for theme in actions}
        selected_actions = []
        selected_action_ids: set[str] = set()
        for action_id in data.highlight_action_ids:
            if action_id in actions_by_id and action_id not in selected_action_ids:
                selected_actions.append(actions_by_id[action_id])
                selected_action_ids.add(action_id)
            if len(selected_actions) == 5:
                break
        remaining_actions = [theme for theme in actions if theme.id not in selected_action_ids]
    else:
        selected_actions = actions[:5]
        remaining_actions = actions[5:]
    signals = [theme for theme in themes if theme.kind == "detector"]
    selected_signals = signals[:5]
    severe = next((theme for theme in signals if theme.severity == "high"), None)
    if severe is not None and severe not in selected_signals:
        selected_signals[-1] = severe
    metrics = [
        (data.traces, "traces"),
        (f"${data.total_cost:,.2f}" if data.total_cost is not None else "unknown", "reported or estimated spend"),
        (data.total_findings, "detector occurrences"),
        (data.detector_signals, "distinct detector signals"),
        (
            f"{data.truth_successes}/{data.truth_known}" if data.truth_known else "unavailable",
            "recorded outcome success",
        ),
    ]
    metric_html = "".join(
        f'<div class="metric"><strong>{_e(value)}</strong>{_e(label)}</div>' for value, label in metrics
    )
    harness_note = ", ".join(f"{name}: {count:,}" for name, count in data.harness_runs.items())
    coverage = data.classifier_coverage
    coverage_html = (
        "<ul>" + "".join(f"<li>{_e(scope)}: {_e(summary)}</li>" for scope, summary in coverage.items()) + "</ul>"
        if coverage
        else "<p>No classifier review records are available.</p>"
    )
    detector_rows = "".join(
        f"<tr id='index-{_e(row.id)}'><td>{_e(row.title)}</td>"
        f"<td>{', '.join(f'<code>{_e(name)}</code>' for name in row.member_signals) or '—'}</td>"
        f"<td>{row.affected_runs:,}</td><td>{_e(row.severity or 'unknown')}</td>"
        f"<td>{row.incidents:,}</td></tr>"
        for row in signals
    )
    action_rows = "".join(
        f"<tr id='{_e(row.id)}'><td>{_e(row.title)}</td><td>{_e(row.status)}</td>"
        f"<td>{row.affected_runs:,}</td><td>{row.score:.0f}</td></tr>"
        for row in remaining_actions
    )
    more_actions = (
        f"<details><summary>Browse {len(remaining_actions)} more action themes</summary>"
        '<div class="scroll"><table><caption>Remaining action themes</caption><thead><tr>'
        '<th scope="col">Theme</th><th scope="col">Status</th><th scope="col">Runs</th>'
        '<th scope="col">Score</th></tr></thead><tbody>'
        f"{action_rows}</tbody></table></div></details>"
        if remaining_actions
        else ""
    )
    omitted_signals = len(signals) - len(selected_signals)
    spend = data.spend_overview
    if spend is not None and spend.calls:
        model_rows = "".join(
            f"<tr><td>{_e(row.name)}</td><td>{row.calls:,}</td>"
            f"<td>{row.unknown_calls:,}</td><td>{_e(_dollars(row.invoice.amount))}</td></tr>"
            for row in spend.by_model[:8]
        )
        task_rows = "".join(
            f"<tr><td>{_brief(row.label, 110)}<br><small><code>{_e(row.task_id)}</code></small></td>"
            f"<td>{_e(', '.join(model.name for model in row.by_model[:3]))}</td>"
            f"<td>{_e(_dollars(row.invoice.amount))}</td>"
            f"<td><code>{_e(', '.join(row.trace_ids[:2]))}</code></td></tr>"
            for row in spend.top_tasks[:6]
        )
        write_charge = (
            f"; {_e(_dollars(spend.cache_write_charge.amount))} at configured prices"
            if spend.cache_write_charge is not None
            else "; charge unavailable"
        )
        spend_html = (
            '<h2 id="cost-work">Cost and work</h2>'
            f"<p>{_e(_dollars(spend.invoice.amount))} in per-response invoice estimates across "
            f"{spend.calls:,} counted response records; {spend.exact_calls:,} use exact pricing, "
            f"{spend.estimated_calls:,} use estimated pricing, and {spend.unknown_calls:,} have unknown pricing. "
            f"{spend.anonymous_calls:,} lack a stable response ID and cannot be deduplicated across runs. "
            f"Cache writes: {spend.cache_write_tokens:,} tokens{write_charge}. "
            "These figures describe recorded usage and configured prices, not avoidable savings.</p>"
            '<div class="scroll"><table><caption>Response spend by model</caption><thead><tr>'
            '<th scope="col">Model</th><th scope="col">Calls</th><th scope="col">Unpriced</th>'
            '<th scope="col">Priced estimate</th></tr></thead><tbody>'
            f"{model_rows}</tbody></table></div>"
            "<p class='muted'>Model totals describe where charges occurred. Compare matched tasks and outcomes "
            "before changing the model for a task.</p>"
            '<div class="scroll"><table><caption>Most expensive root tasks with available invoices</caption>'
            '<thead><tr><th scope="col">Task</th><th scope="col">Models used</th>'
            '<th scope="col">Priced estimate</th><th scope="col">Example traces</th></tr></thead><tbody>'
            f"{task_rows}</tbody></table></div>"
            f"<p class='muted'>Showing {min(6, len(spend.top_tasks))} of {spend.total_tasks:,} root tasks. "
            "Task type is unavailable where the trace has no reliable category.</p>"
        )
    else:
        spend_html = '<h2 id="cost-work">Cost and work</h2><p>Response-level spend is unavailable in this slice.</p>'
    errors = data.error_actions
    if errors is not None and errors.total:
        review_groups = sorted(
            (group for group in errors.groups if group.urgency == "review" and group.id != "unknown"),
            key=lambda group: (-group.count, group.id),
        )
        error_cards = "".join(_error_card(group, run_links) for group in review_groups[:5])
        ordinary_groups = [group for group in errors.groups if group.urgency == "low"]
        ordinary = sum(group.count for group in ordinary_groups)
        ordinary_leaves = ", ".join(
            f"{leaf.replace('_', ' ')} ({count:,})"
            for group in ordinary_groups
            for leaf, count in sorted(group.leaf_counts.items(), key=lambda item: -item[1])[:4]
        )
        error_html = (
            '<h2 id="failure-actions">Failures by next action</h2>'
            f"<p>{errors.total:,} failed tool results grouped once each. "
            f"{ordinary:,} are expected coding checks or negative observations. "
            f"{errors.unknown_count:,} ({errors.unknown_share:.0%}) are unresolved and need trace review "
            "before a policy or model change.</p>"
            + (error_cards or "<p>No attributable failure action group is available.</p>")
            + f"<details><summary>Expected coding iteration and negative checks ({ordinary:,})</summary>"
            f"<p class='muted'>{_e(ordinary_leaves)}</p></details>"
        )
    else:
        error_html = '<h2 id="failure-actions">Failures by next action</h2><p>No failure records are available.</p>'
    body = (
        '<a class="skip" href="#main">Skip to findings</a><main id="main"><header>'
        '<div class="tag">Agent hotwash · trace review</div>'
        '<h1>Where to look first</h1><p class="muted">Grouped findings ranked for review. '
        "The score is a provisional attention order, not measured savings or a quality verdict.</p>"
        '<nav aria-label="Report sections"><a href="#actions">Actions</a>'
        '<a href="#decision-map">Decision map</a><a href="#signals">Signals</a>'
        '<a href="#cost-work">Cost and work</a><a href="#failure-actions">Failure actions</a>'
        '<a href="#method">Scoring</a>'
        '<a href="#all-signals">All detector groups</a>'
        + (f'<a href="{_e(runs_index_href)}">Run evidence</a>' if runs_index_href else "")
        + "</nav></header>"
        f'<div class="metrics">{metric_html}</div><p class="muted">Harnesses: {_e(harness_note or "unknown")}</p>'
        '<div class="note"><strong>Interpretation:</strong> Observed costs can overlap. '
        "Duration is not recoverable time. Detector counts include possible false positives. "
        "No accepted task outcome rate is claimed without recorded truth.</div>"
        + '<h2 id="actions">Actions to examine</h2>'
        + (
            "".join(
                _card(
                    row,
                    discovery=row.id in data.review_rescued_ids,
                    run_links=run_links,
                    example_links=example_links,
                )
                for row in selected_actions
            )
            or "<p>No action candidates in this slice.</p>"
        )
        + more_actions
        + _decision_map(themes)
        + '<h2 id="signals">Signals to investigate</h2>'
        + (
            "".join(_card(row, run_links=run_links) for row in selected_signals)
            or "<p>No detector signals in this slice.</p>"
        )
        + f'<p class="muted">{omitted_signals} further detector themes appear in the index below.</p>'
        + spend_html
        + error_html
        + '<h2 id="method">How this order was made</h2><p>Scores order themes within their own lane. '
        "Action scores combine affected runs (30 points), observed charge relative to the largest action charge "
        "in this report (15), actionability (35), and ease of testing (20). Detector scores combine affected runs "
        "(60) and maximum recorded severity (40). These weights guide attention; they are not calibrated savings "
        "or risk estimates. Missing charge remains unknown and adds no score points. Unreviewed actionability and "
        "ease use status defaults; a low-confidence review moves those estimates only partway from the defaults. "
        "Open a card's score breakdown to see its components. A JeV discovery candidate, when shown, uses "
        "separate conservative semantic checks to surface one otherwise omitted theme; its original status "
        "and numeric score remain unchanged.</p>"
        "<details><summary>Classifier coverage</summary>" + coverage_html + "</details>"
        '<h2 id="all-signals">All detector groups</h2><div class="scroll"><table>'
        "<caption>All detector groups by ranked order; groups may include several detectors</caption><thead><tr>"
        '<th scope="col">Signal</th><th scope="col">Included detectors</th><th scope="col">Runs</th>'
        '<th scope="col">Severity</th><th scope="col">Occurrences</th></tr></thead>'
        f"<tbody>{detector_rows}</tbody></table></div>"
        + '<footer><p class="muted">Full per-run findings, source coordinates, and quantitative records are '
        + (
            f'available in the <a href="{_e(runs_index_href)}">run evidence index</a>.'
            if runs_index_href
            else "available in the JSON report and the optional full HTML view."
        )
        + "</p></footer></main>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>agent-hotwash · highlights</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )


__all__ = ["HighlightsData", "build_highlights_data", "render_highlights"]
