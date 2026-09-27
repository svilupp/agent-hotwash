"""The compact view keeps evidence generation independent of presentation."""

from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import Finding, Severity
from agent_hotwash.report.highlights import HighlightsData, build_highlights_data, render_highlights
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.report.ranking import Judgment, RankedTheme, score_theme
from agent_hotwash.runner import run_trace


def test_highlights_round_trip_omits_per_run_payload(tf):
    run = run_trace(tf.trace(tf.session([tf.user("work")])), load_config(), semantic_mode="off")
    run.findings.append(
        Finding(
            id="EXAMPLE",
            kind="smell",
            severity=Severity.high,
            confidence="high",
            session_id="s0",
            message="Visible evidence <script>bad()</script>",
        )
    )
    report = Report.build([run], ReportMeta(tool_version="test"))
    brief = build_highlights_data(report)
    serialized = brief.model_dump_json()
    assert '"runs"' not in serialized and '"findings"' not in serialized
    html = render_highlights(HighlightsData.model_validate_json(serialized))
    assert "<h3>Example</h3>" in html
    assert "&lt;script&gt;" in html and "<script>bad()" not in html
    assert len(html.encode()) < 20_000


def test_low_confidence_semantic_judgment_cannot_dominate_score():
    theme = RankedTheme(
        id="a",
        kind="action",
        title="A",
        status="investigate",
        affected_runs=3,
        incidents=4,
        evidence="Observed",
        next_step="Inspect",
        verify="Measure",
        limit="Unknown",
    )
    confident = score_theme(theme, 10, judgment=Judgment(actionability=1, ease=1, confidence=1))
    uncertain = score_theme(theme, 10, judgment=Judgment(actionability=1, ease=1, confidence=0.1))
    assert confident.score > uncertain.score
    assert sum(uncertain.components.values()) == uncertain.score


def test_rare_high_severity_signal_gets_highlight_slot():
    themes = [
        RankedTheme(
            id=f"detector-{i}",
            kind="detector",
            title=f"Signal {i}",
            status="signal",
            affected_runs=20,
            incidents=20,
            severity="low" if i < 5 else "high",
            evidence="Observed",
            next_step="Inspect",
            verify="Measure",
            limit="May be false positive",
            score=80 - i,
            member_signals={f"SIGNAL_{i}": 20},
        )
        for i in range(6)
    ]
    html = render_highlights(
        HighlightsData(traces=20, total_cost=None, total_findings=120, detector_signals=6, themes=themes)
    )
    assert '<article id="detector-5">' in html
    assert '<article id="detector-4">' not in html
    assert "<code>SIGNAL_5</code>" in html
    assert "20 runs · 20 detector occurrences" in html


def test_remaining_actions_are_browsable_without_full_cards():
    themes = [
        RankedTheme(
            id=f"action-{i}",
            kind="action",
            title=f"Action {i}",
            status="investigate",
            affected_runs=i + 1,
            incidents=i + 1,
            evidence="Observed",
            next_step="Inspect",
            verify="Measure",
            limit="Unknown",
            score=70 - i,
            components={"affected_runs": 10, "observed_spend": 0},
        )
        for i in range(7)
    ]
    html = render_highlights(
        HighlightsData(traces=20, total_cost=None, total_findings=0, detector_signals=0, themes=themes)
    )
    assert 'href="#actions"' in html and 'href="#all-signals"' in html
    assert '<article id="action-4">' in html
    assert '<article id="action-5">' not in html
    assert "Browse 2 more action themes" in html
    assert "Action 5" in html and "Action 6" in html
    assert html.index('<h2 id="actions">') < html.index('<h2 id="cost-work">')
    assert html.index('<h2 id="actions">') < html.index('<h2 id="failure-actions">')
    assert "Score breakdown" in html and "affected runs</dt><dd>10 points" in html
    assert "Observed charge unknown; spend adds no score points." in html
    assert "Action scores combine affected runs (30 points)" in html
    assert "Detector scores combine affected runs (60)" in html
    assert "No detector signals in this slice." in html


def test_selected_action_ids_rescue_omitted_action_without_duplicate_cards():
    themes = [
        RankedTheme(
            id=f"action-{i}",
            kind="action",
            title=f"Action {i}",
            status="investigate",
            affected_runs=i + 1,
            incidents=i + 1,
            evidence="Observed",
            next_step="Inspect",
            verify="Measure",
            limit="Unknown",
        )
        for i in range(7)
    ]
    data = HighlightsData(
        traces=7,
        total_cost=None,
        total_findings=0,
        detector_signals=0,
        themes=themes,
        highlight_action_ids=["missing", "action-6", "action-1", "action-6"],
    )
    html = render_highlights(HighlightsData.model_validate_json(data.model_dump_json()))
    assert html.index('<article id="action-6">') < html.index('<article id="action-1">')
    assert html.count('<article id="action-6">') == 1
    assert html.count('<article id="action-1">') == 1
    assert '<article id="action-0">' not in html
    assert "Browse 5 more action themes" in html
    assert "Action 0</td>" in html
    assert "Action 6</td>" not in html
    assert "Action 1</td>" not in html


def test_card_bounds_free_text_from_brief():
    theme = RankedTheme(
        id="action-large",
        kind="action",
        title="Large",
        status="investigate",
        affected_runs=1,
        incidents=1,
        evidence="E" * 10_000,
        next_step="Inspect",
        verify="Measure",
        limit="Unknown",
    )
    html = render_highlights(
        HighlightsData(traces=1, total_cost=None, total_findings=0, detector_signals=0, themes=[theme])
    )
    assert "E" * 360 not in html
    assert "E" * 300 in html
    assert len(html.encode()) < 10_000


def test_action_example_prefers_verified_incident_link():
    theme = RankedTheme(
        id="investigate-tool_latency",
        kind="action",
        title="Inspect latency",
        status="investigate",
        affected_runs=1,
        incidents=1,
        evidence="One slow call",
        next_step="Inspect",
        verify="Measure",
        limit="Timing only",
        example_trace_ids=["trace-1"],
    )
    data = HighlightsData(traces=1, total_cost=None, total_findings=0, detector_signals=0, themes=[theme])
    html = render_highlights(
        data,
        run_links={"trace-1": "report-evidence/runs/run-000001.html"},
        example_links={(theme.id, "trace-1"): "report-evidence/runs/run-000001.html#tail-1"},
    )
    assert 'href="report-evidence/runs/run-000001.html#tail-1"' in html
