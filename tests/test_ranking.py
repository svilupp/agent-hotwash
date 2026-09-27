"""Ranking keeps measured evidence separate from editorial attention scores."""

from types import SimpleNamespace
from typing import Any, cast

from agent_hotwash.report.ranking import Judgment, RankedTheme, build_themes, score_theme


def _theme(**changes):
    data = {
        "id": "investigate-example",
        "kind": "action",
        "title": "Inspect example",
        "status": "investigate",
        "affected_runs": 2,
        "incidents": 3,
        "evidence": "Measured",
        "next_step": "Inspect",
        "verify": "Compare",
        "limit": "Unknown cause",
    }
    data.update(changes)
    return RankedTheme.model_validate(data)


def test_action_score_keeps_unknown_charge_and_uses_cohort_reference():
    unknown = score_theme(_theme(), 10, cost_reference=200)
    known = score_theme(_theme(observed_cost=100), 10, cost_reference=200)
    assert unknown.observed_cost is None
    assert unknown.components["observed_spend"] == 0
    assert known.components["observed_spend"] == 7.5
    assert known.score_basis == "action_attention_v2"
    assert known.score == round(sum(known.components.values()), 2)
    # A direct score without a report cohort does not invent a dollar target.
    assert score_theme(_theme(observed_cost=100), 10).components["observed_spend"] == 0


def test_confidence_shrinks_judgment_toward_status_default():
    theme = _theme()
    baseline = score_theme(theme, 10)
    uncertain = score_theme(theme, 10, judgment=Judgment(actionability=1, ease=1, confidence=0))
    partial = score_theme(theme, 10, judgment=Judgment(actionability=1, ease=1, confidence=0.5))
    confident = score_theme(theme, 10, judgment=Judgment(actionability=1, ease=1, confidence=1))
    assert baseline.score == uncertain.score
    assert uncertain.score < partial.score < confident.score


def test_detector_score_has_no_invented_intervention_components():
    detector = score_theme(_theme(kind="detector", status="signal", severity="high"), 10)
    assert detector.score_basis == "detector_attention_v2"
    assert set(detector.components) == {"affected_runs", "max_severity"}
    assert detector.components["max_severity"] == 40


def test_family_groups_union_runs_and_retain_member_counts():
    def run(trace_id, *ids):
        return SimpleNamespace(
            analysis=SimpleNamespace(trace_id=trace_id),
            findings=[SimpleNamespace(id=detector_id, message=f"Evidence for {detector_id}") for detector_id in ids],
        )

    report = SimpleNamespace(
        priorities=[],
        runs=[
            run("trace-1", "LOOKS_RIGHT_RUNS_WRONG", "UNVERIFIED_COMPLETION", "LINEAR_SCAN"),
            run("trace-2", "UNVERIFIED_COMPLETION", "LINEAR_SCAN", "linear_scan_search"),
        ],
        finding_histogram={
            "LOOKS_RIGHT_RUNS_WRONG": 1,
            "UNVERIFIED_COMPLETION": 2,
            "LINEAR_SCAN": 2,
            "linear_scan_search": 1,
        },
        finding_severity={
            "LOOKS_RIGHT_RUNS_WRONG": {"medium": 1},
            "UNVERIFIED_COMPLETION": {"medium": 2},
            "LINEAR_SCAN": {"low": 2},
            "linear_scan_search": {"low": 1},
        },
    )
    themes = {theme.id: theme for theme in build_themes(cast("Any", report))}
    verification = themes["detector-family-post_edit_verification"]
    assert verification.affected_runs == 2
    assert verification.incidents == 3  # Occurrences may overlap; not distinct incidents.
    assert verification.member_signals == {"LOOKS_RIGHT_RUNS_WRONG": 1, "UNVERIFIED_COMPLETION": 2}
    assert verification.severity == "medium"
    assert verification.example_trace_ids == ["trace-1", "trace-2"]
    assert themes["detector-family-linear_search"].affected_runs == 2
    assert "detector-LINEAR_SCAN" not in themes


def test_action_detector_alias_is_suppressed_without_losing_action_counts():
    action = SimpleNamespace(
        id="investigate-cache_rebuilds",
        title="Inspect cache writes",
        status="investigate",
        affected_runs=3,
        incidents=4,
        observed_cost=50.0,
        cost_basis="estimated charge",
        evidence="Four groups",
        next_step="Inspect",
        verify="Compare",
        limit="Overlap",
        examples=[],
    )
    report = SimpleNamespace(
        priorities=[action],
        runs=[SimpleNamespace(analysis=SimpleNamespace(trace_id="trace-1"), findings=[])],
        finding_histogram={"TAIL_CACHE_REBUILDS": 4},
        finding_severity={"TAIL_CACHE_REBUILDS": {"low": 4}},
    )
    themes = build_themes(cast("Any", report))
    assert len(themes) == 1
    assert themes[0].incidents == 4
    assert themes[0].observed_cost == 50.0
