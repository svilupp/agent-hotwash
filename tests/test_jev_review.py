"""Conservative semantic discovery leaves the quantitative rank intact."""

from __future__ import annotations

from typing import cast

import pytest

from agent_hotwash.report.highlights import HighlightsData, render_highlights
from agent_hotwash.report.jev_review import QUESTIONS, decide, project_action_state, review_brief
from agent_hotwash.report.priorities import ActionReviewContext
from agent_hotwash.report.ranking import RankedTheme
from agent_hotwash.semantic.client import SystemOneAsker


def _features(**updates: float) -> dict[str, float]:
    values = dict.fromkeys(QUESTIONS, 0.0)
    values.update(
        target_match=0.92,
        behavior_mechanism=0.9,
        visibility_gap=0.92,
        measurement_only=0.93,
        bounded_scope=0.93,
        overbroad=0.2,
        small_trial=0.9,
        checks_effect=0.9,
        guards_quality=0.85,
    )
    values.update(updates)
    return values


def _theme(index: int) -> RankedTheme:
    return RankedTheme(
        id=f"action-{index}",
        kind="action",
        title=f"Action {index}",
        status="investigate",
        affected_runs=index,
        incidents=index,
        evidence=f"pattern {index}",
        next_step="Record terminal state for linked child tasks.",
        verify="Compare known terminal states and task completion.",
        limit="The child may finish after capture.",
    )


class _Asker:
    model = "jev-1.13.0"

    def ask(self, state: dict, questions: dict) -> dict:
        index = int(state["observation"]["pattern"].split()[-1])
        features = _features() if index == 6 else _features(bounded_scope=0.4)
        return {name: {"noul": features[name]} for name in questions}


def test_atomic_gate_requires_each_boundary() -> None:
    assert decide(_features()) == "strong_visibility"
    assert decide(_features(measurement_only=0.1)) == "strong_behavior"
    assert decide(_features(measurement_only=0.1, behavior_mechanism=0.5)) == "review"
    assert decide(_features(overbroad=0.7)) == "review"
    assert decide(_features(bounded_scope=0.6)) == "review"
    with pytest.raises(ValueError, match="incomplete"):
        decide({"target_match": 0.99})


def test_projection_excludes_existing_status_and_rank() -> None:
    state = project_action_state(_theme(1))
    assert state["observation"]["pattern"] == "pattern 1"
    assert state["proposal"]["step"].startswith("Record terminal")
    assert "status" not in str(state)
    assert "score" not in str(state)


def test_review_context_cannot_replace_card_evidence() -> None:
    theme = _theme(1).model_copy(
        update={"review_context": ActionReviewContext(observation={"pattern": "different evidence"})}
    )
    with pytest.raises(ValueError, match="overrides canonical observation"):
        project_action_state(theme)


def test_review_rescues_only_one_omitted_action() -> None:
    data = HighlightsData(
        traces=6,
        total_cost=None,
        total_findings=0,
        detector_signals=0,
        themes=[_theme(i) for i in range(1, 7)],
    )
    reviewed, details = review_brief(data, cast("SystemOneAsker", _Asker()))
    assert details.rescued_ids == ["action-6"]
    assert reviewed.highlight_action_ids == ["action-1", "action-2", "action-3", "action-4", "action-6"]
    assert [theme.id for theme in reviewed.themes] == [theme.id for theme in data.themes]
    html = render_highlights(reviewed)
    assert "JeV discovery candidate" in html
    assert 'id="action-6"' in html
    assert 'id="action-5"' not in html


def test_discovery_does_not_displace_supported_card() -> None:
    themes = [_theme(i).model_copy(update={"status": "supported"}) for i in range(1, 6)] + [_theme(6)]
    data = HighlightsData(traces=6, total_cost=None, total_findings=0, detector_signals=0, themes=themes)
    reviewed, details = review_brief(data, cast("SystemOneAsker", _Asker()))
    assert details.rescued_ids == []
    assert reviewed.highlight_action_ids == [f"action-{i}" for i in range(1, 6)]
