"""Conservative JeV review of proposed actions, separate from report generation.

The questions judge one candidate at a time. Code owns counts, thresholds and
highlight selection. A Noul value is support for *yes*, not calibrated
confidence; the wide margins below are provisional safety defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from agent_hotwash.report.priorities import ACTION_REVIEW_CONTEXT, ActionReviewContext
from agent_hotwash.report.ranking import RankedTheme

if TYPE_CHECKING:
    from agent_hotwash.report.highlights import HighlightsData
    from agent_hotwash.semantic.client import SystemOneAsker


class ActionReview(BaseModel):
    model: str
    features: dict[str, float]
    checks: dict[str, bool] = Field(default_factory=dict)
    decision: Literal["strong_behavior", "strong_visibility", "review"] = "review"
    rule: str = "jev_action_v3_provisional"


class ReviewResult(BaseModel):
    themes: dict[str, ActionReview] = Field(default_factory=dict)
    rescued_ids: list[str] = Field(default_factory=list)


def project_action_state(theme: RankedTheme) -> dict[str, Any]:
    """Project one theme; do not send existing status, rank, title or verdicts."""
    limit = theme.limit.replace("Supported by bounded trace evidence and JeV features. ", "")
    state = {
        "observation": {"pattern": theme.evidence},
        "proposal": {"step": theme.next_step},
        "verification": {"plan": theme.verify},
        "uncertainty": {"known_limit": limit},
    }
    context = theme.review_context or ACTION_REVIEW_CONTEXT.get(theme.id.partition("-")[2])
    return _add_context(state, context)


def _add_context(state: dict[str, Any], context: ActionReviewContext | None) -> dict[str, Any]:
    for section, fields in (context.model_dump() if context else {}).items():
        collisions = set(state.get(section, {})) & set(fields)
        if collisions:
            raise ValueError(f"review context overrides canonical {section} fields: {sorted(collisions)}")
        state.setdefault(section, {}).update(fields)
    return state


def project_case_state(state: dict[str, str], context: ActionReviewContext | None = None) -> dict[str, Any]:
    """Apply the same text projection to saved blinded evaluation cases."""
    return _add_context(
        {
            "observation": {"pattern": state["evidence"]},
            "proposal": {"step": state["next_step"]},
            "verification": {"plan": state["verify"]},
            "uncertainty": {"known_limit": state["limit"]},
        },
        context,
    )


def _noul(question: str, inspect: list[str], yes: str, no: str) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": {"question": question, "inspect": inspect},
        "criteria": {"true": {"what": yes}, "false": {"what": no}},
    }


QUESTIONS: dict[str, dict[str, Any]] = {
    "target_match": _noul(
        "Does `proposal.step` target the specific pattern in `observation.pattern`?",
        ["proposal", "observation"],
        "The proposed action addresses the observed operation, failure, or missing record.",
        "The proposal concerns only the same general topic or a different operation.",
    ),
    "behavior_mechanism": _noul(
        "Does `observation.pattern` show a concrete mechanism by which the behavior change "
        "in `proposal.step` could improve the observed case?",
        ["observation", "proposal", "uncertainty"],
        "The observed trace behavior supports this particular change, "
        "even though a bounded trial must verify its benefit.",
        "Only a large count, cost, or duration is shown, or the proposed change needs a cause not observed here.",
    ),
    "visibility_gap": _noul(
        "Does `observation.pattern` show a missing or incomplete measurement "
        "that `proposal.step` would directly record?",
        ["observation", "proposal", "uncertainty"],
        "A relevant outcome, terminal state, scope fact, or repeated-work condition "
        "is not visible, and the step would record it.",
        "The needed fact is already observed, or the step does not record the missing fact.",
    ),
    "measurement_only": _noul(
        "Does `proposal.step` only add recording or redacted triage for the observed pattern, "
        "without suppressing work or imposing a new outcome?",
        ["proposal", "observation"],
        "The step records or routes evidence while leaving task behavior and accepted outcomes intact.",
        "The step blocks, truncates, terminates, disables, or changes behavior; it is more than measurement or triage.",
    ),
    "bounded_scope": _noul(
        "Is `proposal.step` restricted to the observed operation or a matching condition, "
        "instead of every workflow regardless of evidence?",
        ["proposal", "observation"],
        "It names a matching trigger or target, or can be trialed on the observed operation alone.",
        "It applies a blanket rule to unrelated operations or workflows.",
    ),
    "overbroad": _noul(
        "Would `proposal.step` impose a rule on operations beyond what `observation.pattern` supports?",
        ["proposal", "observation", "uncertainty"],
        "The action extends to unrelated cases or removes a legitimate option without supporting evidence.",
        "The action is conditional on the observed case, or only adds relevant measurement.",
    ),
    "small_trial": _noul(
        "Can `proposal.step` be tried on a small matched set and stopped if `verification.plan` shows harm?",
        ["proposal", "verification"],
        "A bounded trial and reversal are practical for this proposal.",
        "The proposal is an irreversible, global, or untestable change.",
    ),
    "checks_effect": _noul(
        "Does `verification.plan` measure the intended effect of `proposal.step`?",
        ["proposal", "verification"],
        "The plan measures the burden or missing datum the step is intended to change.",
        "The plan does not measure whether the step achieved its stated purpose.",
    ),
    "guards_quality": _noul(
        "Does `verification` explicitly monitor task completion, correctness, useful output, "
        "or evidence completeness while `proposal.step` is tried?",
        ["proposal", "verification"],
        "The plan keeps a relevant task or evidence quality measure visible during the trial.",
        "The plan checks only reduced volume, cost, or duration without a quality guard.",
    ),
}


# These margins are intentionally conservative screening defaults. They were
# inspected on overlapping September examples and are not calibrated.
YES = 0.85
NO = 0.35


def gate_checks(features: dict[str, float]) -> dict[str, bool]:
    if any(not 0 <= value <= 1 for value in features.values()) or set(features) != set(QUESTIONS):
        raise ValueError("incomplete or invalid JeV action features")
    return {
        "target_match_ge_085": features["target_match"] >= YES,
        "bounded_scope_ge_085": features["bounded_scope"] >= YES,
        "overbroad_le_035": features["overbroad"] <= NO,
        "small_trial_ge_080": features["small_trial"] >= 0.8,
        "guards_quality_ge_070": features["guards_quality"] >= 0.7,
        "visibility_gap_ge_085": features["visibility_gap"] >= YES,
        "measurement_only_ge_085": features["measurement_only"] >= YES,
        "behavior_mechanism_ge_085": features["behavior_mechanism"] >= YES,
        "checks_effect_ge_080": features["checks_effect"] >= 0.8,
    }


def decide(features: dict[str, float]) -> Literal["strong_behavior", "strong_visibility", "review"]:
    checks = gate_checks(features)
    common = all(
        checks[name]
        for name in (
            "target_match_ge_085",
            "bounded_scope_ge_085",
            "overbroad_le_035",
            "small_trial_ge_080",
            "guards_quality_ge_070",
        )
    )
    if not common:
        return "review"
    if checks["visibility_gap_ge_085"] and checks["measurement_only_ge_085"]:
        return "strong_visibility"
    if checks["behavior_mechanism_ge_085"] and checks["checks_effect_ge_080"]:
        return "strong_behavior"
    return "review"


def review_strength(review: ActionReview) -> float:
    """The weakest required signal limits the discovery-slot ranking."""
    f = review.features
    common = [f["target_match"], f["bounded_scope"], 1 - f["overbroad"], f["small_trial"], f["guards_quality"]]
    if review.decision == "strong_visibility":
        return min(*common, f["visibility_gap"], f["measurement_only"])
    if review.decision == "strong_behavior":
        return min(*common, f["behavior_mechanism"], f["checks_effect"])
    return 0.0


def review_state(state: dict[str, Any], asker: SystemOneAsker) -> ActionReview:
    answers = asker.ask(state, QUESTIONS)
    features = {name: float(answers[name]["noul"]) for name in QUESTIONS}
    return ActionReview(model=asker.model, features=features, checks=gate_checks(features), decision=decide(features))


def review_brief(
    data: HighlightsData, asker: SystemOneAsker, *, card_limit: int = 5
) -> tuple[HighlightsData, ReviewResult]:
    """Rescue only a strong omitted action; leave the status-first base order."""
    actions = [theme for theme in data.themes if theme.kind == "action"]
    reviews = {theme.id: review_state(project_action_state(theme), asker) for theme in actions}
    selected = [theme.id for theme in actions[:card_limit]]
    omitted = [theme for theme in actions[card_limit:] if reviews[theme.id].decision != "review"]
    rescued: list[str] = []
    replace_index = next(
        (index for index in range(len(actions[:card_limit]) - 1, -1, -1) if actions[index].status != "supported"), None
    )
    if omitted and replace_index is not None:
        # Keep the status-first order in the inventory. A strong JeV judgment
        # changes only the compact card selection, replacing the last card.
        winner = max(
            omitted,
            key=lambda theme: (review_strength(reviews[theme.id]), theme.affected_runs, theme.incidents, theme.id),
        )
        selected[replace_index] = winner.id
        rescued.append(winner.id)
    updated = data.model_copy(update={"highlight_action_ids": selected, "review_rescued_ids": rescued})
    return updated, ReviewResult(themes=reviews, rescued_ids=rescued)


__all__ = [
    "QUESTIONS",
    "ActionReview",
    "ReviewResult",
    "decide",
    "gate_checks",
    "project_action_state",
    "project_case_state",
    "review_brief",
    "review_state",
    "review_strength",
]
