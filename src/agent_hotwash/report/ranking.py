"""Presentation-independent inventory and provisional ranking of report themes.

Counts and charges are measured in code. Semantic answers may be supplied by a
separate review pass; rendering never calls a model. Scores prioritize reading,
not expected savings. Action and detector scores have different rubrics and
must only be compared within their own lanes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from agent_hotwash.report.priorities import ActionReviewContext

if TYPE_CHECKING:
    from agent_hotwash.report.model import Report


class Judgment(BaseModel):
    actionability: float | None = Field(default=None, ge=0, le=1)
    ease: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str = "unreviewed"


class RankedTheme(BaseModel):
    id: str
    kind: Literal["action", "detector"]
    title: str
    status: str
    affected_runs: int
    incidents: int
    observed_cost: float | None = None
    cost_basis: str | None = None
    severity: str | None = None
    evidence: str
    next_step: str
    verify: str
    limit: str
    example_trace_ids: list[str] = Field(default_factory=list)
    member_signals: dict[str, int] = Field(default_factory=dict)
    review_context: ActionReviewContext | None = None
    score: float = 0
    score_basis: str = "unscored"
    components: dict[str, float] = Field(default_factory=dict)
    judgment: Judgment = Field(default_factory=Judgment)


def score_theme(
    theme: RankedTheme,
    total_runs: int,
    *,
    judgment: Judgment | None = None,
    cost_reference: float | None = None,
) -> RankedTheme:
    """Apply a transparent within-lane attention rubric, never a savings estimate.

    ``cost_reference`` is the largest observed action charge in this report.
    With no cohort reference, charge is retained but does not affect the score.
    The component weights are editorial and have no fitted outcome calibration.
    """
    reviewed = judgment or theme.judgment
    volume = min(1.0, max(theme.affected_runs, 0) / max(total_runs, 1))
    if theme.kind == "detector":
        severity = {"high": 1.0, "medium": 0.67, "low": 0.33, "info": 0.0}.get(theme.severity or "", 0.0)
        components = {"affected_runs": round(60 * volume, 2), "max_severity": round(40 * severity, 2)}
        return theme.model_copy(
            update={
                "score": round(sum(components.values()), 2),
                "score_basis": "detector_attention_v2",
                "components": components,
                "judgment": reviewed,
            }
        )
    spend = (
        min(1.0, max(theme.observed_cost, 0) / cost_reference)
        if theme.observed_cost is not None and cost_reference is not None and cost_reference > 0
        else 0.0
    )
    default_action = {"supported": 0.85, "investigate": 0.4, "measure": 0.25, "signal": 0.15}.get(theme.status, 0.15)
    default_ease = 0.7 if theme.status == "supported" else 0.25
    # Confidence is a shrinkage weight toward the status baseline, not an
    # estimated probability of correctness or a penalty on missing evidence.
    weight = reviewed.confidence if reviewed.confidence is not None else 1.0
    action = (
        default_action + weight * ((reviewed.actionability or 0) - default_action)
        if reviewed.actionability is not None
        else default_action
    )
    ease = default_ease + weight * ((reviewed.ease or 0) - default_ease) if reviewed.ease is not None else default_ease
    components = {
        "affected_runs": round(30 * volume, 2),
        "observed_spend": round(15 * spend, 2),
        "actionability": round(35 * action, 2),
        "ease": round(20 * ease, 2),
    }
    return theme.model_copy(
        update={
            "score": round(sum(components.values()), 2),
            "score_basis": "action_attention_v2",
            "components": components,
            "judgment": reviewed,
        }
    )


# Families describe shared investigation questions, not proven shared causes.
_SIGNAL_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "post_edit_verification": (
        "Verify code after the final edit",
        ("LOOKS_RIGHT_RUNS_WRONG", "UNVERIFIED_COMPLETION"),
    ),
    "recovery_loops": (
        "Inspect repeated recovery and editing loops",
        ("EDIT_THRASH", "NO_ADAPT_RETRY", "RETRY_STORM"),
    ),
    "linear_search": (
        "Inspect repeated linear searches",
        ("LINEAR_SCAN", "linear_scan_search"),
    ),
}

_ACTION_SIGNALS = {
    "TAIL_TOOL_LATENCY": "investigate-tool_latency",
    "TAIL_CACHE_REBUILDS": "investigate-cache_rebuilds",
    "TAIL_PARENT_WAIT": "investigate-parent_wait",
    "TAIL_RETRY_ATTEMPTS": "investigate-retry_attempts",
    "TAIL_DELEGATION_OPEN": "measure-delegation_open",
    "TAIL_STATUS_PROBES": "supported-status_probes",
}


def build_themes(report: Report, *, judgments: dict[str, Judgment] | None = None) -> list[RankedTheme]:
    """Group observations before ranking; no per-incident semantic questions."""
    judgments = judgments or {}
    themes: list[RankedTheme] = []
    actions = report.priorities
    action_ids = {action.id for action in actions}
    for action in actions:
        themes.append(
            RankedTheme(
                id=action.id,
                kind="action",
                title=action.title,
                status=action.status,
                affected_runs=action.affected_runs,
                incidents=action.incidents,
                observed_cost=action.observed_cost,
                cost_basis=action.cost_basis,
                evidence=action.evidence,
                next_step=action.next_step,
                verify=action.verify,
                limit=action.limit,
                example_trace_ids=[example.trace_id for example in action.examples],
                review_context=getattr(action, "review_context", None),
            )
        )
    by_id: dict[str, set[str]] = defaultdict(set)
    sample: dict[str, str] = {}
    for run in report.runs:
        for finding in run.findings:
            by_id[finding.id].add(run.analysis.trace_id)
            sample.setdefault(finding.id, finding.message)
    eligible = {
        detector_id: count
        for detector_id, count in report.finding_histogram.items()
        if _ACTION_SIGNALS.get(detector_id) not in action_ids
    }
    family_ids = {detector_id for _, members in _SIGNAL_FAMILIES.values() for detector_id in members}
    groups: list[tuple[str, str, dict[str, int]]] = []
    for family, (title, members) in _SIGNAL_FAMILIES.items():
        present = {detector_id: eligible[detector_id] for detector_id in members if detector_id in eligible}
        if present:
            groups.append((f"detector-family-{family}", title, present))
    groups.extend(
        (f"detector-{detector_id}", detector_id.replace("_", " ").capitalize(), {detector_id: count})
        for detector_id, count in eligible.items()
        if detector_id not in family_ids
    )
    for group_id, title, members in groups:
        trace_ids: set[str] = set()
        for detector_id in members:
            trace_ids.update(by_id[detector_id])
        severity = next(
            (
                s
                for s in ("high", "medium", "low", "info")
                if any(report.finding_severity.get(detector_id, {}).get(s) for detector_id in members)
            ),
            None,
        )
        occurrences = sum(members.values())
        evidence = (
            "; ".join(f"{detector_id}: {count:,}" for detector_id, count in members.items())
            + " detector occurrences; signals may describe the same event."
            if len(members) > 1
            else sample.get(next(iter(members))) or f"{occurrences:,} recorded occurrences."
        )
        themes.append(
            RankedTheme(
                id=group_id,
                kind="detector",
                title=title,
                status="signal",
                affected_runs=len(trace_ids),
                incidents=occurrences,
                severity=severity,
                evidence=evidence,
                next_step="Inspect representative traces and decide whether one shared cause or intervention exists.",
                verify="Recount the signal on comparable work and check accepted task outcomes.",
                limit="Detector occurrences can be false positives or overlap other themes; no savings are inferred.",
                example_trace_ids=sorted(trace_ids)[:3],
                member_signals=members,
            )
        )
    cost_reference = max(
        (t.observed_cost for t in themes if t.kind == "action" and t.observed_cost is not None), default=None
    )
    return sorted(
        (score_theme(t, len(report.runs), judgment=judgments.get(t.id), cost_reference=cost_reference) for t in themes),
        key=lambda t: (
            0 if t.kind == "action" else 1,
            {"supported": 0, "investigate": 1, "measure": 2, "signal": 3}.get(t.status, 4),
            -t.score,
            t.id,
        ),
    )


__all__ = ["Judgment", "RankedTheme", "build_themes", "score_theme"]
