"""Diagnostics engine: cost views + MECE waste + precedence (§7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_hotwash.config import Config
from agent_hotwash.diagnostics.continuation import continuation_diagnoses
from agent_hotwash.diagnostics.cost_views import (
    CostView,
    CostViews,
    Diagnosis,
    Money,
    iter_sessions,
    phase_spend_diagnoses,
    worse_status,
)
from agent_hotwash.diagnostics.precedence import apply_precedence
from agent_hotwash.diagnostics.waste import partition_waste
from agent_hotwash.events import PricingStatus, Trace
from agent_hotwash.semantic.results import FeatureSet
from agent_hotwash.structure.episodes import Episode
from agent_hotwash.structure.tasks import Task


def diagnose(
    trace: Trace,
    tasks: Sequence[Task],
    episodes: Sequence[Episode],
    features: Sequence[FeatureSet],
    existing_findings: Sequence[Any],
    config: Config,
) -> list[Diagnosis]:
    """Run cost views, the MECE waste partition, continuation, and precedence.

    Returns diagnoses including informational ``PHASE_SPEND``. Monetary waste
    rules abstain unless ``pricing_status`` is ``exact``.
    """
    task_list = list(tasks)
    episode_list = list(episodes)
    feat_list = list(features)
    findings = list(existing_findings)
    semantic_on = bool(feat_list) or config.semantic.mode != "off"

    out: list[Diagnosis] = []
    sessions = iter_sessions(trace)
    for session in sessions:
        out.extend(phase_spend_diagnoses(session, config, episode_list))
    out.extend(partition_waste(trace.root, task_list, episode_list, feat_list, findings, config))
    for session in sessions:
        sess_tasks = [t for t in task_list if t.session_id == session.session_id]
        if not sess_tasks and session is trace.root:
            sess_tasks = task_list
        if sess_tasks:
            out.extend(continuation_diagnoses(session, sess_tasks, feat_list, config))

    ranked = apply_precedence(
        out,
        tasks=task_list,
        episodes=episode_list,
        features=feat_list,
        existing_findings=findings,
        config=config,
        semantic_on=semantic_on,
    )
    return ranked


def attach_counterfactual(views: CostViews, diagnoses: Sequence[Diagnosis]) -> CostViews:
    """Fill ``views.counterfactual`` from CONTINUATION_BURDEN ranges, if any."""
    lows: list[float] = []
    highs: list[float] = []
    statuses: list[PricingStatus] = []
    for d in diagnoses:
        if d.id != "CONTINUATION_BURDEN" or d.amount is None:
            continue
        statuses.append(d.amount.pricing_status)
        if d.amount.amount_low is not None:
            lows.append(d.amount.amount_low)
        if d.amount.amount_high is not None:
            highs.append(d.amount.amount_high)
        elif d.amount.amount is not None:
            lows.append(d.amount.amount)
            highs.append(d.amount.amount)
    if lows or highs:
        views.counterfactual = Money(
            amount=(min(lows) + max(highs)) / 2.0 if lows and highs else None,
            view=CostView.counterfactual,
            pricing_status=worse_status(*statuses) if statuses else PricingStatus.unknown,
            amount_low=min(lows) if lows else None,
            amount_high=max(highs) if highs else None,
            assumptions=["union of CONTINUATION_BURDEN ranges"],
        )
    views.diagnoses = list(diagnoses)
    return views


__all__ = ["Diagnosis", "attach_counterfactual", "diagnose"]
