"""Cross-run aggregation.

``aggregate(analyses) -> Aggregate`` groups a batch of :class:`Analysis` objects
by agent kind, model and experiment and produces per-group summaries: success
rates (proxy and, where a harness ``resolved`` is present, ground truth), cost
per task, a tool-error leaderboard, a failure-mode histogram, p50/p95 trace
length and duration, and the most-thrashed files. Runs with no usable stream
(zero events) are reported as ``skipped`` counts, never dropped.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agent_hotwash.analytics import Analysis


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile (``pct`` in [0, 100]); ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _top(counter: Counter[str], n: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(n)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class GroupStats(BaseModel):
    """Summary metrics for one group of traces."""

    n: int = 0
    skipped: int = 0

    outcome_histogram: dict[str, int] = Field(default_factory=dict)
    success_rate: float | None = None  # positive / n
    ground_truth_success_rate: float | None = None  # resolved True / with-truth
    proxy_truth_agreement: float | None = None  # agree / with-truth

    total_cost: float | None = None
    cost_per_task: float | None = None

    tool_error_leaderboard: list[tuple[str, int]] = Field(default_factory=list)
    failure_histogram: dict[str, int] = Field(default_factory=dict)
    most_error_prone_tools: list[tuple[str, int]] = Field(default_factory=list)
    most_thrashed_files: list[tuple[str, int]] = Field(default_factory=list)

    p50_trace_length: float | None = None
    p95_trace_length: float | None = None
    p50_duration_seconds: float | None = None
    p95_duration_seconds: float | None = None


class Aggregate(BaseModel):
    """Cross-run rollup: overall plus grouped-by-agent/model/experiment stats."""

    total_traces: int = 0
    skipped: int = 0
    overall: GroupStats = Field(default_factory=GroupStats)
    by_agent: dict[str, GroupStats] = Field(default_factory=dict)
    by_model: dict[str, GroupStats] = Field(default_factory=dict)
    by_experiment: dict[str, GroupStats] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _group_stats(analyses: Sequence[Analysis]) -> GroupStats:
    g = GroupStats(n=len(analyses))

    tool_errors: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    thrashed: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()

    costs: list[float] = []
    lengths: list[float] = []
    durations: list[float] = []

    positives = 0
    with_truth = 0
    truth_positive = 0
    agree = 0

    for a in analyses:
        if a.root.event_count == 0:
            g.skipped += 1

        outcomes[a.outcome.label] += 1
        if a.outcome.label == "positive":
            positives += 1

        gt = a.outcome.ground_truth_resolved
        if gt is None:
            gt = a.resolved
        if gt is not None:
            with_truth += 1
            if gt:
                truth_positive += 1
            if (a.outcome.label == "positive") == bool(gt):
                agree += 1

        if a.cost is not None:
            costs.append(a.cost)

        # counts across root + subagents
        for m in [a.root, *a.subagents]:
            for tool, c in m.errors_by_tool.items():
                tool_errors[tool] += c
            for cat, c in m.error_categories.items():
                failures[cat] += c
            for path, c in m.files_by_edits.items():
                thrashed[path] += c

        lengths.append(float(a.root.event_count))
        if a.root.duration_seconds is not None:
            durations.append(a.root.duration_seconds)

    g.outcome_histogram = dict(outcomes)
    g.success_rate = positives / g.n if g.n else None
    if with_truth:
        g.ground_truth_success_rate = truth_positive / with_truth
        g.proxy_truth_agreement = agree / with_truth

    g.total_cost = sum(costs) if costs else None
    g.cost_per_task = _mean(costs)

    g.tool_error_leaderboard = _top(tool_errors)
    g.most_error_prone_tools = _top(tool_errors)
    g.failure_histogram = dict(failures)
    g.most_thrashed_files = _top(thrashed)

    g.p50_trace_length = _percentile(lengths, 50)
    g.p95_trace_length = _percentile(lengths, 95)
    g.p50_duration_seconds = _percentile(durations, 50)
    g.p95_duration_seconds = _percentile(durations, 95)

    return g


def _grouped(analyses: Sequence[Analysis], key) -> dict[str, GroupStats]:
    buckets: dict[str, list[Analysis]] = {}
    for a in analyses:
        k = key(a)
        if k is None:
            continue
        buckets.setdefault(str(k), []).append(a)
    return {k: _group_stats(v) for k, v in buckets.items()}


def aggregate(analyses: Sequence[Analysis]) -> Aggregate:
    """Roll a batch of analyses into overall + grouped summaries."""
    analyses = list(analyses)
    overall = _group_stats(analyses)
    return Aggregate(
        total_traces=len(analyses),
        skipped=overall.skipped,
        overall=overall,
        by_agent=_grouped(analyses, lambda a: a.agent.value),
        by_model=_grouped(analyses, lambda a: a.model),
        by_experiment=_grouped(analyses, lambda a: a.experiment),
    )


__all__ = ["Aggregate", "GroupStats", "aggregate"]
