"""Top five percent by observed cost basis, with ties and missingness explicit."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_hotwash.report.model import RunResult


class ExpensiveRun(BaseModel):
    trace_id: str
    cost: float
    basis: str
    cohort_n: int
    rank: int
    cutoff: float
    cost_share: float
    token_count: int | None = None
    outcome: str
    estimated_child_usage: bool = False
    workload: dict[str, Any] = Field(default_factory=dict)
    assessment: str = "not_reviewed"
    action: str = "Inspect scope, delivered outcome, verification, and repeated recovery."
    feature_set_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ExpenseTail(BaseModel):
    fraction: float = 0.05
    review_budget: int = 50
    reviewed_runs: int = 0
    missing_cost_runs: int = 0
    zero_cost_runs: int = 0
    duplicate_trace_ids: int = 0
    cohort_sizes: dict[str, int] = Field(default_factory=dict)
    selected_cost_share: dict[str, float] = Field(default_factory=dict)
    runs: list[ExpensiveRun] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=lambda: [
            "Ranked separately by provenance versus estimated cost; includes ties at the cutoff.",
            "This is a triage sample, not an estimate of waste or proof of cost optimality.",
            "For fewer than 20 priced runs, the top observation represents more than 5%.",
        ]
    )


def expensive_tail(runs: list[RunResult]) -> ExpenseTail:
    out = ExpenseTail()
    cohorts: defaultdict[str, list[RunResult]] = defaultdict(list)
    seen: set[str] = set()
    for run in runs:
        a = run.analysis
        if a.trace_id in seen:
            out.duplicate_trace_ids += 1
            continue
        seen.add(a.trace_id)
        if a.cost is None or not math.isfinite(a.cost) or a.cost < 0:
            out.missing_cost_runs += 1
        elif a.cost == 0:
            out.zero_cost_runs += 1
        else:
            cohorts[a.cost_source or "unknown"].append(run)
    for basis, cohort in sorted(cohorts.items()):
        cohort.sort(key=lambda r: (-(r.analysis.cost or 0), r.analysis.trace_id))
        n = len(cohort)
        out.cohort_sizes[basis] = n
        cutoff = cohort[math.ceil(n * out.fraction) - 1].analysis.cost or 0
        total = sum(r.analysis.cost or 0 for r in cohort)
        selected = [r for r in cohort if (r.analysis.cost or 0) >= cutoff]
        out.selected_cost_share[basis] = sum(r.analysis.cost or 0 for r in selected) / total
        for rank, run in enumerate(selected, 1):
            a = run.analysis
            out.runs.append(
                ExpensiveRun(
                    trace_id=a.trace_id,
                    cost=a.cost or 0,
                    basis=basis,
                    cohort_n=n,
                    rank=rank,
                    cutoff=cutoff,
                    cost_share=(a.cost or 0) / total,
                    token_count=a.total_tokens.total,
                    outcome=a.outcome.label,
                    workload={
                        "root_tokens": a.root.tokens.total,
                        "child_tokens": sum(s.tokens.total or 0 for s in a.subagents),
                        "subagent_count": a.subagent_count,
                        "root_user_turns": a.root.user_turns,
                        "root_tool_calls": a.root.tool_calls_total,
                        "root_failed_results": a.root.tool_error_count,
                        "root_edit_test_cycles": a.root.edit_test_cycles,
                        "root_cache_hit_ratio": a.root.cache_hit_ratio,
                        "tail_threshold_crossings": sum(r.exceeds_threshold for r in run.tails.incidents),
                    },
                    estimated_child_usage=any("usage_estimated" in h for h in a.degraded)
                    or any(h.usage_estimated for h in a.handovers),
                )
            )
    return out
