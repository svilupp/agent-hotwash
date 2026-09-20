"""Cross-run aggregation.

``aggregate(analyses) -> Aggregate`` groups a batch of :class:`Analysis` objects
by agent kind, model and experiment and produces per-group summaries: success
rates (proxy and, where a harness ``resolved`` is present, ground truth), cost
per task, a tool-error leaderboard, a failure-mode histogram, p50/p95 trace
length and duration, and the most-thrashed files. Runs with no usable stream
(zero events) are reported as ``skipped`` counts, never dropped.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from agent_hotwash.config import DiagnosticsConfig

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


# ---------------------------------------------------------------------------
# Monthly root-task rollup (WP10 / §7.7)
# ---------------------------------------------------------------------------

_SKIP_PREVALENCE = frozenset({"PHASE_SPEND", "CONTEXT_CARRYOVER"})


class DiagnosisPrevalence(BaseModel):
    """Three unmixed prevalence measures for one diagnosis id."""

    id: str
    task_count: int = 0
    invoice_dollars: float | None = None
    counterfactual_low: float | None = None
    counterfactual_high: float | None = None


class MonthlyCell(BaseModel):
    """One (month, model class, effort) stratum."""

    month: str
    model_class: str | None = None
    effort: str | None = None
    n_tasks: int = 0
    invoice_total: float | None = None
    pricing_status: str = "unknown"
    prevalence: list[DiagnosisPrevalence] = Field(default_factory=list)
    semantic_coverage: float = 0.0
    min_support: int = 0
    support: int = 0


class MonthlyRollup(BaseModel):
    """Root-task monthly aggregation with three-measure diagnosis ranking."""

    timezone: str = "UTC"
    cells: list[MonthlyCell] = Field(default_factory=list)
    prevalence: list[DiagnosisPrevalence] = Field(default_factory=list)
    ranked_by_task_count: list[str] = Field(default_factory=list)
    ranked_by_invoice: list[str] = Field(default_factory=list)
    ranked_by_counterfactual: list[str] = Field(default_factory=list)
    overspend_statement: str | None = None


def _as_datetime(ts: Any) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _month_of(ts: Any, timezone: str) -> str | None:
    dt = _as_datetime(ts)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local = dt.astimezone(ZoneInfo(timezone))
    return f"{local.year:04d}-{local.month:02d}"


def _analysis_of(item: Any) -> Any:
    return getattr(item, "analysis", item)


def _model_class(model: str | None) -> str | None:
    """Model identity with effort stripped (C11: keep class and effort separate)."""
    if not model:
        return None
    return model.split(":")[0]


def _item_min_support(item: Any) -> int | None:
    for val in (
        getattr(item, "min_support", None),
        getattr(getattr(item, "diagnostics", None), "min_support", None),
        getattr(getattr(getattr(item, "config", None), "diagnostics", None), "min_support", None),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        return int(val)
    return None


def _rank_ids(rows: list[DiagnosisPrevalence], measure: str) -> list[str]:
    scored: list[tuple[float, str]] = []
    for row in rows:
        if measure == "task_count":
            val = float(row.task_count)
        elif measure == "invoice":
            if row.invoice_dollars is None:
                continue
            val = float(row.invoice_dollars)
        else:
            if row.counterfactual_high is None and row.counterfactual_low is None:
                continue
            hi = row.counterfactual_high
            lo = row.counterfactual_low
            val = float(hi if hi is not None else lo or 0.0)
        scored.append((val, row.id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _v, i in scored]


def _accum_prev(
    store: dict[str, DiagnosisPrevalence],
    did: str,
    *,
    task_id: str | None,
    invoice: float | None,
    cf_low: float | None,
    cf_high: float | None,
    tasks_seen: dict[str, set[str]],
) -> None:
    row = store.setdefault(did, DiagnosisPrevalence(id=did))
    if task_id:
        seen = tasks_seen.setdefault(did, set())
        if task_id not in seen:
            seen.add(task_id)
            row.task_count = len(seen)
    if invoice is not None:
        row.invoice_dollars = (row.invoice_dollars or 0.0) + invoice
    if cf_low is not None:
        row.counterfactual_low = cf_low if row.counterfactual_low is None else min(row.counterfactual_low, cf_low)
    if cf_high is not None:
        row.counterfactual_high = cf_high if row.counterfactual_high is None else max(row.counterfactual_high, cf_high)


def monthly_rollup(analyses_or_runs: Sequence[Any], *, timezone: str = "UTC") -> MonthlyRollup:
    """Roll root tasks in thread trees by calendar month.

    Unit = root task; global ``(thread_id, response_id)`` dedup; child spend
    rolls into exactly one root; diagnosis prevalence is reported as task count,
    invoice dollars, and counterfactual range — three columns, never mixed.
    """
    seen_calls: set[tuple[str, str]] = set()
    # cell key -> {root_task_ids, invoice sum, statuses, coverage samples, diag store}
    cell_tasks: dict[tuple[str, str | None, str | None], set[str]] = defaultdict(set)
    cell_invoice: dict[tuple[str, str | None, str | None], float] = defaultdict(float)
    cell_status: dict[tuple[str, str | None, str | None], list[str]] = defaultdict(list)
    cell_cov: dict[tuple[str, str | None, str | None], list[float]] = defaultdict(list)
    cell_prev: dict[tuple[str, str | None, str | None], dict[str, DiagnosisPrevalence]] = defaultdict(dict)
    cell_prev_tasks: dict[tuple[str, str | None, str | None], dict[str, set[str]]] = defaultdict(dict)
    overall: dict[str, DiagnosisPrevalence] = {}
    overall_tasks: dict[str, set[str]] = {}
    min_support = 0
    task_to_cells: dict[str, set[tuple[str, str | None, str | None]]] = defaultdict(set)
    task_diags: dict[str, list[Any]] = defaultdict(list)

    for item in analyses_or_runs:
        analysis = _analysis_of(item)
        views = getattr(item, "cost_views", None)
        features = getattr(item, "features", None) or []
        structure = getattr(item, "structure", None)
        observed_ms = _item_min_support(item)
        if observed_ms is not None:
            min_support = max(min_support, observed_ms)

        charges = list(getattr(views, "per_response", None) or [])
        diagnoses = list(getattr(views, "diagnoses", None) or [])
        coverages = [float(getattr(fs, "coverage", 0.0) or 0.0) for fs in features]

        fallback_root = None
        if structure is not None:
            tasks = getattr(structure, "tasks", None) or []
            for task in tasks:
                if not getattr(task, "parent_task", None):
                    fallback_root = getattr(task, "task_id", None)
                    break
        if fallback_root is None:
            fallback_root = getattr(analysis, "trace_id", None) or "task0"

        for d in diagnoses:
            if d.id in _SKIP_PREVALENCE:
                continue
            tids = []
            if isinstance(d.evidence, dict) and d.evidence.get("task_id"):
                tids.append(str(d.evidence["task_id"]))
            for s in d.spans or []:
                tids.append(str(s))
            if not tids:
                tids = [str(fallback_root)]
            for tid in tids:
                task_diags[tid].append(d)

        if not charges:
            # Analysis-only: one synthetic charge from headline cost.
            month = "unknown"
            model = _model_class(getattr(analysis, "model", None))
            key = (month, model, None)
            cell_tasks[key].add(str(fallback_root))
            if getattr(analysis, "cost", None) is not None:
                cell_invoice[key] += float(analysis.cost)
            cell_status[key].append("estimated" if getattr(analysis, "cost", None) is not None else "unknown")
            if coverages:
                cell_cov[key].extend(coverages)
            continue

        for charge in charges:
            thread_id = getattr(charge, "thread_id", "") or ""
            rid = getattr(charge, "response_id", None) or ""
            dedup_key = (str(thread_id), str(rid))
            if rid and dedup_key in seen_calls:
                continue
            if rid:
                seen_calls.add(dedup_key)
            month = _month_of(getattr(charge, "ts_start", None), timezone) or "unknown"
            model = _model_class(getattr(charge, "model", None))
            effort = getattr(charge, "effort", None)
            root_tid = getattr(charge, "root_task_id", None) or fallback_root
            key = (month, model, effort)
            cell_tasks[key].add(str(root_tid))
            task_to_cells[str(root_tid)].add(key)
            inv = getattr(charge, "invoice", None)
            amt = getattr(inv, "amount", None) if inv is not None else None
            if amt is not None:
                cell_invoice[key] += float(amt)
            status = getattr(inv, "pricing_status", None) if inv is not None else None
            cell_status[key].append(status.value if hasattr(status, "value") else str(status or "unknown"))
            if coverages:
                cell_cov[key].extend(coverages)

    # Attribute diagnoses to the cells their root task spent in.
    for task_id, diags in task_diags.items():
        cells = task_to_cells.get(task_id) or {("unknown", None, None)}
        for d in diags:
            inv_amt = None
            cf_lo = cf_hi = None
            amount = getattr(d, "amount", None)
            view = getattr(amount, "view", None) if amount is not None else None
            view_s = view.value if hasattr(view, "value") else str(view or "")
            if amount is not None and view_s == "invoice" and amount.amount is not None:
                inv_amt = float(amount.amount)
            if amount is not None and view_s == "counterfactual":
                cf_lo = amount.amount_low
                cf_hi = amount.amount_high
                if cf_lo is None and amount.amount is not None:
                    cf_lo = float(amount.amount)
                if cf_hi is None and amount.amount is not None:
                    cf_hi = float(amount.amount)
            _accum_prev(
                overall,
                d.id,
                task_id=task_id,
                invoice=inv_amt,
                cf_low=cf_lo,
                cf_high=cf_hi,
                tasks_seen=overall_tasks,
            )
            for key in cells:
                _accum_prev(
                    cell_prev[key],
                    d.id,
                    task_id=task_id,
                    invoice=inv_amt,
                    cf_low=cf_lo,
                    cf_high=cf_hi,
                    tasks_seen=cell_prev_tasks[key],
                )

    status_rank = {"exact": 0, "estimated": 1, "unknown": 2}
    cells: list[MonthlyCell] = []
    for key in sorted(cell_tasks, key=lambda k: (k[0], str(k[1]), str(k[2]))):
        month, model, effort = key
        statuses = cell_status.get(key) or ["unknown"]
        worst = max(statuses, key=lambda s: status_rank.get(s, 2))
        covs = cell_cov.get(key) or []
        n = len(cell_tasks[key])
        cells.append(
            MonthlyCell(
                month=month,
                model_class=model,
                effort=effort,
                n_tasks=n,
                invoice_total=cell_invoice.get(key),
                pricing_status=worst,
                prevalence=sorted(cell_prev.get(key, {}).values(), key=lambda r: r.id),
                semantic_coverage=(sum(covs) / len(covs)) if covs else 0.0,
                min_support=min_support or DiagnosticsConfig().min_support,
                support=n,
            )
        )

    prev_list = sorted(overall.values(), key=lambda r: r.id)
    by_tasks = _rank_ids(prev_list, "task_count")
    by_inv = _rank_ids(prev_list, "invoice")
    by_cf = _rank_ids(prev_list, "counterfactual")
    statement = None
    if by_inv:
        statement = f"you mostly overspend on {by_inv[0]} (ranked by invoice dollars)"
    elif by_cf:
        statement = f"you mostly overspend on {by_cf[0]} (ranked by counterfactual range)"
    elif by_tasks:
        statement = f"you mostly overspend on {by_tasks[0]} (ranked by task count)"

    return MonthlyRollup(
        timezone=timezone,
        cells=cells,
        prevalence=prev_list,
        ranked_by_task_count=by_tasks,
        ranked_by_invoice=by_inv,
        ranked_by_counterfactual=by_cf,
        overspend_statement=statement,
    )


__all__ = [
    "Aggregate",
    "DiagnosisPrevalence",
    "GroupStats",
    "MonthlyCell",
    "MonthlyRollup",
    "aggregate",
    "monthly_rollup",
]
