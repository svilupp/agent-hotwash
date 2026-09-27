"""Compact invoice overview from per-response charges.

Amounts are observed invoice estimates, not error attribution or savings. A
response with no ID has no safe cross-run identity and is counted each time it
appears; only (thread_id, response_id) pairs can be deduplicated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agent_hotwash.diagnostics.cost_views import CostView, Money, ResponseCharge, worse_status
from agent_hotwash.events import PricingStatus
from agent_hotwash.semantic.redact import redact_state

if TYPE_CHECKING:
    from agent_hotwash.config import Config
    from agent_hotwash.report.model import RunResult


class SpendRow(BaseModel):
    """A model or task invoice subtotal; missing prices remain visible in coverage."""

    name: str
    invoice: Money
    calls: int = 0
    exact_calls: int = 0
    estimated_calls: int = 0
    unknown_calls: int = 0


class TaskSpend(SpendRow):
    task_id: str
    trace_ids: list[str] = Field(default_factory=list)
    label: str = "Task text unavailable"
    task_type: str = "unknown"
    by_model: list[SpendRow] = Field(default_factory=list)


class SpendOverview(BaseModel):
    """Portable state for an overview without full run payloads."""

    invoice: Money
    calls: int = 0
    exact_calls: int = 0
    estimated_calls: int = 0
    unknown_calls: int = 0
    by_model: list[SpendRow] = Field(default_factory=list)
    top_tasks: list[TaskSpend] = Field(default_factory=list)
    total_tasks: int = 0
    cache_write_tokens: int = 0
    cache_write_charge: Money | None = None
    anonymous_calls: int = 0


class _Bucket:
    def __init__(self) -> None:
        self.amount = 0.0
        self.calls = 0
        self.statuses: dict[PricingStatus, int] = defaultdict(int)

    def add(self, charge: ResponseCharge) -> None:
        self.calls += 1
        self.statuses[charge.invoice.pricing_status] += 1
        if charge.invoice.amount is not None:
            self.amount += charge.invoice.amount

    def money(self) -> Money:
        status = worse_status(*self.statuses) if self.calls else PricingStatus.unknown
        return Money(
            amount=self.amount if self.calls and self.statuses[PricingStatus.unknown] < self.calls else None,
            view=CostView.invoice,
            pricing_status=status,
        )

    def row(self, name: str) -> SpendRow:
        return SpendRow(
            name=name,
            invoice=self.money(),
            calls=self.calls,
            exact_calls=self.statuses[PricingStatus.exact],
            estimated_calls=self.statuses[PricingStatus.estimated],
            unknown_calls=self.statuses[PricingStatus.unknown],
        )


def _task_label(
    run: RunResult | Mapping[str, Any], task_id: str, *, limit: int = 100, secrets: list[str] | None = None
) -> str:
    if isinstance(run, Mapping):
        structure = run.get("structure")
        candidate_tasks = structure.get("tasks") if isinstance(structure, Mapping) else None
        tasks = candidate_tasks if isinstance(candidate_tasks, list) else []
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            if task.get("task_id") != task_id:
                continue
            turns = task.get("turns")
            for turn in turns if isinstance(turns, list) else []:
                if not isinstance(turn, Mapping):
                    continue
                user_input = turn.get("user_input")
                text = user_input.get("text") if isinstance(user_input, Mapping) else None
                raw = " ".join(text.split()) if isinstance(text, str) else ""
                if raw:
                    safe = str(redact_state(raw[:2000], secrets or []))
                    return safe if len(safe) <= limit else safe[: limit - 1].rstrip() + "…"
        return "Task text unavailable"
    if run.structure:
        for task in run.structure.tasks:
            if task.task_id != task_id:
                continue
            for turn in task.turns:
                raw = " ".join(turn.user_input.text.split())
                if raw:
                    safe = str(redact_state(raw[:2000], secrets or []))
                    return safe if len(safe) <= limit else safe[: limit - 1].rstrip() + "…"
    return "Task text unavailable"


class SpendOverviewBuilder:
    """Incremental builder: add one RunResult at a time for large saved reports."""

    def __init__(self, *, pricing: Config | None = None, top_tasks: int = 10) -> None:
        """Use the report's original pricing config for cache-write dollars.

        Price lookup supplies dated/exact or fallback/estimated status; callers
        should omit ``pricing`` for saved reports when that config is unknown.
        """
        self.pricing = pricing
        self.top_tasks = max(0, top_tasks)
        self.all_spend = _Bucket()
        self.models: dict[str, _Bucket] = defaultdict(_Bucket)
        self.tasks: dict[str, _Bucket] = defaultdict(_Bucket)
        self.task_models: dict[str, dict[str, _Bucket]] = defaultdict(lambda: defaultdict(_Bucket))
        self.task_traces: dict[str, set[str]] = defaultdict(set)
        self.task_labels: dict[str, str] = {}
        self.seen: set[tuple[str, str]] = set()
        self.anonymous_calls = 0
        self.cache_write_tokens = 0
        self.cache_write_amount = 0.0
        self.cache_write_statuses: list[PricingStatus] = []
        self.cache_write_unpriced = False

    def add_run(self, run: RunResult | Mapping[str, Any]) -> None:
        """Accept a model or one decoded ``runs.item`` JSON object from ijson."""
        if isinstance(run, Mapping):
            raw_views = run.get("cost_views")
            charges = raw_views.get("per_response") if isinstance(raw_views, Mapping) else None
            raw_charges = charges if isinstance(charges, list) else []
            analysis = run.get("analysis")
            raw_trace_id = analysis.get("trace_id") if isinstance(analysis, Mapping) else None
            trace_id = raw_trace_id if isinstance(raw_trace_id, str) else "unknown trace"
        elif run.cost_views is not None:
            raw_charges = run.cost_views.per_response
            trace_id = run.analysis.trace_id
        else:
            return
        secrets = list(self.pricing.lexicons.secret) if self.pricing else []
        for raw in raw_charges:
            if isinstance(raw, ResponseCharge):
                charge = raw
            elif isinstance(raw, Mapping):
                charge = ResponseCharge.model_validate(raw)
            else:
                continue
            if charge.response_id:
                key = (charge.thread_id, charge.response_id)
                if key in self.seen:
                    continue
                self.seen.add(key)
            else:
                self.anonymous_calls += 1
            model = charge.model or "Unknown model"
            task_id = charge.root_task_id or charge.task_id or f"{charge.thread_id}:unknown-task"
            self.all_spend.add(charge)
            self.models[model].add(charge)
            self.tasks[task_id].add(charge)
            self.task_models[task_id][model].add(charge)
            self.task_traces[task_id].add(trace_id)
            if self.task_labels.get(task_id, "Task text unavailable") == "Task text unavailable":
                self.task_labels[task_id] = _task_label(run, task_id, secrets=secrets)
            write_tokens = charge.usage.cache_write or 0
            self.cache_write_tokens += write_tokens
            if write_tokens and self.pricing:
                price, lookup_status = self.pricing.price_lookup(charge.model)
                if price is None:
                    self.cache_write_unpriced = True
                else:
                    self.cache_write_amount += write_tokens * price.cache_write / 1_000_000
                    self.cache_write_statuses.append(worse_status(charge.invoice.pricing_status, lookup_status))

    def build(self) -> SpendOverview:
        by_model = [bucket.row(name) for name, bucket in self.models.items()]
        by_model.sort(key=lambda row: (-(row.invoice.amount or 0), row.name))
        tasks = []
        for task_id, bucket in self.tasks.items():
            model_rows = [part.row(name) for name, part in self.task_models[task_id].items()]
            model_rows.sort(key=lambda row: (-(row.invoice.amount or 0), row.name))
            tasks.append(
                TaskSpend(
                    **bucket.row(task_id).model_dump(),
                    task_id=task_id,
                    trace_ids=sorted(self.task_traces[task_id])[:5],
                    label=self.task_labels[task_id],
                    by_model=model_rows,
                )
            )
        tasks.sort(key=lambda row: (-(row.invoice.amount or 0), row.task_id))
        cache_charge = None
        if self.pricing and self.cache_write_tokens:
            status = worse_status(*self.cache_write_statuses)
            if self.cache_write_unpriced:
                status = PricingStatus.unknown
            cache_charge = Money(amount=self.cache_write_amount, view=CostView.invoice, pricing_status=status)
        return SpendOverview(
            invoice=self.all_spend.money(),
            calls=self.all_spend.calls,
            exact_calls=self.all_spend.statuses[PricingStatus.exact],
            estimated_calls=self.all_spend.statuses[PricingStatus.estimated],
            unknown_calls=self.all_spend.statuses[PricingStatus.unknown],
            by_model=by_model,
            top_tasks=tasks[: self.top_tasks],
            total_tasks=len(tasks),
            cache_write_tokens=self.cache_write_tokens,
            cache_write_charge=cache_charge,
            anonymous_calls=self.anonymous_calls,
        )


def build_spend_overview(
    runs: Iterable[RunResult | Mapping[str, Any]], *, pricing: Config | None = None, top_tasks: int = 10
) -> SpendOverview:
    builder = SpendOverviewBuilder(pricing=pricing, top_tasks=top_tasks)
    for run in runs:
        builder.add_run(run)
    return builder.build()


__all__ = ["SpendOverview", "SpendOverviewBuilder", "SpendRow", "TaskSpend", "build_spend_overview"]
