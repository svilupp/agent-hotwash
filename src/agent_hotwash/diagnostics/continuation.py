"""CONTINUATION_BURDEN proxy and informational CONTEXT_CARRYOVER (§7.4)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent_hotwash.config import Config
from agent_hotwash.diagnostics.cost_views import (
    CostView,
    Diagnosis,
    Money,
    input_cost_of,
    iter_billed_calls,
    monetary_enabled,
    mtok,
)
from agent_hotwash.events import EventKind, PricingStatus, Session, Usage
from agent_hotwash.structure.tasks import Task

_CACHE_MISS_RATIO = 0.10  # cache_read / (input+cache_read) below this is a miss


def _feature_index(features: list[Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fs in features or []:
        oid = getattr(fs, "object_id", None)
        if oid:
            out[str(oid)] = fs
    return out


def _jev_abstain(fs: Any, feature_id: str) -> bool:
    if fs is None:
        return False
    values = getattr(fs, "values", None) or {}
    fv = values.get(feature_id)
    if fv is None:
        return False
    if getattr(fv, "source", "jev") != "jev":
        return False
    if getattr(fv, "reason", None) == "low_support":
        return True
    conf = getattr(fv, "confidence", None)
    return conf is not None and 0.3 <= float(conf) <= 0.7


def _is_cache_miss(usage: Usage) -> bool:
    cached = usage.cache_read or 0
    uncached = usage.input or 0
    total = cached + uncached
    return total > 0 and (cached / total) < _CACHE_MISS_RATIO


def _occupancy(usage: Usage) -> int:
    return (usage.input or 0) + (usage.cache_read or 0)


def _compaction_in_turn(session: Session, turn_id: str | None) -> bool:
    if not turn_id:
        return any(ev.kind is EventKind.compaction for ev in session.events)
    for ev in session.events:
        if ev.kind is EventKind.compaction and ev.turn_id == turn_id:
            return True
    turn = next((t for t in session.turns if t.turn_id == turn_id), None)
    if turn is not None:
        return turn.compactions > 0
    return False


def _gap_before(session: Session, ts: datetime | None, gap_minutes: float) -> bool:
    if ts is None:
        return False
    prior = [ev.ts for ev in session.events if ev.ts is not None and ev.ts < ts and ev.kind is not EventKind.meta]
    if not prior:
        return False
    delta = (ts - max(prior)).total_seconds() / 60.0
    return delta > gap_minutes


def _artifact_reuse(task: Task, prior_tasks: list[Task]) -> int:
    current = set(task.ledger.artifacts) if task.ledger else set()
    if not current:
        return 0
    prior: set[str] = set()
    for other in prior_tasks:
        if other.ledger:
            prior |= set(other.ledger.artifacts)
    return len(current & prior)


def _fresh_baseline_range(
    turn_text: str,
    occupancy: int,
    input_price: float,
) -> tuple[float, float, list[str]]:
    request_tokens = max(1, len(turn_text) // 4) if turn_text else 1
    # System prompt size is not observed; bound it.
    system_low = 0
    system_high = max(0, min(occupancy, max(occupancy - request_tokens, occupancy // 5)))
    assumptions = [
        f"request_tokens≈{request_tokens} (chars/4 of user input)",
        f"system_prompt_tokens in [{system_low}, {system_high}] (unobserved; occupancy={occupancy})",
        "fresh session billed at uncached input price; no cache write",
    ]
    low = mtok(request_tokens + system_low) * input_price
    high = mtok(request_tokens + system_high) * input_price
    if high < low:
        low, high = high, low
    return low, high, assumptions


def continuation_diagnoses(
    session: Session,
    tasks: list[Task],
    features: list[Any] | None,
    config: Config,
) -> list[Diagnosis]:
    """Emit CONTINUATION_BURDEN (counterfactual range) or CONTEXT_CARRYOVER."""
    idx = _feature_index(features)
    gap = config.analytics.idle_gap_minutes
    pressure = config.diagnostics.context_pressure_pct
    window = config.analytics.context_window_tokens
    out: list[Diagnosis] = []

    for i, task in enumerate(tasks):
        edge = task.edge_to_prev
        reuse = _artifact_reuse(task, tasks[:i])
        fs = idx.get(task.task_id) or idx.get(task.turns[0].turn_id if task.turns else "")
        if _jev_abstain(fs, "turn.relationship.task_identity"):
            continue

        turn = task.turns[0] if task.turns else None
        turn_id = turn.turn_id if turn else None
        calls = [(rid, u, m, ts, e, tid) for rid, u, m, ts, e, tid in iter_billed_calls(session) if tid == turn_id]
        if not calls and turn is not None:
            calls = [
                (
                    c.response_id,
                    c.usage,
                    (turn.model_config_active.model if turn.model_config_active else None),
                    c.ts_start,
                    None,
                    turn.turn_id,
                )
                for c in turn.model_calls
                if c.usage is not None
            ]
        first = calls[0] if calls else None
        cache_miss = bool(first and _is_cache_miss(first[1]))
        occupancy = _occupancy(first[1]) if first else 0
        ctx_tokens = turn.context_window_tokens if turn and turn.context_window_tokens else window
        under_pressure = ctx_tokens > 0 and occupancy / ctx_tokens > pressure
        compacted = _compaction_in_turn(session, turn_id)
        after_gap = bool(first and _gap_before(session, first[3], gap))
        proxy_trigger = (cache_miss and after_gap) or under_pressure or compacted

        unrelated = edge == "unrelated"
        fires = unrelated and reuse == 0 and proxy_trigger

        model = (turn.model_config_active.model if turn and turn.model_config_active else None) or session.model
        entry, status = config.price_lookup(model)
        text = turn.user_input.text if turn else ""
        spans = [turn_id or task.task_id]

        if fires and monetary_enabled(status) and entry is not None and first is not None:
            inv_in = input_cost_of(first[1], entry, status)
            # If the turn has several calls, sum input-side invoice.
            if len(calls) > 1:
                total = 0.0
                for _rid, usage, _m, _ts, _e, _tid in calls:
                    total += input_cost_of(usage, entry, status).amount or 0.0
                inv_in = Money(amount=total, view=CostView.invoice, pricing_status=status)
            base_low, base_high, assumptions = _fresh_baseline_range(text, occupancy, entry.input)
            invoice_amt = inv_in.amount or 0.0
            # burden = invoice input - fresh baseline; larger baseline means smaller burden
            burden_low = max(0.0, invoice_amt - base_high)
            burden_high = max(0.0, invoice_amt - base_low)
            money = Money(
                amount=(burden_low + burden_high) / 2.0,
                view=CostView.counterfactual,
                pricing_status=PricingStatus.exact,  # prices exact; quantity is a modelled range
                amount_low=burden_low,
                amount_high=burden_high,
                assumptions=assumptions,
                label="fresh-session counterfactual (range)",
            )
            out.append(
                Diagnosis(
                    id="CONTINUATION_BURDEN",
                    view=CostView.counterfactual,
                    amount=money,
                    pricing_status=money.pricing_status,
                    evidence={
                        "edge": edge,
                        "artifact_reuse": reuse,
                        "cache_miss": cache_miss,
                        "after_gap": after_gap,
                        "context_occupancy": occupancy,
                        "context_pressure": under_pressure,
                        "compaction": compacted,
                        "invoice_input": inv_in.as_dict(),
                    },
                    spans=spans,
                    group_id=f"continuation:{task.task_id}",
                    counts_as_agent_waste=True,
                )
            )
        else:
            out.append(
                Diagnosis(
                    id="CONTEXT_CARRYOVER",
                    view=CostView.invoice,
                    amount=None,
                    pricing_status=status,
                    evidence={
                        "edge": edge,
                        "artifact_reuse": reuse,
                        "cache_miss": cache_miss,
                        "context_occupancy": occupancy,
                        "compaction": compacted,
                        "fired_burden": False,
                    },
                    spans=spans,
                    group_id=f"continuation:{task.task_id}",
                    informational=True,
                    counts_as_agent_waste=False,
                )
            )
    return out


__all__ = ["continuation_diagnoses"]
