"""Normative cost views: invoice, origin-attributed, and counterfactual (§7.1).

Every monetary figure carries a view name and a pricing status. Reasoning tokens
are an informational subset of output and are never added into a cost.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.config import Config, PriceEntry
from agent_hotwash.events import PricingStatus, Session, Trace, Usage

if TYPE_CHECKING:
    from agent_hotwash.structure.episodes import Episode
    from agent_hotwash.structure.tasks import Task


class CostView(StrEnum):
    invoice = "invoice"
    origin_attributed = "origin-attributed"
    counterfactual = "counterfactual"


def mtok(n: int | None) -> float:
    return (n or 0) / 1_000_000.0


def worse_status(*statuses: PricingStatus) -> PricingStatus:
    rank = {PricingStatus.exact: 0, PricingStatus.estimated: 1, PricingStatus.unknown: 2}
    worst = PricingStatus.exact
    any_ = False
    for status in statuses:
        any_ = True
        if rank[status] > rank[worst]:
            worst = status
    return worst if any_ else PricingStatus.unknown


def monetary_enabled(status: PricingStatus) -> bool:
    """Monetary waste diagnoses require an exact dated price row (C11)."""
    return status is PricingStatus.exact


class Money(BaseModel):
    """One dollar figure, always tagged with its cost view and pricing status."""

    model_config = ConfigDict(extra="ignore")

    amount: float | None = None
    view: CostView
    pricing_status: PricingStatus
    amount_low: float | None = None
    amount_high: float | None = None
    assumptions: list[str] = Field(default_factory=list)
    label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "amount": self.amount,
            "view": self.view.value,
            "pricing_status": self.pricing_status.value,
        }
        if self.amount_low is not None:
            out["amount_low"] = self.amount_low
        if self.amount_high is not None:
            out["amount_high"] = self.amount_high
        if self.assumptions:
            out["assumptions"] = list(self.assumptions)
        if self.label:
            out["label"] = self.label
        return out


class Diagnosis(BaseModel):
    """One diagnostic finding (waste rule, informational indicator, or detector)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    view: CostView = CostView.invoice
    amount: Money | None = None
    pricing_status: PricingStatus = PricingStatus.unknown
    evidence: dict[str, Any] = Field(default_factory=dict)
    spans: list[str] = Field(default_factory=list)
    superseded_ids: list[str] = Field(default_factory=list)
    group_id: str = ""
    corroborates: list[str] = Field(default_factory=list)
    informational: bool = False
    reason: str | None = None
    tier: str | None = None
    counts_as_agent_waste: bool = True


class ResponseCharge(BaseModel):
    """Invoice for one billed model call, after ``response_id`` dedup."""

    model_config = ConfigDict(extra="ignore")

    thread_id: str
    response_id: str | None = None
    ts_start: datetime | None = None
    model: str | None = None
    effort: str | None = None
    invoice: Money
    episode_id: str | None = None
    task_id: str | None = None
    turn_id: str | None = None
    root_task_id: str | None = None
    usage: Usage = Field(default_factory=Usage)


class CostViews(BaseModel):
    """Trace-level cost surfaces. ``diagnoses`` is filled by ``diagnose``."""

    model_config = ConfigDict(extra="ignore")

    invoice: Money
    origin_attributed: Money | None = None
    counterfactual: Money | None = None
    tree_rollup: Money | None = None
    fork_carryover: Money | None = None
    rollup_status: Literal["full", "partial"] = "full"
    per_response: list[ResponseCharge] = Field(default_factory=list)
    diagnoses: list[Diagnosis] = Field(default_factory=list)


def invoice_of(usage: Usage, price_entry: PriceEntry | None, status: PricingStatus) -> Money:
    """Billed charge of one usage row. Reasoning is never added."""
    if price_entry is None:
        return Money(amount=None, view=CostView.invoice, pricing_status=PricingStatus.unknown)
    amount = (
        mtok(usage.input) * price_entry.input
        + mtok(usage.output) * price_entry.output
        + mtok(usage.cache_read) * price_entry.cache_read
        + mtok(usage.cache_write) * price_entry.cache_write
    )
    return Money(amount=amount, view=CostView.invoice, pricing_status=status)


def input_cost_of(usage: Usage, price_entry: PriceEntry | None, status: PricingStatus) -> Money:
    """Invoice input-side cost only (uncached + cache read + cache write)."""
    if price_entry is None:
        return Money(amount=None, view=CostView.invoice, pricing_status=PricingStatus.unknown)
    amount = (
        mtok(usage.input) * price_entry.input
        + mtok(usage.cache_read) * price_entry.cache_read
        + mtok(usage.cache_write) * price_entry.cache_write
    )
    return Money(amount=amount, view=CostView.invoice, pricing_status=status)


def reasoning_spend_of(usage: Usage, price_entry: PriceEntry | None, status: PricingStatus) -> Money:
    """Observed reasoning spend: ``reasoning_output * output price`` (informational)."""
    label = "observed, avoidable amount unknown without matched runs"
    if price_entry is None:
        return Money(
            amount=None,
            view=CostView.invoice,
            pricing_status=PricingStatus.unknown,
            label=label,
        )
    amount = mtok(usage.reasoning_output) * price_entry.output
    return Money(amount=amount, view=CostView.invoice, pricing_status=status, label=label)


def _add_money(parts: Sequence[Money], *, view: CostView) -> Money:
    statuses = [p.pricing_status for p in parts]
    amounts = [p.amount for p in parts if p.amount is not None]
    status = worse_status(*statuses) if statuses else PricingStatus.unknown
    if not amounts:
        return Money(amount=None if status is PricingStatus.unknown else 0.0, view=view, pricing_status=status)
    return Money(amount=sum(amounts), view=view, pricing_status=status)


def _lookup(config: Config, model: str | None) -> tuple[PriceEntry | None, PricingStatus]:
    return config.price_lookup(model)


def iter_sessions(trace: Trace) -> list[Session]:
    return [trace.root, *list(trace.subagents)]


def _dedup_key(response_id: str | None, fallback: str) -> str:
    return response_id if response_id else fallback


def iter_billed_calls(
    session: Session,
) -> list[tuple[str | None, Usage, str | None, datetime | None, str | None, str | None]]:
    """``(response_id, usage, model, ts_start, effort, turn_id)`` deduped by response_id."""
    seen: set[str] = set()
    out: list[tuple[str | None, Usage, str | None, datetime | None, str | None, str | None]] = []

    def _take(
        response_id: str | None,
        usage: Usage | None,
        model: str | None,
        ts: datetime | None,
        effort: str | None,
        turn_id: str | None,
        fallback: str,
    ) -> None:
        if usage is None:
            return
        key = _dedup_key(response_id, fallback)
        if key in seen:
            return
        seen.add(key)
        out.append((response_id, usage, model, ts, effort, turn_id))

    if session.turns and any(t.model_calls for t in session.turns):
        for turn in session.turns:
            cfg = turn.model_config_active
            model = (cfg.model if cfg else None) or session.model
            effort = cfg.reasoning_effort if cfg else None
            for i, call in enumerate(turn.model_calls):
                _take(
                    call.response_id,
                    call.usage,
                    model,
                    call.ts_start,
                    effort,
                    turn.turn_id,
                    f"{session.session_id}:{turn.turn_id}:{i}",
                )
        return out

    for i, ev in enumerate(session.events):
        if ev.usage is None:
            continue
        _take(ev.response_id, ev.usage, session.model, ev.ts, None, ev.turn_id, f"{session.session_id}:ev{i}")
    return out


def session_invoice(session: Session, config: Config) -> Money:
    """Sum per-response usage (dedup ``response_id``). Reasoning is never added."""
    parts: list[Money] = []
    for _rid, usage, model, _ts, _effort, _turn in iter_billed_calls(session):
        entry, status = _lookup(config, model or session.model)
        parts.append(invoice_of(usage, entry, status))
    if not parts:
        _entry, status = _lookup(config, session.model)
        return Money(amount=0.0 if _entry is not None else None, view=CostView.invoice, pricing_status=status)
    return _add_money(parts, view=CostView.invoice)


def _parent_map(trace: Trace) -> dict[str, str]:
    parents: dict[str, str] = {}
    for link in trace.links:
        if link.child_id and link.parent_id and link.child_id not in parents:
            parents[link.child_id] = link.parent_id
    for child in trace.subagents:
        if child.parent_session_id and child.session_id not in parents:
            parents[child.session_id] = child.parent_session_id
    return parents


def _root_session_id(session_id: str, parents: dict[str, str], known: set[str]) -> str:
    seen: set[str] = set()
    cur = session_id
    while cur in parents and parents[cur] in known and cur not in seen:
        seen.add(cur)
        cur = parents[cur]
    return cur


def _missing_parent(trace: Trace) -> bool:
    notes = list(trace.provenance.notes) if trace.provenance else []
    if any("parent not in input" in n for n in notes):
        return True
    linkage = trace.provenance.thread_linkage if trace.provenance else None
    return linkage == "partial"


def _root_task_id(tasks: Sequence[Task] | None, root_session_id: str) -> str:
    if tasks:
        for task in tasks:
            if task.session_id == root_session_id and not task.parent_task:
                return task.task_id
        for task in tasks:
            if not task.parent_task:
                return task.task_id
        return tasks[0].task_id
    return f"{root_session_id}:task0"


def _episode_for_response(
    episodes: Sequence[Episode] | None, response_id: str | None, turn_id: str | None
) -> Episode | None:
    if not episodes:
        return None
    if response_id:
        for ep in episodes:
            if response_id in ep.response_ids:
                return ep
    if turn_id:
        for ep in episodes:
            if ep.turn_id == turn_id:
                return ep
    return None


def collect_response_charges(
    trace: Trace,
    config: Config,
    *,
    episodes: Sequence[Episode] | None = None,
    tasks: Sequence[Task] | None = None,
) -> list[ResponseCharge]:
    """Per-response invoice rows for the tree, globally deduped by (thread, response)."""
    parents = _parent_map(trace)
    known = {s.session_id for s in iter_sessions(trace)}
    root_ids = {s.session_id: _root_session_id(s.session_id, parents, known) for s in iter_sessions(trace)}
    charges: list[ResponseCharge] = []
    seen: set[tuple[str, str]] = set()
    for session in iter_sessions(trace):
        root_sid = root_ids.get(session.session_id, session.session_id)
        root_tid = _root_task_id(tasks, root_sid)
        for i, (rid, usage, model, ts, effort, turn_id) in enumerate(iter_billed_calls(session)):
            key = (session.session_id, rid or f"anon:{i}")
            if key in seen:
                continue
            seen.add(key)
            entry, status = _lookup(config, model or session.model)
            ep = _episode_for_response(episodes, rid, turn_id)
            charges.append(
                ResponseCharge(
                    thread_id=session.session_id,
                    response_id=rid,
                    ts_start=ts,
                    model=model or session.model,
                    effort=effort,
                    invoice=invoice_of(usage, entry, status),
                    episode_id=ep.episode_id if ep else None,
                    task_id=ep.task_id if ep else root_tid,
                    turn_id=turn_id,
                    root_task_id=root_tid,
                    usage=usage,
                )
            )
    return charges


def _fork_child_ids(trace: Trace) -> set[str]:
    out: set[str] = set()
    for link in trace.links:
        kind = link.kind.value if hasattr(link.kind, "value") else str(link.kind)
        if kind == "fork":
            out.add(link.child_id)
    return out


def fork_carryover_line(trace: Trace, config: Config) -> Money | None:
    """Informational billed carryover on forked children; not an extra rollup term."""
    forks = _fork_child_ids(trace)
    if not forks:
        return None
    parts: list[Money] = []
    for session in iter_sessions(trace):
        if session.session_id not in forks:
            continue
        calls = iter_billed_calls(session)
        if not calls:
            continue
        _rid, usage, model, _ts, _effort, _turn = calls[0]
        entry, status = _lookup(config, model or session.model)
        # Inherited context shows up as the first call's input-side tokens.
        parts.append(input_cost_of(usage, entry, status))
    if not parts:
        return None
    money = _add_money(parts, view=CostView.invoice)
    money.label = "fork carryover (informational; already included in incremental invoice)"
    return money


def tree_rollup(trace: Trace, config: Config) -> tuple[Money, Money | None, Literal["full", "partial"]]:
    """Root invoice + descendants' incremental invoice. Carryover is separate."""
    parts = [session_invoice(s, config) for s in iter_sessions(trace)]
    rollup = _add_money(parts, view=CostView.invoice)
    carry = fork_carryover_line(trace, config)
    status: Literal["full", "partial"] = "partial" if _missing_parent(trace) else "full"
    return rollup, carry, status


def _origin_attributed(
    charges: Sequence[ResponseCharge],
    config: Config,
) -> Money:
    """Reallocate billed dollars to the episode that introduced the tokens.

    Cache-read re-billing is charged back to prior origin episodes in proportion
    to tokens they introduced. Always labelled estimated (allocation is modelled).
    """
    if not charges:
        return Money(amount=0.0, view=CostView.origin_attributed, pricing_status=PricingStatus.estimated)

    introduced: list[tuple[str, float]] = []  # origin key, tokens
    allocated: dict[str, float] = defaultdict(float)
    statuses: list[PricingStatus] = []

    for charge in charges:
        origin = charge.episode_id or charge.turn_id or charge.thread_id
        usage = charge.usage
        entry, status = _lookup(config, charge.model)
        oa_status = PricingStatus.unknown if entry is None else PricingStatus.estimated
        statuses.append(oa_status)
        if entry is None:
            continue
        new_tokens = float((usage.input or 0) + (usage.cache_write or 0) + (usage.output or 0))
        uncached_cost = mtok(usage.input) * entry.input + mtok(usage.cache_write) * entry.cache_write
        output_cost = mtok(usage.output) * entry.output
        allocated[origin] += uncached_cost + output_cost
        cache_cost = mtok(usage.cache_read) * entry.cache_read
        total_intro = sum(t for _k, t in introduced)
        if cache_cost and total_intro > 0:
            for key, tokens in introduced:
                allocated[key] += cache_cost * (tokens / total_intro)
        elif cache_cost:
            allocated[origin] += cache_cost
        if new_tokens > 0:
            introduced.append((origin, new_tokens))

    total = sum(allocated.values())
    status = worse_status(*statuses) if statuses else PricingStatus.estimated
    if status is not PricingStatus.unknown:
        status = PricingStatus.estimated
    return Money(amount=total, view=CostView.origin_attributed, pricing_status=status)


def build_cost_views(
    trace: Trace,
    config: Config,
    *,
    episodes: Sequence[Episode] | None = None,
    tasks: Sequence[Task] | None = None,
) -> CostViews:
    """Assemble the three named views plus per-response invoice rows."""
    charges = collect_response_charges(trace, config, episodes=episodes, tasks=tasks)
    rollup, carry, rollup_status = tree_rollup(trace, config)
    origin = _origin_attributed(charges, config)
    return CostViews(
        invoice=rollup,
        origin_attributed=origin,
        counterfactual=None,
        tree_rollup=rollup,
        fork_carryover=carry,
        rollup_status=rollup_status,
        per_response=charges,
    )


def _episodes_for_session(session: Session, episodes: Sequence[Episode] | None) -> list[Episode]:
    if not episodes:
        return []
    session_rids = {rid for rid, *_rest in iter_billed_calls(session) if rid}
    turn_ids = {t.turn_id for t in session.turns}
    matched = [
        ep
        for ep in episodes
        if (session_rids and any(r in session_rids for r in ep.response_ids if r))
        or (ep.turn_id in turn_ids)
        or ep.episode_id.startswith(f"{session.session_id}:")
    ]
    return matched


BilledCall = tuple[str | None, Usage, str | None, datetime | None, str | None, str | None]


def allocate_calls_to_episodes(
    calls: Sequence[BilledCall], episodes: Sequence[Episode]
) -> tuple[dict[str, list[BilledCall]], list[BilledCall]]:
    """Assign each billed call to the atom that owns its ``response_id``.

    A call without a ``response_id`` (unterminated trailing call) goes to the
    single atom of its turn that recorded a ``None`` response id, when there is
    exactly one. Everything else is *unallocated* and returned separately.
    """
    by_rid: dict[str, Episode] = {}
    none_slots: dict[str | None, list[Episode]] = defaultdict(list)
    for ep in episodes:
        for rid in ep.response_ids:
            if rid is None:
                none_slots[ep.turn_id].append(ep)
            elif rid not in by_rid:
                by_rid[rid] = ep
    per_episode: dict[str, list[BilledCall]] = {ep.episode_id: [] for ep in episodes}
    unallocated: list[BilledCall] = []
    for call in calls:
        rid, _usage, _model, _ts, _effort, turn_id = call
        ep = by_rid.get(rid) if rid else None
        if ep is None and rid is None:
            slots = none_slots.get(turn_id, [])
            ep = slots[0] if len(slots) == 1 else None
        if ep is None:
            unallocated.append(call)
        else:
            per_episode[ep.episode_id].append(call)
    return per_episode, unallocated


def _calls_invoice(calls: Sequence[BilledCall], session: Session, config: Config) -> Money:
    """Invoice of a call set, each call priced at *its own* turn's model."""
    parts = [invoice_of(usage, *_lookup(config, model or session.model)) for _rid, usage, model, *_rest in calls]
    if not parts:
        _entry, status = _lookup(config, session.model)
        return Money(amount=0.0, view=CostView.invoice, pricing_status=status)
    return _add_money(parts, view=CostView.invoice)


def phase_spend_diagnoses(
    session: Session,
    config: Config,
    episodes: Sequence[Episode] | None = None,
) -> list[Diagnosis]:
    """Informational PHASE_SPEND lines whose invoice sum equals ``session_invoice``.

    Every atom is priced from the billed calls it owns, each at the model active
    for the call's turn (mixed-model threads price correctly). Calls that no atom
    owns are reported on one extra row tagged ``unallocated`` — never folded
    into the last atom.
    """
    billed = session_invoice(session, config)
    matched = _episodes_for_session(session, episodes)
    calls = iter_billed_calls(session)
    out: list[Diagnosis] = []

    def _row(money: Money, evidence: dict[str, Any], span: str) -> Diagnosis:
        return Diagnosis(
            id="PHASE_SPEND",
            view=CostView.invoice,
            amount=money,
            pricing_status=money.pricing_status,
            evidence=evidence,
            spans=[span],
            group_id=f"phase:{span}",
            informational=True,
            counts_as_agent_waste=False,
        )

    if not matched:
        return [_row(billed, {"session_id": session.session_id, "unallocated": True}, session.session_id)]

    per_episode, unallocated = allocate_calls_to_episodes(calls, matched)
    for ep in matched:
        ep_calls = per_episode.get(ep.episode_id, [])
        money = _calls_invoice(ep_calls, session, config)
        models = sorted({m or session.model or "" for _rid, _u, m, *_rest in ep_calls if (m or session.model)})
        out.append(
            _row(
                money,
                {
                    "episode_id": ep.episode_id,
                    "phase_activity": ep.phase_activity,
                    "phase_purpose": ep.phase_purpose,
                    "trigger": ep.trigger,
                    "termination": ep.termination,
                    "billed_calls": len(ep_calls),
                    "models": models,
                },
                ep.episode_id,
            )
        )
    if unallocated:
        money = _calls_invoice(unallocated, session, config)
        out.append(
            _row(
                money,
                {
                    "session_id": session.session_id,
                    "unallocated": True,
                    "billed_calls": len(unallocated),
                    "response_ids": [rid for rid, *_rest in unallocated],
                },
                f"{session.session_id}:unallocated",
            )
        )
    return out


def walk_monetary(obj: Any, *, _seen: set[int] | None = None) -> list[Money]:
    """Collect every :class:`Money` (and money-shaped dict) under ``obj``."""
    seen = _seen if _seen is not None else set()
    oid = id(obj)
    if oid in seen:
        return []
    seen.add(oid)
    found: list[Money] = []
    if isinstance(obj, Money):
        return [obj]
    if isinstance(obj, dict):
        if "view" in obj and "pricing_status" in obj and ("amount" in obj or "amount_low" in obj):
            with suppress(TypeError, ValueError):
                found.append(
                    Money(
                        amount=obj.get("amount"),
                        view=CostView(obj["view"]),
                        pricing_status=PricingStatus(obj["pricing_status"]),
                        amount_low=obj.get("amount_low"),
                        amount_high=obj.get("amount_high"),
                        assumptions=list(obj.get("assumptions") or []),
                        label=obj.get("label"),
                    )
                )
        for val in obj.values():
            found.extend(walk_monetary(val, _seen=seen))
        return found
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found.extend(walk_monetary(item, _seen=seen))
        return found
    if isinstance(obj, BaseModel):
        found.extend(walk_monetary(obj.model_dump(mode="python"), _seen=seen))
    return found


__all__ = [
    "CostView",
    "CostViews",
    "Diagnosis",
    "Money",
    "ResponseCharge",
    "allocate_calls_to_episodes",
    "build_cost_views",
    "collect_response_charges",
    "fork_carryover_line",
    "input_cost_of",
    "invoice_of",
    "iter_billed_calls",
    "iter_sessions",
    "monetary_enabled",
    "mtok",
    "phase_spend_diagnoses",
    "reasoning_spend_of",
    "session_invoice",
    "tree_rollup",
    "walk_monetary",
    "worse_status",
]
