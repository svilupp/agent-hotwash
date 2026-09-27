"""Observed parent cache transitions around a child-result wait.

This is descriptive. Provider cache keys and a matched counterfactual are not
available, so no transition is attributed to the child or priced as waste.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from agent_hotwash.config import Config
from agent_hotwash.events import Event, EventKind, ModelCall, Session, Trace
from agent_hotwash.primitives.coordination import wait_mode
from agent_hotwash.primitives.handovers import HandoverRecord


class CacheWait(BaseModel):
    handover_id: str
    parent_id: str
    wait_event_idx: int
    wait_seconds: float | None = None
    wait_mode: str = "unspecified"
    duration_band: str = "unknown"
    band_lower_seconds: float | None = None
    band_upper_seconds: float | None = None
    prior_response_id: str | None = None
    next_response_id: str | None = None
    prior_event_start: int | None = None
    next_event_start: int | None = None
    prior_model: str | None = None
    next_model: str | None = None
    prior_input: int | None = None
    next_input: int | None = None
    prior_cache_read: int | None = None
    prior_cache_write: int | None = None
    next_cache_read: int | None = None
    next_cache_write: int | None = None
    compaction_between: bool = False
    context_edit_between: bool = False
    model_switch: bool = False
    new_instructions_between: bool = False
    comparable: bool = False
    cache_rewrite_after_wait: bool | None = None
    prior_input_cost: float | None = None
    next_input_cost: float | None = None
    prior_input_cost_components: dict[str, float] = Field(default_factory=dict)
    next_input_cost_components: dict[str, float] = Field(default_factory=dict)
    prior_pricing_status: str = "unknown"
    next_pricing_status: str = "unknown"
    rewrite_write_cost: float | None = None


def observed_model_calls(session: Session) -> list[tuple[ModelCall, str | None]]:
    # Canonical heuristic call intervals begin after the preceding usage event;
    # they may include a tool result before the next model response. Those
    # intervals cannot locate the first response after a wait. Use actual
    # response/source-record groups, as the work-mix attribution does.
    groups: defaultdict[tuple[str, str | int], list[Event]] = defaultdict(list)
    for event in session.events:
        if event.kind not in {EventKind.assistant_msg, EventKind.thinking, EventKind.tool_call} and not (
            event.response_id and event.usage and not event.usage.cumulative
        ):
            continue
        key = (
            ("response", event.response_id)
            if event.response_id
            else (("source", event.source.record_index) if event.source else None)
        )
        if key:
            groups[key].append(event)
    models = {
        m
        for m in [
            session.model,
            *[e.usage_model for e in session.events],
            *[t.model_config_active.model for t in session.turns],
            *[m.model for t in session.turns for m in t.model_config_revisions],
        ]
        if m
    }
    stable_model = next(iter(models)) if len(models) == 1 else None
    rows = []
    for group in groups.values():
        usage = [event for event in group if event.usage and not event.usage.cumulative]
        event = usage[0] if len(usage) == 1 else None
        rows.append(
            (
                ModelCall(
                    response_id=group[0].response_id,
                    event_start=min(e.idx for e in group),
                    event_end=max(e.idx for e in group),
                    usage=event.usage if event else None,
                    ts_start=next((e.ts for e in group if e.ts is not None), None),
                    ts_end=event.ts if event else None,
                ),
                (event.usage_model or stable_model) if event else None,
            )
        )
    if any(e.usage for group in groups.values() for e in group):
        return rows
    # Explicit call rows remain usable for sources without grouped observations.
    # Do not rescue ambiguous grouped usage with a neighboring-call guess.
    if any(e.usage and (e.response_id or e.source) for e in session.events):
        return []
    return [
        (call, stable_model)
        for turn in session.turns
        for call in turn.model_calls
        if call.usage and not call.usage.cumulative
    ]


def _price(record: CacheWait, config: Config, *, reliable: bool) -> None:
    if not reliable:
        return
    for prefix in ("prior", "next"):
        model = getattr(record, f"{prefix}_model")
        tokens = {k: getattr(record, f"{prefix}_{k}") for k in ("input", "cache_read", "cache_write")}
        if not model or any(v is None for v in tokens.values()):
            continue
        price, status = config.price_lookup(model)
        setattr(record, f"{prefix}_pricing_status", status.value)
        if price is None:
            continue
        components = {k: v * getattr(price, k) / 1_000_000 for k, v in tokens.items()}
        setattr(record, f"{prefix}_input_cost_components", components)
        setattr(record, f"{prefix}_input_cost", sum(components.values()))
    if record.cache_rewrite_after_wait:
        record.rewrite_write_cost = record.next_input_cost_components.get("cache_write")


def _band(record: CacheWait, bounds: tuple[float, ...]) -> None:
    if record.wait_seconds is None:
        return
    low = 0.0
    for high in bounds:
        if record.wait_seconds < high:
            record.duration_band = f"{low:g} <= wait < {high:g}s"
            record.band_lower_seconds, record.band_upper_seconds = low, high
            return
        low = high
    record.duration_band = f"wait >= {low:g}s"
    record.band_lower_seconds = low


def build_cache_waits(
    trace: Trace, handovers: list[HandoverRecord], *, min_write_tokens: int, config: Config | None = None
) -> list[CacheWait]:
    sessions = {s.session_id: s for s in [trace.root, *trace.subagents]}
    rows: list[CacheWait] = []
    calls_by_parent: dict[str, list[tuple[ModelCall, str | None]]] = {}
    for handover in handovers:
        parent = sessions.get(handover.parent_id)
        if parent is None:
            continue
        if parent.session_id not in calls_by_parent:
            calls_by_parent[parent.session_id] = observed_model_calls(parent)
        calls = calls_by_parent[parent.session_id]
        call_counts = Counter(e.call_id for e in parent.events if e.kind == EventKind.tool_call and e.call_id)
        for wait in (event for event in handover.events if event.kind == "wait"):
            wait_call = parent.events[wait.event_idx]
            results = [
                event
                for event in parent.events[wait.event_idx + 1 :]
                if event.kind.value == "tool_result" and event.call_id and event.call_id == wait_call.call_id
            ]
            result = results[0] if len(results) == 1 and call_counts[wait_call.call_id or ""] == 1 else None
            end_idx = result.idx if result else wait.event_idx
            # Include the response which issued the wait itself.
            before = [(call, model) for call, model in calls if call.event_start <= wait.event_idx]
            after = [(call, model) for call, model in calls if result is not None and call.event_start > end_idx]
            prior, prior_model = max(before, key=lambda item: item[0].event_end) if before else (None, None)
            next_call, next_model = min(after, key=lambda item: item[0].event_start) if after else (None, None)
            wait_ts = parent.events[wait.event_idx].ts
            end_ts = result.ts if result else None
            # Check the whole gap between the adjacent model calls. A user
            # instruction after the wait result still changes comparability.
            gap_start = prior.event_end + 1 if prior else wait.event_idx
            gap_end = next_call.event_start if next_call else end_idx + 1
            interval = parent.events[gap_start:gap_end]
            record = CacheWait(
                handover_id=handover.id,
                parent_id=parent.session_id,
                wait_event_idx=wait.event_idx,
                wait_seconds=(end_ts - wait_ts).total_seconds() if wait_ts and end_ts and end_ts >= wait_ts else None,
                wait_mode=wait_mode(wait_call),
                prior_response_id=prior.response_id if prior else None,
                next_response_id=next_call.response_id if next_call else None,
                prior_event_start=prior.event_start if prior else None,
                next_event_start=next_call.event_start if next_call else None,
                prior_model=prior_model,
                next_model=next_model,
                prior_input=prior.usage.input if prior and prior.usage else None,
                next_input=next_call.usage.input if next_call and next_call.usage else None,
                prior_cache_read=prior.usage.cache_read if prior and prior.usage else None,
                prior_cache_write=prior.usage.cache_write if prior and prior.usage else None,
                next_cache_read=next_call.usage.cache_read if next_call and next_call.usage else None,
                next_cache_write=next_call.usage.cache_write if next_call and next_call.usage else None,
                compaction_between=any(ev.kind.value == "compaction" for ev in interval),
                context_edit_between=any(ev.raw_type == "context_edit" for ev in interval),
                new_instructions_between=any(ev.kind.value == "user_msg" for ev in interval),
                model_switch=bool(prior_model and next_model and prior_model != next_model),
            )
            record.comparable = bool(
                parent.usage_reliable
                and "usage_estimated" not in parent.degraded
                and result
                and prior
                and next_call
                and prior_model
                and next_model
                and prior_model == next_model
                and prior.usage
                and next_call.usage
                and prior.usage.cache_read is not None
                and next_call.usage.cache_read is not None
                and next_call.usage.cache_write is not None
                and not record.compaction_between
                and not record.context_edit_between
                and not record.new_instructions_between
            )
            if record.comparable:
                record.cache_rewrite_after_wait = bool(
                    record.next_cache_read == 0
                    and (record.next_cache_write or 0) > 0
                    and (record.next_cache_write or 0) >= min_write_tokens
                )
            _band(record, config.diagnostics.cache_wait_bands_seconds if config else (300, 3600))
            if config:
                _price(record, config, reliable=parent.usage_reliable and "usage_estimated" not in parent.degraded)
            rows.append(record)
    return rows


class CacheWaitCohort(BaseModel):
    model: str | None = None
    wait_mode: str
    duration_band: str
    band_lower_seconds: float | None = None
    band_upper_seconds: float | None = None
    waits: int = 0
    comparable: int = 0
    rewrites: int = 0
    priced_responses: int = 0
    next_input_cost: float | None = None
    rewrite_write_cost: float | None = None


def cache_wait_cohorts(records: list[tuple[str, CacheWait]]) -> list[CacheWaitCohort]:
    """Deduplicate multi-child waits and shared next responses within each cohort.

    One next response can follow waits in different cohorts, so cohort charges
    must not be added into a global total.
    """
    groups: dict[tuple[str | None, str, str], CacheWaitCohort] = {}
    seen_waits: set[tuple[str, str, int]] = set()
    seen_prices: set[tuple[tuple[str | None, str, str], str, str, int | None]] = set()
    seen_rewrites: set[tuple[tuple[str | None, str, str], str, str, int | None]] = set()
    for trace_id, record in records:
        wait_key = (trace_id, record.parent_id, record.wait_event_idx)
        if wait_key in seen_waits:
            continue
        seen_waits.add(wait_key)
        key = (record.next_model, record.wait_mode, record.duration_band)
        group = groups.setdefault(
            key,
            CacheWaitCohort(
                model=record.next_model,
                wait_mode=record.wait_mode,
                duration_band=record.duration_band,
                band_lower_seconds=record.band_lower_seconds,
                band_upper_seconds=record.band_upper_seconds,
            ),
        )
        group.waits += 1
        group.comparable += record.comparable
        group.rewrites += record.cache_rewrite_after_wait is True
        price_key = (key, trace_id, record.parent_id, record.next_event_start)
        if record.next_input_cost is not None and record.next_event_start is not None and price_key not in seen_prices:
            seen_prices.add(price_key)
            group.priced_responses += 1
            group.next_input_cost = (group.next_input_cost or 0) + record.next_input_cost
        if (
            record.rewrite_write_cost is not None
            and record.next_event_start is not None
            and price_key not in seen_rewrites
        ):
            seen_rewrites.add(price_key)
            group.rewrite_write_cost = (group.rewrite_write_cost or 0) + record.rewrite_write_cost
    return sorted(groups.values(), key=lambda g: (g.model or "", g.wait_mode, g.band_lower_seconds or 0))


__all__ = ["CacheWait", "CacheWaitCohort", "build_cache_waits", "cache_wait_cohorts", "observed_model_calls"]
