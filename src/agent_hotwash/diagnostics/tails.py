"""Measured extremes, not estimates of avoidable waste.

Keep separate units and cohorts. Timestamp differences are observed round trips,
not CPU time; unfinished delegations are censored observations, never proof of a
hung worker. All retained rows carry source coordinates and an intervention.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from statistics import median
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from agent_hotwash.config import TailsConfig
from agent_hotwash.events import Event, EventKind, SourceRef, ToolCategory, Trace, Usage

if TYPE_CHECKING:
    from agent_hotwash.primitives.handovers import HandoverRecord


class TailIncident(BaseModel):
    id: str
    kind: str
    session_id: str
    event_indices: list[int]
    sources: list[SourceRef] = Field(default_factory=list)
    value: float
    unit: str
    threshold: float
    exceeds_threshold: bool
    cohort: str
    cohort_n: int = 0
    median_ratio: float | None = None
    label: str
    action: str
    assessment: Literal["observation", "supported_opportunity", "justified_repeat", "reuse_visible", "unclear"] = (
        "observation"
    )
    assessment_reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class TailSummary(BaseModel):
    cohort: str
    unit: str
    n: int
    p50: float
    p95: float
    p99: float
    maximum: float
    top_one_percent_share: float | None = None
    # Sum of overlapping observations is burden, never elapsed wall time.
    observed_sum: float


class ModelActivity(BaseModel):
    category: str
    coordination_kind: str | None = None
    coordination_calls: dict[str, int] = Field(default_factory=dict)
    session_kind: str = "root"
    model: str | None = None
    rounds: int = 0
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    incomplete_usage_rounds: int = 0
    estimated_cost: float | None = None
    estimated_cost_components: dict[str, float] = Field(default_factory=dict)
    pricing_status: str = "unknown"


class TailAnalysis(BaseModel):
    incidents: list[TailIncident] = Field(default_factory=list)
    distributions: list[TailSummary] = Field(default_factory=list)
    coverage: dict[str, int] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    retained_per_cohort: int = 10
    model_activity: list[ModelActivity] = Field(default_factory=list)


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    try:
        value = (end - start).total_seconds()
    except TypeError:
        return None
    return value if value >= 0 else None


def attach_tail_excerpts(trace: Trace, tails: TailAnalysis, secret_patterns: list[str]) -> None:
    """Human-readable evidence for retained call extremes, redacted before clipping."""
    from agent_hotwash.semantic.redact import redact_state

    sessions = {s.session_id: {e.idx: e for e in s.events} for s in [trace.root, *trace.subagents]}
    for row in tails.incidents:
        if row.kind not in {"tool_latency", "parent_wait", "output_volume"}:
            continue
        events = [sessions[row.session_id][idx] for idx in row.event_indices]
        call = next((e for e in events if e.kind == EventKind.tool_call), None)
        result = next((e for e in events if e.kind == EventKind.tool_result), None)
        if call is None:
            continue
        row.evidence.setdefault("tool", call.tool_name or call.op_kind or "unknown")
        request = call.tool_args.get("command") or call.tool_args.get("cmd") or call.tool_args
        payload = redact_state(
            {
                "request": request,
                "result": (result.diagnostic_excerpt or result.error_text or result.output) if result else None,
            },
            secret_patterns,
        )
        for key in ("request", "result"):
            value = payload[key]
            if value is None:
                continue
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            row.evidence[f"{key}_excerpt"] = text[:600]
            row.evidence[f"{key}_excerpt_clipped"] = len(text) > 600 or bool(
                key == "result" and result and result.output_truncated
            )


def _wait(call: Event) -> bool:
    return call.op_kind == "agent.wait" or (call.tool_name or "").lower() in {
        "wait_agent",
        "get_subagent_result",
        "wait",
        "collaboration.wait_agent",
    }


def _poll(call: Event) -> bool:
    return (
        _wait(call)
        or call.op_kind == "agent.status"
        or (call.tool_name or "").lower()
        in {
            "write_stdin",
            "functions.write_stdin",
            "list_agents",
            "collaboration.list_agents",
        }
    )


def build_tails(
    trace: Trace,
    handovers: list[HandoverRecord],
    config: TailsConfig | None = None,
    *,
    cache_min_write_tokens: int = 10_000,
) -> TailAnalysis:
    """Retain top ten per cohort plus every threshold crossing, even in small runs."""
    config = config or TailsConfig()
    thresholds = config.model_dump(exclude={"retained_per_cohort", "max_semantic_incidents", "max_expense_reviews"})
    out = TailAnalysis(thresholds=thresholds, retained_per_cohort=config.retained_per_cohort)
    rows: list[TailIncident] = []
    coverage: defaultdict[str, int] = defaultdict(int)

    def add(
        kind: str,
        sid: str,
        events: list[Event],
        value: float,
        unit: str,
        cohort: str,
        label: str,
        action: str,
        **evidence: Any,
    ) -> None:
        indices = [e.idx for e in events]
        key = json.dumps([trace.trace_id, sid, kind, indices], separators=(",", ":"))
        threshold = thresholds[kind]
        rows.append(
            TailIncident(
                id=hashlib.sha256(key.encode()).hexdigest()[:24],
                kind=kind,
                session_id=sid,
                event_indices=indices,
                sources=[e.source for e in events if e.source is not None],
                value=value,
                unit=unit,
                threshold=threshold,
                exceeds_threshold=value >= threshold,
                cohort=cohort,
                label=label,
                action=action,
                evidence=evidence,
            )
        )

    sessions = {s.session_id: s for s in [trace.root, *trace.subagents]}
    for session in sessions.values():
        if "usage_estimated" in session.degraded:
            coverage["estimated_sessions_excluded"] += 1
            continue
        for ev in session.events:
            usage = ev.usage
            if usage is None or usage.cumulative or not session.usage_reliable:
                continue
            # Parsers normalize usage to per-response deltas; never add reasoning
            # tokens to output, since providers may include them already.
            for kind, value in (("model_input", usage.input), ("model_output", usage.output)):
                if value is None or value < 0:
                    continue
                coverage[f"{kind}_observations"] += 1
                add(
                    kind,
                    session.session_id,
                    [ev],
                    value,
                    "tokens",
                    f"{kind}:{session.model or 'unknown'}",
                    "Large uncached model input" if kind == "model_input" else "Large model output",
                    "Inspect context reuse and repeated context construction."
                    if kind == "model_input"
                    else "Inspect whether reasoning or generated output needs a tighter budget.",
                    model=session.model,
                    model_basis="session",
                    savings_not_inferred=True,
                )
        # Reused or missing IDs cannot provide an unambiguous time join.
        calls: defaultdict[str, list[Event]] = defaultdict(list)
        results: defaultdict[str, list[Event]] = defaultdict(list)
        for ev in session.events:
            if ev.kind is EventKind.tool_call and ev.call_id:
                calls[ev.call_id].append(ev)
            if ev.kind is EventKind.tool_result and ev.call_id:
                results[ev.call_id].append(ev)
        pairs: dict[int, tuple[Event, Event]] = {}
        for ev in session.events:
            if ev.kind is not EventKind.tool_call:
                continue
            coverage["tool_calls"] += 1
            matched = results.get(ev.call_id or "", [])
            if len(calls.get(ev.call_id or "", [])) != 1 or len(matched) != 1 or matched[0].idx <= ev.idx:
                coverage["unpaired_or_ambiguous_calls"] += 1
                continue
            result = matched[0]
            pairs[ev.idx] = (ev, result)
            coverage["paired_calls"] += 1
            tool = ev.tool_name or ev.op_kind or "unknown"
            elapsed = _seconds(ev.ts, ev.ts_end or result.ts_end or result.ts)
            if elapsed is not None:
                coverage["timed_calls"] += 1
                kind = "parent_wait" if _wait(ev) else "tool_latency"
                add(
                    kind,
                    session.session_id,
                    [ev, result],
                    elapsed,
                    "seconds",
                    f"{kind}:{tool}",
                    "Observed wait" if _wait(ev) else "Observed tool round trip",
                    "Inspect the slow operation; narrow scope, cache work, or set a timeout "
                    "after checking its purpose.",
                    tool=tool,
                    timing_basis="call_to_observed_end",
                    outcome=result.ok,
                    async_completion_not_inferred=True,
                )
            else:
                coverage["missing_or_invalid_call_timing"] += 1
            size = result.output_chars_original
            if size is None and not result.output_truncated and result.output is not None:
                size = len(result.output or "")
            if size is not None and size >= 0:
                coverage["sized_outputs"] += 1
                add(
                    "output_volume",
                    session.session_id,
                    [ev, result],
                    size,
                    "characters",
                    f"output_volume:{tool}",
                    "Large tool output",
                    "Filter or summarize at the producer; write the full result to an artifact.",
                    original_size_known=result.output_chars_original is not None,
                    token_cost_not_inferred=True,
                )

        retries: dict[str, list[tuple[Event, Event]]] = {}
        chain: list[tuple[Event, Event]] = []
        polls: dict[str, list[Event]] = {}

        def flush_retry(
            key: str,
            *,
            retries: dict[str, list[tuple[Event, Event]]] = retries,
            session_id: str = session.session_id,
            session_events: list[Event] = session.events,
        ) -> None:
            attempts = retries.pop(key, [])
            failed = sum(r.ok is False for _, r in attempts)
            if len(attempts) < 2 or failed < 2:
                return
            events = [e for pair in attempts for e in pair]
            edits = sum(
                e.kind is EventKind.tool_call and e.tool_category is ToolCategory.write
                for e in session_events
                if events[0].idx < e.idx < events[-1].idx
            )
            add(
                "retry_attempts",
                session_id,
                events,
                len(attempts),
                "attempts",
                "retry_attempts",
                "Repeated verification with intervening edits" if edits else "Repeated identical invocation",
                "Inspect why repeated edits did not satisfy the check; isolate the failing fixture or prerequisite."
                if edits
                else "After repeated failure, inspect the diagnostic and change the approach before retrying.",
                intervening_edit_calls=edits,
                failed_results=failed,
                recovered=attempts[-1][1].ok,
                elapsed_seconds=_seconds(events[0].ts, events[-1].ts),
                signature_hash=hashlib.sha256(key.encode()).hexdigest()[:16],
            )

        def flush_chain(*, chain: list[tuple[Event, Event]] = chain, session_id: str = session.session_id) -> None:
            if len(chain) >= 3:
                events = [e for pair in chain for e in pair]
                add(
                    "failure_chain",
                    session_id,
                    events,
                    len(chain),
                    "failed calls",
                    "failure_chain",
                    "Consecutive failed operations",
                    "Check whether these failures share a goal; repair the prerequisite or simplify the execution.",
                    same_goal_unverified=True,
                    elapsed_seconds=_seconds(events[0].ts, events[-1].ts),
                )
            chain.clear()

        def flush_polls(*, polls: dict[str, list[Event]] = polls, session_id: str = session.session_id) -> None:
            for events in polls.values():
                if len(events) >= 3:
                    add(
                        "poll_amplification",
                        session_id,
                        events,
                        len(events),
                        "calls",
                        "poll_amplification",
                        "Repeated status polling",
                        "Use completion notifications or longer blocking waits when supported.",
                        elapsed_seconds=_seconds(events[0].ts, events[-1].ts),
                    )
            polls.clear()

        for ev in session.events:
            if (ev.kind is EventKind.user_msg and ev.role_hint is None) or (
                ev.kind is EventKind.user_msg and str(ev.role_hint) == "user"
            ):
                for key in list(retries):
                    flush_retry(key)
                flush_chain()
                flush_polls()
            if ev.kind is not EventKind.tool_call:
                continue
            if _poll(ev):
                key = json.dumps([ev.tool_name, ev.tool_args], sort_keys=True, default=str)
                polls.setdefault(key, []).append(ev)
                continue
            flush_polls()
            pair = pairs.get(ev.idx)
            if pair is None:
                for key in list(retries):
                    flush_retry(key)
                flush_chain()
                continue
            _, result = pair
            key = json.dumps([ev.tool_name, ev.op_kind, ev.tool_args], sort_keys=True, default=str)
            if result.ok is False:
                retries.setdefault(key, []).append(pair)
                chain.append(pair)
            else:
                if key in retries:
                    retries[key].append(pair)
                    flush_retry(key)
                flush_chain()
        for key in list(retries):
            flush_retry(key)
        flush_chain()
        flush_polls()

    # Cost-tail investigation surfaced wide delegation and repeated large cached
    # contexts. Group by observed user boundary rather than whole-session totals.
    spawns: defaultdict[str, set[int]] = defaultdict(set)
    for handover in handovers:
        for event in handover.events:
            if event.kind == "spawn":
                spawns[handover.parent_id].add(event.event_idx)
    for session in sessions.values():
        if "usage_estimated" in session.degraded:
            continue
        groups: list[list[Event]] = []
        for ev in session.events:
            if ev.kind is EventKind.user_msg and (ev.role_hint is None or str(ev.role_hint) == "user"):
                groups.append([])
            if groups:
                groups[-1].append(ev)
        for group in groups:
            delegated = [e for e in group if e.idx in spawns[session.session_id]]
            if len(delegated) >= 2:
                add(
                    "delegation_fanout",
                    session.session_id,
                    delegated,
                    len(delegated),
                    "spawns",
                    "delegation_fanout",
                    "Wide delegation within one request",
                    "Check scope separation and setup overhead; stage delegation when work depends on earlier results.",
                    request_event_idx=group[0].idx,
                    waste_not_inferred=True,
                )
            cached = [
                e
                for e in group
                if e.usage is not None
                and not e.usage.cumulative
                and e.usage.cache_read is not None
                and e.usage.cache_read > 0
            ]
            if len(cached) >= 5 and session.usage_reliable:
                add(
                    "context_replay",
                    session.session_id,
                    cached,
                    sum(e.usage.cache_read or 0 for e in cached if e.usage is not None),
                    "cached input tokens",
                    "context_replay",
                    "Repeated cached context within one request",
                    "Inspect parent context size and number of model calls; use bounded summaries at task boundaries.",
                    request_event_idx=group[0].idx,
                    model_calls=len(cached),
                    savings_not_inferred=True,
                )

            created = [
                e for e in group if e.usage is not None and not e.usage.cumulative and (e.usage.cache_write or 0) > 0
            ]
            if len(created) >= 5 and session.usage_reliable:
                add(
                    "cache_creation",
                    session.session_id,
                    created,
                    sum(e.usage.cache_write or 0 for e in created if e.usage is not None),
                    "cache-write input tokens",
                    "cache_creation",
                    "Cache creation accumulated within one request",
                    "Inspect prompt growth and cache-prefix changes; "
                    "stabilize reusable prefixes and trim repeated scaffolding.",
                    request_event_idx=group[0].idx,
                    model_calls=len(created),
                    invalidation_not_proven=True,
                    savings_not_inferred=True,
                )

    for row in handovers:
        coverage["delegations"] += 1
        parent = sessions.get(row.parent_id)
        spawn = next((e for e in row.events if e.kind == "spawn"), None)
        if parent is None or spawn is None:
            continue
        start = next((e for e in parent.events if e.idx == spawn.event_idx), None)
        if start is None:
            continue
        lifetime = row.spawn_to_final_seconds
        kind = "delegation_lifetime"
        if lifetime is None and row.status in {"unknown", "running-at-capture", "result-unavailable"}:
            kind = "delegation_open"
            timestamps = [e.ts for e in parent.events if e.ts is not None]
            lifetime = _seconds(spawn.ts, max(timestamps) if timestamps else None)
        if lifetime is None or lifetime < 0:
            coverage["delegations_without_duration"] += 1
            continue
        add(
            kind,
            row.parent_id,
            [start],
            lifetime,
            "seconds",
            kind,
            "Delegation completion unobserved at capture" if kind == "delegation_open" else "Long delegation lifetime",
            "Record explicit completion or cancellation and inspect the last observed progress before tuning deadlines."
            if kind == "delegation_open"
            else "Inspect child progress and parent blocking; use a deadline and a bounded return contract.",
            handover_id=row.id,
            status=row.status,
            child_id=row.child_id,
            censored=kind == "delegation_open",
            hang_proven=False,
            includes_continuations=row.continuation_count > 0,
            child_transcript_present=row.child_transcript_present,
        )

    from agent_hotwash.diagnostics.cache_rebuilds import cache_rebuilds
    from agent_hotwash.diagnostics.repetition import repeated_work
    from agent_hotwash.diagnostics.work import work_patterns

    activity, work_rows, work_coverage = work_patterns(trace, config)
    out.model_activity = activity
    rows.extend(work_rows)
    rows.extend(repeated_work(trace, handovers, config))
    rows.extend(cache_rebuilds(trace, config, min_write_tokens=cache_min_write_tokens))
    coverage.update(work_coverage)
    cohorts: defaultdict[str, list[TailIncident]] = defaultdict(list)
    for row in rows:
        cohorts[row.cohort].append(row)
    for cohort, members in sorted(cohorts.items()):
        members.sort(key=lambda r: (-r.value, r.id))
        values = sorted(r.value for r in members)
        n = len(values)
        p50 = float(median(values))
        total = sum(values)
        out.distributions.append(
            TailSummary(
                cohort=cohort,
                unit=members[0].unit,
                n=n,
                p50=p50,
                p95=values[math.ceil(n * 0.95) - 1],
                p99=values[math.ceil(n * 0.99) - 1],
                maximum=values[-1],
                observed_sum=total,
                top_one_percent_share=sum(values[-max(1, math.ceil(n * 0.01)) :]) / total if total else None,
            )
        )
        for rank, row in enumerate(members):
            row.cohort_n = n
            row.median_ratio = row.value / p50 if p50 > 0 and n >= 5 else None
            if rank < out.retained_per_cohort or row.exceeds_threshold:
                out.incidents.append(row)
    out.coverage = dict(coverage)
    return out
