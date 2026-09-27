"""Full cache-write transitions following observed warm-cache responses.

This is a token pattern, not a cache-key, prefix-identity or expiry detector.
Missing usage and observed context boundaries break the sequence.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import suppress
from itertools import pairwise

from agent_hotwash.config import Config, TailsConfig
from agent_hotwash.diagnostics.handover_cache import observed_model_calls
from agent_hotwash.diagnostics.repetition import request_scopes
from agent_hotwash.diagnostics.tails import TailIncident
from agent_hotwash.events import EventKind, Trace
from agent_hotwash.primitives.coordination import coordination_kind, is_coordination


def cache_rebuilds(trace: Trace, config: TailsConfig, *, min_write_tokens: int) -> list[TailIncident]:
    out = []
    for session in [trace.root, *trace.subagents]:
        if not session.usage_reliable or "usage_estimated" in session.degraded:
            continue
        scopes = request_scopes(session.events)
        calls = sorted(observed_model_calls(session), key=lambda item: item[0].event_start)
        groups: defaultdict[int, list[dict]] = defaultdict(list)
        for (prior, prior_model), (current, model) in pairwise(calls):
            if (
                prior.usage is None
                or current.usage is None
                or not model
                or model != prior_model
                or scopes.get(prior.event_start, -1) != scopes.get(current.event_start, -1)
                or scopes.get(current.event_start, -1) < 0
            ):
                continue
            fields = (prior.usage.cache_read, current.usage.cache_read, current.usage.cache_write)
            if any(v is None for v in fields):
                continue
            if not (
                prior.usage.cache_read
                and current.usage.cache_read == 0
                and (current.usage.cache_write or 0) > 0
                and (current.usage.cache_write or 0) >= min_write_tokens
            ):
                continue
            if any(
                e.kind == EventKind.compaction or e.raw_type == "context_edit"
                for e in session.events
                if prior.event_start < e.idx <= current.event_end
            ):
                continue
            current_calls = [
                e
                for e in session.events
                if current.event_start <= e.idx <= current.event_end and e.kind == EventKind.tool_call
            ]
            pure_coordination = bool(current_calls) and all(is_coordination(e) for e in current_calls)
            subtypes = {coordination_kind(e) for e in current_calls}
            gap = None
            if prior.ts_end and current.ts_end:
                with suppress(TypeError):
                    gap = (current.ts_end - prior.ts_end).total_seconds()
            groups[scopes[current.event_start]].append(
                {
                    "prior_event_start": prior.event_start,
                    "event_start": current.event_start,
                    "event_end": current.event_end,
                    "response_id": current.response_id,
                    "model": model,
                    "prior_cache_read": prior.usage.cache_read,
                    "cache_read": current.usage.cache_read,
                    "cache_write": current.usage.cache_write,
                    "response_gap_seconds": gap if gap is not None and gap >= 0 else None,
                    "coordination_kind": (next(iter(subtypes)) if len(subtypes) == 1 else "mixed")
                    if pure_coordination
                    else None,
                }
            )
        for scope, transitions in groups.items():
            indices = sorted(
                {int(r[k]) for r in transitions for k in ("prior_event_start", "event_start", "event_end")}
            )
            events = [e for e in session.events if e.idx in indices]
            out.append(
                TailIncident(
                    id=hashlib.sha256(
                        json.dumps([trace.trace_id, session.session_id, "cache_rebuilds", indices]).encode()
                    ).hexdigest()[:24],
                    kind="cache_rebuilds",
                    session_id=session.session_id,
                    event_indices=indices,
                    sources=[e.source for e in events if e.source],
                    value=len(transitions),
                    unit="warm-to-full-write transitions",
                    threshold=config.cache_rebuilds,
                    exceeds_threshold=len(transitions) >= config.cache_rebuilds,
                    cohort="cache_rebuilds",
                    label="Full cache writes after observed cache hits",
                    action="Inspect prompt-prefix changes, long response gaps, and context replacement "
                    "at the linked responses; preserve required context.",
                    evidence={
                        "request_event_idx": scope,
                        "transitions": transitions,
                        "cache_write_tokens": sum(r["cache_write"] for r in transitions),
                        "min_write_tokens": min_write_tokens,
                        "prefix_identity_observed": False,
                        "expiry_not_proven": True,
                        "savings_not_inferred": True,
                    },
                )
            )
    return out


def price_cache_rebuilds(rows: list[TailIncident], config: Config) -> None:
    for row in rows:
        if row.kind != "cache_rebuilds":
            continue
        amounts = []
        for transition in row.evidence["transitions"]:
            price, status = config.price_lookup(transition["model"])
            transition["pricing_status"] = status.value
            transition["cache_write_cost"] = (
                (transition["cache_write"] * price.cache_write / 1_000_000) if price else None
            )
            if transition["cache_write_cost"] is not None:
                amounts.append(transition["cache_write_cost"])
        row.evidence["priced_transitions"] = len(amounts)
        row.evidence["observed_cache_write_cost"] = sum(amounts) if amounts else None
