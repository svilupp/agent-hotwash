"""Successful repetition and the model rounds spent acquiring information.

Source-record or response-ID joins attribute usage only when exactly one model
usage observation belongs to the same group as its requested tool calls. No
neighboring-event guess is used. These are measured burdens, never savings.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

from agent_hotwash.config import Config, TailsConfig
from agent_hotwash.diagnostics.tails import ModelActivity, TailAnalysis, TailIncident
from agent_hotwash.events import Event, EventKind, ToolCategory, Trace, Usage
from agent_hotwash.primitives.commands import classify_command
from agent_hotwash.primitives.coordination import coordination_kind, is_coordination
from agent_hotwash.primitives.coordination import status_probe as status_probe

_USAGE_FIELDS = ("input", "output", "cache_read", "cache_write")
_PENDING = re.compile(r"\bStatus:\s*running\b|\bAgent is still running\b|\bScript running with\b", re.I)
_PI_ELAPSED = re.compile(
    r"(?m)^(Type: [^\n]+ \| Status: running \| Tool uses: \d+ \| [\d.]+k? token "
    r"\| Context: \d+% \| Duration:) [\d.]+s \(running\)$"
)


def inspection_call(event: Event) -> bool:
    if event.tool_category == ToolCategory.read:
        return True
    cmd = event.tool_args.get("command") or event.tool_args.get("cmd")
    return isinstance(cmd, str) and classify_command(cmd) == "inspect"


def _usage(events: list[Event]) -> Usage:
    return Usage(**{key: sum(getattr(e.usage, key) or 0 for e in events if e.usage) for key in _USAGE_FIELDS})


def work_patterns(trace: Trace, config: TailsConfig) -> tuple[list[ModelActivity], list[TailIncident], dict[str, int]]:
    rows: list[TailIncident] = []
    activity: dict[tuple[str, str | None, str, str | None, bool], ModelActivity] = {}
    coverage: Counter[str] = Counter()

    def add(
        kind: str, sid: str, events: list[Event], value: float, unit: str, label: str, action: str, evidence: dict
    ) -> None:
        events = sorted({e.idx: e for e in events}.values(), key=lambda e: e.idx)
        if not events:
            return
        indices = [e.idx for e in events]
        threshold = getattr(config, kind)
        rows.append(
            TailIncident(
                id=hashlib.sha256(json.dumps([trace.trace_id, sid, kind, indices]).encode()).hexdigest()[:24],
                kind=kind,
                session_id=sid,
                event_indices=indices,
                sources=[e.source for e in events if e.source],
                value=value,
                unit=unit,
                threshold=threshold,
                exceeds_threshold=value >= threshold,
                cohort=kind,
                label=label,
                action=action,
                evidence=evidence,
            )
        )

    for session in [trace.root, *trace.subagents]:
        if "usage_estimated" in session.degraded:
            continue
        events = session.events
        by_call: defaultdict[str, list[Event]] = defaultdict(list)
        by_result: defaultdict[str, list[Event]] = defaultdict(list)
        scopes: dict[int, int] = {}
        groups: defaultdict[tuple[str, str | int], list[Event]] = defaultdict(list)
        request_idx = -1
        for ev in events:
            if ev.kind == EventKind.user_msg and (ev.role_hint is None or str(ev.role_hint) == "user"):
                request_idx = ev.idx
            scopes[ev.idx] = request_idx
            if ev.kind == EventKind.tool_call and ev.call_id:
                by_call[ev.call_id].append(ev)
            elif ev.kind == EventKind.tool_result and ev.call_id:
                by_result[ev.call_id].append(ev)
            if ev.usage and not ev.usage.cumulative:
                coverage["model_usage_observations"] += 1
            if ev.kind not in {EventKind.assistant_msg, EventKind.thinking, EventKind.tool_call} and not (
                ev.response_id and ev.usage and not ev.usage.cumulative
            ):
                continue
            key = (
                ("response", ev.response_id)
                if ev.response_id
                else ("source", ev.source.record_index)
                if ev.source
                else None
            )
            if key:
                groups[key].append(ev)

        pairs = {
            call[0].idx: (call[0], by_result[cid][0])
            for cid, call in by_call.items()
            if len(call) == 1 and len(by_result[cid]) == 1 and by_result[cid][0].idx > call[0].idx
        }
        inspections: defaultdict[int, list[tuple[list[Event], Event]]] = defaultdict(list)
        pending_usage: defaultdict[int, list[Event]] = defaultdict(list)
        # Session-level prices are unsafe when the recorded session switches model.
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
        model = next(iter(models)) if len(models) == 1 else None
        for group in groups.values():
            usage = [e for e in group if e.usage and not e.usage.cumulative]
            if len(usage) != 1 or not session.usage_reliable:
                continue
            calls = [e for e in group if e.kind == EventKind.tool_call]
            category = (
                "no_tools"
                if not calls
                else "inspection"
                if all(inspection_call(e) for e in calls)
                else "coordination"
                if all(is_coordination(e) for e in calls)
                else "execution"
                if all(e.tool_category in {ToolCategory.execute, ToolCategory.write} for e in calls)
                else "mixed"
            )
            coverage["attributed_model_rounds"] += 1
            observed_model = usage[0].usage_model or model
            session_kind = "root" if session is trace.root else "child"
            observed = usage[0].usage
            assert observed is not None
            incomplete = any(getattr(observed, key) is None for key in _USAGE_FIELDS)
            kinds = Counter(coordination_kind(e) for e in calls if is_coordination(e))
            subtype = (next(iter(kinds)) if len(kinds) == 1 else "mixed") if category == "coordination" else None
            item = activity.setdefault(
                (category, observed_model, session_kind, subtype, incomplete),
                ModelActivity(
                    category=category, coordination_kind=subtype, model=observed_model, session_kind=session_kind
                ),
            )
            item.rounds += 1
            item.tool_calls += len(calls)
            item.incomplete_usage_rounds += incomplete
            for kind, count in kinds.items():
                item.coordination_calls[kind] = item.coordination_calls.get(kind, 0) + count
            for key in _USAGE_FIELDS:
                setattr(item.usage, key, (getattr(item.usage, key) or 0) + (getattr(observed, key) or 0))
            scope = scopes[usage[0].idx]
            if category == "inspection":
                inspections[scope].append((calls, usage[0]))
            if calls and all(
                status_probe(e)
                and e.idx in pairs
                and pairs[e.idx][1].ok is not False
                and _PENDING.search(pairs[e.idx][1].output or "")
                for e in calls
            ):
                pending_usage[scope].append(usage[0])

        for scope, rounds in inspections.items():
            if len(rounds) < 3:
                continue
            add(
                "inspection_rounds",
                session.session_id,
                [e for calls, u in rounds for e in [u, *calls]],
                len(rounds),
                "model rounds",
                "Model rounds spent inspecting",
                "Inspect whether known targets can be read together and prior results reused; "
                "preserve dependent discovery.",
                {
                    "request_event_idx": scope,
                    "tool_calls": sum(len(calls) for calls, _ in rounds),
                    "observed_usage": _usage([u for _, u in rounds]).model_dump(),
                    "attribution": "same source record or response ID",
                    "waste_not_inferred": True,
                },
            )

        probes: defaultdict[int, list[tuple[Event, Event]]] = defaultdict(list)
        repeated: defaultdict[tuple[int, str, str], list[tuple[Event, Event]]] = defaultdict(list)
        for call, result in pairs.values():
            scope = scopes[call.idx]
            if status_probe(call):
                probes[scope].append((call, result))
                coverage["nonblocking_status_calls"] += 1
            # Full captured-text identity survives parser clipping. Without it,
            # only an explicitly complete output can support equality.
            digest = result.output_sha256
            if digest is None and result.output is not None and not result.output_truncated:
                digest = hashlib.sha256(result.output.encode()).hexdigest()
            if result.ok is not True or not inspection_call(call) or not digest:
                continue
            signature = json.dumps([call.tool_name, call.op_kind, call.tool_args], sort_keys=True, default=str)
            repeated[(scope, signature, digest)].append((call, result))
        for scope, seq in probes.items():
            if len(seq) < 3:
                continue
            pending = sum(r.ok is not False and bool(_PENDING.search(r.output or "")) for _, r in seq)
            usages = pending_usage[scope]
            # Count identical pending responses only for the same invocation
            # target. Changing progress and different agents stay separate.
            pending_fingerprints: Counter[tuple[str, str]] = Counter()
            progress_fingerprints: Counter[tuple[str, str]] = Counter()
            for call, result in seq:
                if result.ok is False or not _PENDING.search(result.output or ""):
                    continue
                digest = result.output_sha256
                if digest is None and result.output is not None and not result.output_truncated:
                    digest = hashlib.sha256(result.output.encode()).hexdigest()
                if digest:
                    signature = json.dumps([call.tool_name, call.tool_args], sort_keys=True, default=str)
                    pending_fingerprints[(signature, digest)] += 1
                    progress_digest = digest
                    if result.output is not None and not result.output_truncated:
                        # Only the known Pi status header's elapsed clock is
                        # ignored. Tool counts, tokens, context and free text
                        # must still match byte for byte.
                        normalized = _PI_ELAPSED.sub(r"\1 [elapsed] (running)", result.output)
                        progress_digest = hashlib.sha256(normalized.encode()).hexdigest()
                    progress_fingerprints[(signature, progress_digest)] += 1
            add(
                "status_probes",
                session.session_id,
                [e for pair in seq for e in pair],
                len(seq),
                "calls",
                "Status probes interleaved with work",
                "Use a blocking wait or completion notification when available; "
                "reduce probes that cannot change the next action.",
                {
                    "request_event_idx": scope,
                    "pending_results": pending,
                    "unchanged_pending_results": sum(n - 1 for n in pending_fingerprints.values()),
                    "unchanged_pending_progress_results": sum(n - 1 for n in progress_fingerprints.values()),
                    "pending_progress_normalization": "Only elapsed seconds in the recognized Pi running-status header",
                    "pending_only_model_rounds": len(usages),
                    "pending_only_model_usage": _usage(usages).model_dump(),
                    "usage_attribution": "same source record or response ID",
                    "interleaved_work_preserved": True,
                    "waste_not_inferred": True,
                },
            )
        for (scope, _, digest), seq in repeated.items():
            if len(seq) < 2:
                continue
            sizes = [
                r.output_chars_original if r.output_chars_original is not None else len(r.output or "") for _, r in seq
            ]
            if len(set(sizes)) != 1:
                continue
            extra = sizes[0] * (len(seq) - 1)
            if extra < 1000:
                continue
            between = [e for e in events if seq[0][0].idx < e.idx < seq[-1][1].idx]
            add(
                "output_repetition",
                session.session_id,
                [e for pair in seq for e in pair],
                extra,
                "repeated characters",
                "Identical output from repeated inspection",
                "Reuse the captured result when still valid; "
                "check intervening changes, compaction, and freshness requirements.",
                {
                    "request_event_idx": scope,
                    "invocations": len(seq),
                    "output_sha256": digest,
                    "single_output_chars": sizes[0],
                    "intervening_write_calls": sum(
                        e.kind == EventKind.tool_call and e.tool_category == ToolCategory.write for e in between
                    ),
                    "intervening_compactions": sum(e.kind == EventKind.compaction for e in between),
                    "same_invocation": True,
                    "waste_not_inferred": True,
                },
            )
    return list(activity.values()), rows, dict(coverage)


def price_model_activity(tails: TailAnalysis, config: Config) -> None:
    """Price complete observed usage; never price missing or model-switched rows."""
    from agent_hotwash.diagnostics.cost_views import invoice_of

    for row in tails.model_activity:
        if row.incomplete_usage_rounds or row.model is None:
            continue
        price, status = config.price_lookup(row.model)
        row.estimated_cost = invoice_of(row.usage, price, status).amount
        row.pricing_status = status.value
        if price is not None:
            row.estimated_cost_components = {
                field: (getattr(row.usage, field) or 0) * getattr(price, field) / 1_000_000 for field in _USAGE_FIELDS
            }
