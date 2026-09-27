"""Repeated successful work: measurements first, reasons assessed separately.

An earlier return must be visible in the parent's own event stream before a
new delegation can qualify. File overlap is only a candidate selector; it does
not establish that two workers were asked the same question.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

from agent_hotwash.config import TailsConfig
from agent_hotwash.diagnostics.tails import TailIncident
from agent_hotwash.events import Event, EventKind, Trace
from agent_hotwash.primitives.commands import classify_command, segment_head, split_segments, strip_shell_wrapper
from agent_hotwash.primitives.handovers import HandoverRecord


def _review_candidate(handover: HandoverRecord, call: Event) -> bool:
    """Cheap, deliberately permissive prefilter; JeV must still establish scope."""
    role = (handover.role or "").lower()
    if role in {"implementer", "worker"}:
        return False
    if role in {"advisor", "reviewer"}:
        return True
    prompt = str(call.tool_args.get("prompt") or call.tool_args.get("message") or "")
    return bool(re.search(r"\b(?:review|audit|verify|verification|validate|validation|qa|fact.check)\b", prompt, re.I))


def _check_command(command: str) -> bool:
    """The shared intent classifier also includes formatters and arbitrary Make targets."""
    for segment in split_segments(strip_shell_wrapper(command)):
        head = segment_head(segment)
        if head is None or classify_command(segment) != "build_test":
            continue
        tool, args = head
        if tool == "uv" and args[:1] == ["run"]:
            if _check_command(" ".join(args[1:])):
                return True
            continue
        if tool == "make":
            if any(
                re.fullmatch(r"(?:check|test|lint|typecheck|type-check|build|validate)(?:[-:].+)?", a) for a in args
            ):
                return True
            continue
        if tool == "ruff" and ((args[:1] == ["format"] and "--check" not in args) or "--fix" in args):
            continue
        return True
    return False


def request_scopes(events: list[Event]) -> dict[int, int]:
    scope = -1
    out = {}
    for ev in events:
        if ev.kind == EventKind.user_msg and (ev.role_hint is None or str(ev.role_hint) == "user"):
            scope = ev.idx
        out[ev.idx] = scope
    return out


def repeated_work(trace: Trace, handovers: list[HandoverRecord], config: TailsConfig) -> list[TailIncident]:
    rows: list[TailIncident] = []

    def add(kind: str, sid: str, events: list[Event], value: int, label: str, action: str, evidence: dict) -> None:
        events = sorted({e.idx: e for e in events}.values(), key=lambda e: e.idx)
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
                unit="prior completed delegations" if kind == "delegation_repetition" else "repeat invocations",
                threshold=threshold,
                exceeds_threshold=value >= threshold,
                cohort=kind,
                label=label,
                action=action,
                evidence={**evidence, "waste_not_inferred": True},
            )
        )

    for session in [trace.root, *trace.subagents]:
        if "usage_estimated" in session.degraded:
            continue
        events = {e.idx: e for e in session.events}
        scopes = request_scopes(session.events)
        call_counts = Counter(e.call_id for e in session.events if e.kind == EventKind.tool_call and e.call_id)
        result_counts = Counter(e.call_id for e in session.events if e.kind == EventKind.tool_result and e.call_id)
        agent_counts = Counter(h.agent_id for h in handovers if h.parent_id == session.session_id and h.agent_id)
        launches: list[tuple[HandoverRecord, Event, Event | None]] = []
        for h in handovers:
            if h.parent_id != session.session_id or h.request_visibility != "plaintext":
                continue
            spawn = next((e for e in h.events if e.kind == "spawn" and e.session_id == session.session_id), None)
            if not spawn or spawn.event_idx not in events or h.status == "spawn-rejected":
                continue
            spawn_call = events[spawn.event_idx]
            if (spawn_call.call_id and call_counts[spawn_call.call_id] != 1) or (
                h.agent_id and agent_counts[h.agent_id] != 1
            ):
                continue
            if not _review_candidate(h, events[spawn.event_idx]):
                continue
            # A child's private final is insufficient: require a parent-side
            # completed return/notification, with no timestamp inference.
            returns = [
                events[e.event_idx]
                for e in h.events
                if e.kind == "final"
                and e.session_id == session.session_id
                and e.event_idx in events
                and e.event_idx > spawn.event_idx
                and (not (result_id := events[e.event_idx].call_id) or result_counts[result_id] == 1)
            ]
            launches.append((h, events[spawn.event_idx], min(returns, key=lambda e: e.idx) if returns else None))
        launches.sort(key=lambda v: v[1].idx)
        for current, call, _ in launches:
            matches = [
                (prior, pcall, ret)
                for prior, pcall, ret in launches
                if ret is not None
                and pcall.idx < ret.idx < call.idx
                and scopes[pcall.idx] == scopes[ret.idx] == scopes[call.idx]
                and set(prior.request_files) & set(current.request_files)
            ]
            if not matches:
                continue
            prior, pcall, ret = max(matches, key=lambda v: v[2].idx)
            add(
                "delegation_repetition",
                session.session_id,
                [e for _, p, r in matches for e in (p, r)] + [call],
                len(matches),
                "Repeated delegation over review targets",
                "Compare review scope and artifact changes; carry forward prior findings and their disposition.",
                {
                    "request_event_idx": scopes[call.idx],
                    "compared_prior_handover_id": prior.id,
                    "new_handover_id": current.id,
                    "prior_handover_ids": [p.id for p, _, _ in matches],
                    "prior_spawn_idx": pcall.idx,
                    "prior_return_idx": ret.idx,
                    "new_spawn_idx": call.idx,
                    "shared_request_files": sorted(set(prior.request_files) & set(current.request_files)),
                    "artifact_version_known": False,
                },
            )

        calls: defaultdict[str, list[Event]] = defaultdict(list)
        results: defaultdict[str, list[Event]] = defaultdict(list)
        for ev in session.events:
            if ev.call_id and ev.kind == EventKind.tool_call:
                calls[ev.call_id].append(ev)
            elif ev.call_id and ev.kind == EventKind.tool_result:
                results[ev.call_id].append(ev)
        checks: defaultdict[tuple[int, str], list[tuple[Event, Event]]] = defaultdict(list)
        for cid, seq in calls.items():
            if len(seq) != 1 or len(results[cid]) != 1:
                continue
            call, result = seq[0], results[cid][0]
            cmd = call.tool_args.get("command") or call.tool_args.get("cmd")
            if (
                not isinstance(cmd, str)
                or not _check_command(cmd)
                or result.ok is not True
                or result.exit_code not in {None, 0}
                or result.idx <= call.idx
                or scopes[call.idx] != scopes[result.idx]
            ):
                continue
            if result.exit_code is None and re.search(
                r"\b(?:process|script|agent) (?:is (?:still )?)?running\b|\brunning in (?:the )?background\b",
                result.output or "",
                re.I,
            ):
                continue
            signature = json.dumps([call.tool_name, call.tool_args], sort_keys=True, default=str)
            checks[(scopes[call.idx], signature)].append((call, result))
        for (scope, _), seq in checks.items():
            seq.sort(key=lambda pair: pair[0].idx)
            if len(seq) < 2:
                continue
            durations = []
            for call, result in seq[1:]:
                if call.ts and result.ts:
                    try:
                        delta = (result.ts - call.ts).total_seconds()
                    except TypeError:
                        continue
                    if delta >= 0:
                        durations.append(delta)
            add(
                "verification_repetition",
                session.session_id,
                [e for pair in seq for e in pair],
                len(seq) - 1,
                "Repeated check commands after non-error tool returns",
                "Check whether inputs changed or repeated validation was requested before proposing check reuse.",
                {
                    "request_event_idx": scope,
                    "invocations": len(seq),
                    "repeat_round_trip_seconds": sum(durations),
                    "timed_repeats": len(durations),
                    "exact_arguments": True,
                    "explicit_zero_exit_invocations": sum(r.exit_code == 0 for _, r in seq),
                    "unknown_exit_invocations": sum(r.exit_code is None for _, r in seq),
                    "tool_success_is_not_check_success": True,
                    "exit_status_is_not_test_coverage": True,
                },
            )
    return rows
