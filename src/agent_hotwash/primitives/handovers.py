"""Evidence-preserving delegation ledger built from normalized sessions."""

from __future__ import annotations

import hashlib
import posixpath
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent_hotwash.events import Event, EventKind, Session, SourceRef, Trace
from agent_hotwash.semantic.redact import redact_state

_PATH = re.compile(r"(?<!\w)(?:~?/)?(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+\b")
_ID = re.compile(r"Agent(?: ID)?:\s*([\w-]+)")
_EXCERPT_SECRETS = [
    r"\bapikey_[A-Za-z0-9_]{20,}\b",
    r"\b(?:sk|ghp|gho|glpat)-?[A-Za-z0-9_-]{20,}\b",
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}",
]


class HandoverEvent(BaseModel):
    kind: str
    session_id: str
    event_idx: int
    source: SourceRef | None = None
    ts: datetime | None = None
    excerpt: str | None = None


class HandoverRecord(BaseModel):
    id: str
    trace_id: str
    parent_id: str
    agent_id: str | None = None  # provider-issued identity, may differ from session ID
    child_id: str | None = None
    link_candidates: list[str] = Field(default_factory=list)
    link_confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    link_evidence: list[str] = Field(default_factory=list)
    depth: int | None = None
    source_format: str
    role: str | None = None
    requested_model: str | None = None
    observed_model: str | None = None
    observed_model_revisions: list[str] = Field(default_factory=list)
    nickname: str | None = None
    request_visibility: Literal["plaintext", "encrypted", "missing", "truncated"] = "missing"
    reply_visibility: Literal["plaintext", "encrypted", "missing", "truncated"] = "missing"
    request_excerpt: str | None = None
    reply_excerpt: str | None = None
    request_chars: int | None = None
    request_bytes: int | None = None
    reply_chars: int | None = None
    reply_bytes: int | None = None
    reply_original_chars: int | None = None
    request_to_reply_chars: float | None = None
    request_files: list[str] = Field(default_factory=list)
    reply_files: list[str] = Field(default_factory=list)
    child_files_read: list[str] = Field(default_factory=list)
    child_files_written: list[str] = Field(default_factory=list)
    parent_files_read_after: list[str] = Field(default_factory=list)
    child_transcript_present: bool = False
    usage_estimated: bool = False
    child_tokens: dict[str, int | None] | None = None
    status: str = "unknown"
    primary_intervention: str = "investigate_visibility"
    events: list[HandoverEvent] = Field(default_factory=list)
    continuation_count: int = 0
    steer_count: int = 0
    failure_ids: list[str] = Field(default_factory=list)
    spawn_to_final_seconds: float | None = None
    final_to_consumption_seconds: float | None = None
    return_after_parent_final: bool | None = None
    late_return_overlap: bool | None = None
    possible_overlap: bool = False


class OrphanedReturn(BaseModel):
    child_id: str
    parent_id: str | None = None
    source: SourceRef | None = None
    reason: str = "child final observed without a matched launch"


def _id(trace_id: str, parent_id: str, ev: Event) -> str:
    src = ev.source
    parts = (
        trace_id,
        parent_id,
        str(src.record_index if src else ev.idx),
        str(src.ordinal if src else ""),
        ev.call_id or "",
    )
    key = "\x1f".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _text(row: HandoverRecord, value: str | None, *, request: bool) -> None:
    if not value:
        return
    prefix = "request" if request else "reply"
    encrypted = "<encrypted" in value.lower() or (value.startswith("gAAAA") and len(value) > 80)
    setattr(row, f"{prefix}_visibility", "encrypted" if encrypted else "plaintext")
    if getattr(row, f"{prefix}_visibility") == "encrypted":
        setattr(row, f"{prefix}_chars", None)
        setattr(row, f"{prefix}_bytes", None)
        return
    setattr(row, f"{prefix}_excerpt", redact_state(value[:600], _EXCERPT_SECRETS))
    setattr(row, f"{prefix}_chars", len(value))
    setattr(row, f"{prefix}_bytes", len(value.encode("utf-8")))
    setattr(row, f"{prefix}_files", sorted(set(_PATH.findall(value))))


def _event(kind: str, session: Session, ev: Event, text: str | None = None) -> HandoverEvent:
    return HandoverEvent(
        kind=kind,
        session_id=session.session_id,
        event_idx=ev.idx,
        source=ev.source,
        ts=ev.ts,
        excerpt=redact_state(text[:300], _EXCERPT_SECRETS) if text else None,
    )


def _arg(args: dict, *keys: str) -> str | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _result_id(result: Event | None, call: Event) -> str | None:
    receivers = call.tool_args.get("receiver_thread_ids")
    if isinstance(receivers, list) and len(receivers) == 1 and isinstance(receivers[0], str):
        return receivers[0]
    if result is None:
        return None
    direct = _arg(result.tool_args, "agentId", "agent_id", "threadId", "thread_id")
    if direct:
        return direct
    match = _ID.search(result.output or "")
    return match.group(1) if match else None


def _final(row: HandoverRecord, session: Session, ev: Event, text: str) -> None:
    _text(row, text, request=False)
    if ev.output_truncated:
        row.reply_visibility = "truncated"
        row.reply_original_chars = ev.output_chars_original
        row.reply_chars = None
        row.reply_bytes = None
        row.request_to_reply_chars = None
    if row.request_chars and row.reply_chars is not None:
        row.request_to_reply_chars = row.reply_chars / row.request_chars
    row.events.append(_event("final", session, ev, text))
    row.status = "completed"


def build_handovers(trace: Trace) -> list[HandoverRecord]:
    sessions = {s.session_id: s for s in [trace.root, *trace.subagents]}
    links = {link.child_id: link for link in trace.links}
    depths = {trace.root.session_id: 0}
    for _ in range(len(sessions)):
        for link in trace.links:
            if link.parent_id in depths:
                depths[link.child_id] = depths[link.parent_id] + 1
    rows: list[HandoverRecord] = []
    by_child: dict[str, HandoverRecord] = {}
    by_agent: dict[str, HandoverRecord] = {}
    for parent in sessions.values():
        results = {ev.call_id: ev for ev in parent.events if ev.kind is EventKind.tool_result and ev.call_id}
        for call in parent.events:
            if call.kind is not EventKind.tool_call:
                continue
            args = call.tool_args
            tool = (call.tool_name or "").lower()
            if tool == "agent" and args.get("resume"):
                target = str(args["resume"])
                row = by_agent.get(target) or by_child.get(target)
                if row:
                    row.continuation_count += 1
                    row.events.append(_event("resume", parent, call, _arg(args, "prompt")))
                continue
            if tool in {"steer_subagent", "agent.message"} or call.op_kind == "agent.message":
                target = _arg(args, "agent_id", "agentId", "target")
                row = by_agent.get(target or "") or by_child.get(target or "")
                if row:
                    row.steer_count += 1
                    row.events.append(_event("steer", parent, call, _arg(args, "message", "prompt")))
                continue
            if tool in {"get_subagent_result", "agent.wait"} or call.op_kind == "agent.wait":
                target = _arg(args, "agent_id", "agentId")
                row = by_agent.get(target or "") or by_child.get(target or "")
                if row:
                    row.events.append(_event("wait", parent, call))
                    result = results.get(call.call_id)
                    if result and "Status: completed" in (result.output or ""):
                        _final(row, parent, result, result.output or "")
                        row.events.append(_event("consumption", parent, result))
                continue
            if tool not in {"agent", "agent.spawn", "spawn_agent", "create_thread"} and call.op_kind != "agent.spawn":
                continue
            result = results.get(call.call_id)
            agent_id = _result_id(result, call)
            child_id = agent_id
            # A Codex rollout's session_meta is the identity authority. Match only
            # a full ID from the launch result; never use a nickname or short ID.
            if child_id and child_id not in links and child_id not in sessions:
                child_id = None
            row = HandoverRecord(
                id=_id(trace.trace_id, parent.session_id, call),
                trace_id=trace.trace_id,
                parent_id=parent.session_id,
                agent_id=agent_id,
                child_id=child_id,
                source_format=trace.provenance.source_format,
                depth=depths.get(parent.session_id, 0) + 1,
                role=_arg(args, "subagent_type", "agent_type"),
                requested_model=_arg(args, "model"),
                nickname=_arg(args, "agent_nickname"),
            )
            row.events.append(_event("spawn", parent, call))
            _text(row, _arg(args, "prompt", "message"), request=True)
            if result:
                row.events.append(_event("acknowledgement", parent, result))
                row.observed_model = _arg(result.tool_args, "modelName")
                if result.ok is False:
                    row.status = "spawn-rejected"
                    row.primary_intervention = "repair_child_execution"
                elif "background" in (result.output or "").lower():
                    row.status = "running-at-capture"
                elif result.output:
                    _final(row, parent, result, result.output)
                    row.events.append(_event("consumption", parent, result))
            if child_id:
                link = links.get(child_id)
                if link:
                    row.link_confidence = link.confidence
                    row.link_evidence = link.evidence
                    row.depth = link.depth
                by_child[child_id] = row
            if agent_id:
                by_agent[agent_id] = row
            rows.append(row)

    for parent in sessions.values():
        for ev in parent.events:
            if ev.raw_type not in {"subagents:record", "subagent-notification"}:
                continue
            agent_id = _arg(ev.tool_args, "id")
            row = by_agent.get(agent_id or "")
            if row is None:
                continue
            row.events.append(_event("notification", parent, ev))
            status = _arg(ev.tool_args, "status")
            result = _arg(ev.tool_args, "result")
            if status == "completed" and result and row.reply_visibility in {"missing", "truncated"}:
                _final(row, parent, ev, result)
            elif status == "completed" and row.status != "completed":
                row.status = "result-unavailable"
            elif status in {"terminated", "failed", "error"} and row.status != "completed":
                row.status = "terminated" if status == "terminated" else "failed"

    # Pi child session_info contains only a short display prefix. Expose
    # candidate paths, but never promote this to an identity join.
    for row in rows:
        if row.child_id is not None or not row.agent_id or trace.agent.value != "pi":
            continue
        prefix = row.agent_id.replace("-", "")[:8].lower()
        row.link_candidates = [
            session.session_id
            for session in trace.subagents
            if any(
                ev.raw_type == "session_info" and ev.text and ev.text.rsplit("#", 1)[-1].lower() == prefix
                for ev in session.events
            )
        ]
        if row.link_candidates:
            row.link_evidence = ["session_info_prefix_only; no exact identity join"]

    # Recover Codex joins from authoritative child session IDs when the tool
    # response omitted the ID and the parent has only one unmatched spawn.
    for child_id, link in links.items():
        if link.kind.value != "spawn" or child_id in by_child:
            continue
        candidates = [r for r in rows if r.parent_id == link.parent_id and r.child_id is None]
        if len(candidates) == 1:
            row = candidates[0]
            row.child_id = child_id
            row.link_confidence = "medium"
            row.link_evidence = [*link.evidence, "unique_unmatched_spawn"]
            row.depth = link.depth
            by_child[child_id] = row

    for child_id, row in by_child.items():
        child = sessions.get(child_id)
        if child is None:
            continue
        row.child_transcript_present = "usage_estimated" not in child.degraded
        row.usage_estimated = not child.usage_reliable
        row.observed_model = child.model or row.observed_model
        row.observed_model_revisions = list(
            dict.fromkeys(turn.model_config_active.model for turn in child.turns if turn.model_config_active.model)
        )
        if child.usage_reliable:
            usages = [event.usage for event in child.events if event.usage is not None]
            if usages:
                row.child_tokens = {
                    name: sum(value for usage in usages if (value := getattr(usage, name)) is not None)
                    if any(getattr(usage, name) is not None for usage in usages)
                    else None
                    for name in ("input", "output", "cache_read", "cache_write")
                }
        if row.child_transcript_present:
            for ev in child.events:
                if ev.kind is EventKind.tool_call:
                    for artifact in ev.artifacts:
                        bucket = (
                            row.child_files_read if artifact.op.value in {"read", "search"} else row.child_files_written
                        )
                        if artifact.path not in bucket:
                            bucket.append(artifact.path)
                if (
                    ev.kind is EventKind.assistant_msg
                    and (ev.phase == "final_answer" or ev is child.events[-1])
                    and ev.text
                ):
                    _final(row, child, ev, ev.text)
    # Parent-side lifecycle evidence does not require an exact child join.
    for row in rows:
        spawn = next((e for e in row.events if e.kind == "spawn"), None)
        final = next((e for e in row.events if e.kind == "final"), None)
        if spawn and final and spawn.ts and final.ts and final.ts >= spawn.ts:
            row.spawn_to_final_seconds = (final.ts - spawn.ts).total_seconds()
        consumption = next((event for event in row.events if event.kind == "consumption"), None)
        if final and consumption and final.ts and consumption.ts and consumption.ts >= final.ts:
            row.final_to_consumption_seconds = (consumption.ts - final.ts).total_seconds()
        if final and final.ts:
            next_action_recorded = False
            for ev in sessions[row.parent_id].events:
                if ev.ts and ev.ts > final.ts:
                    if ev.kind is EventKind.tool_call:
                        for artifact in ev.artifacts:
                            if (
                                artifact.op.value in {"read", "search"}
                                and artifact.path in row.reply_files
                                and artifact.path not in row.parent_files_read_after
                            ):
                                row.parent_files_read_after.append(artifact.path)
                    if not next_action_recorded:
                        row.events.append(_event("parent_next_action", sessions[row.parent_id], ev))
                        next_action_recorded = True
        if final and final.ts:
            parent_final = next(
                (
                    ev
                    for ev in reversed(sessions[row.parent_id].events)
                    if ev.kind is EventKind.assistant_msg and ev.phase == "final_answer" and ev.ts
                ),
                None,
            )
            if parent_final and parent_final.ts:
                row.return_after_parent_final = final.ts > parent_final.ts
        child_writes = {posixpath.normpath(path) for path in row.child_files_written}
        if child_writes and final:
            overlapping = []
            for ev in sessions[row.parent_id].events:
                if ev.kind is not EventKind.tool_call:
                    continue
                writes = {
                    posixpath.normpath(artifact.path)
                    for artifact in ev.artifacts
                    if artifact.op.value in {"add", "update", "delete", "move"}
                }
                if child_writes & writes:
                    overlapping.append(ev)
            if overlapping:
                if final.ts and all(ev.ts for ev in overlapping):
                    row.late_return_overlap = any(ev.ts >= final.ts for ev in overlapping if ev.ts)
                else:
                    row.possible_overlap = True
        row.primary_intervention = (
            "investigate_visibility" if row.status != "spawn-rejected" else row.primary_intervention
        )
    return rows


def orphaned_returns(trace: Trace, handovers: list[HandoverRecord]) -> list[OrphanedReturn]:
    joined = {row.child_id for row in handovers if row.child_id}
    out = []
    for child in trace.subagents:
        if child.session_id in joined or "usage_estimated" in child.degraded:
            continue
        final = next((ev for ev in reversed(child.events) if ev.kind is EventKind.assistant_msg and ev.text), None)
        if final:
            possible = any(
                row.parent_id == child.parent_session_id and child.session_id in row.link_candidates
                for row in handovers
            )
            out.append(
                OrphanedReturn(
                    child_id=child.session_id,
                    parent_id=child.parent_session_id,
                    source=final.source,
                    reason="display-prefix candidate; exact launch join unavailable"
                    if possible
                    else "child final observed without a matched launch",
                )
            )
    return out


__all__ = ["HandoverEvent", "HandoverRecord", "OrphanedReturn", "build_handovers", "orphaned_returns"]
