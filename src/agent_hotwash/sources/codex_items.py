"""Record-level mappers for the native Codex 0.150+ surface.

Turns one ``item_completed`` item (CommandExecution, FileChange, McpToolCall,
...) or inter-agent ``agent_message`` into canonical :class:`Event` objects.
Pure functions; no session state lives here — see ``codex_native._decode_v2``
for turn/usage bookkeeping.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from typing import Any
from urllib.parse import unquote, urlparse

from agent_hotwash.events import (
    ArtifactInteraction,
    ArtifactOp,
    Event,
    EventKind,
    ParsedCommand,
    RoleHint,
    SourceRef,
)
from agent_hotwash.primitives.commands import classify_command, split_segments
from agent_hotwash.sources._common import (
    flatten_text,
    parse_ts,
    tool_category_of,
    tool_category_of_op,
    truncate,
    truncate_head_tail,
)

_EXIT_RE = re.compile(r"(?:Process |Command )?exited with code\s+(\d+)")
_INNER_EXIT_RE = re.compile(r'"exit_code"\s*:\s*(\d+)')
_TRUNC_WARN_RE = re.compile(r"truncated output \(original token count:\s*(\d+)\)", re.IGNORECASE)
_TRUNC_JSON_RE = re.compile(r'"original_token_count"\s*:\s*(\d+)')
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
_DELEGATION_TAG = "<codex_delegation>"
_INJECTED_TAGS = ("<recommended_plugins>", "<environment_context>", "<app-context>", "<skills_instructions>")
# A ``compacted`` record and its ``ContextCompaction`` item are written a few
# records apart; within this many events they are the same compaction.
_AGENT_FN = {
    "spawn_agent": "agent.spawn",
    "create_thread": "agent.spawn",
    "send_message": "agent.message",
    "followup_task": "agent.message",
    "interrupt_agent": "agent.message",
    "wait_agent": "agent.wait",
    "wait": "agent.wait",
    "list_agents": "agent.message",
}

# parsed_cmd.type → op_kind. unknown falls through to classify_command.
_PARSED_TO_OP = {
    "read": "cmd.read",
    "search": "cmd.search",
    "list_files": "cmd.list",
}

# Primary op_kind precedence when a CommandExecution has mixed parsed_cmd types.
_OP_PRECEDENCE = ("file.edit", "cmd.exec", "cmd.search", "cmd.read", "cmd.list", "other")


# ---------------------------------------------------------------------------
# small parsers
# ---------------------------------------------------------------------------


def _exit_from_output(output: str) -> int | None:
    m = _EXIT_RE.search(output)
    if m:
        return int(m.group(1))
    m = _INNER_EXIT_RE.search(output)
    return int(m.group(1)) if m else None


def _patch_paths(patch: str) -> list[str]:
    return [m.group(1).strip() for m in _PATCH_FILE_RE.finditer(patch)]


def _original_tokens(text: str) -> int | None:
    m = _TRUNC_WARN_RE.search(text) or _TRUNC_JSON_RE.search(text)
    return int(m.group(1)) if m else None


def _decode_cwd(raw: Any) -> str | None:
    """``CommandExecution.cwd`` is a percent-encoded ``file://`` URL in 0.15x;
    ``session_meta.cwd`` is a plain path. Return a plain path either way."""
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("file://"):
        return unquote(urlparse(raw).path) or None
    return raw


def _resolve_path(path: str, cwd: str | None) -> str:
    """Make ``path`` absolute against ``cwd`` so reads (relative in
    ``parsed_cmd``) and edits (absolute in ``FileChange``) share one key."""
    if not path or path.startswith("~"):
        return path
    if posixpath.isabs(path):
        return posixpath.normpath(path)
    if cwd:
        return posixpath.normpath(posixpath.join(cwd, path))
    return posixpath.normpath(path)


# Segment heads whose positional args name files the agent is *reading*.
_READ_HEADS = {"cat", "sed", "head", "tail", "nl", "bat", "less", "more", "wc", "stat", "file"}
# Segment heads whose positional args name paths the agent is *searching/listing*.
_SEARCH_HEADS = {"rg", "grep", "egrep", "fgrep", "ugrep", "find", "fd", "ls", "tree"}
# First positional arg is a pattern/script, not a path.
_PATTERN_FIRST = {"rg", "grep", "egrep", "fgrep", "ugrep", "sed"}


def _looks_like_path(tok: str) -> bool:
    tok = tok.strip("'\"")
    if not tok or len(tok) > 300 or tok in ("--", "-") or tok.startswith(("!", "%", ":!", "$")):
        return False
    return ("/" in tok or "." in tok) and " " not in tok and "\n" not in tok


def _artifacts_from_compound(cmd: str, cwd: str | None) -> list[ArtifactInteraction]:
    """Best-effort read/search artifacts for a ``parsed_cmd.type == "unknown"``
    compound line (``sed -n '1,80p' a.py && rg foo src``). Codex leaves ~40% of
    real commands unparsed, which otherwise hides every read they perform."""
    out: list[ArtifactInteraction] = []
    seen: set[tuple[str, ArtifactOp]] = set()
    for seg in split_segments(cmd):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        if not toks:
            continue
        head = toks[0].rsplit("/", 1)[-1]
        if head in _READ_HEADS:
            op = ArtifactOp.read
        elif head in _SEARCH_HEADS:
            op = ArtifactOp.search
        else:
            continue
        args = [t for t in toks[1:] if not t.startswith("-")]
        if head in _PATTERN_FIRST and args:
            args = args[1:]
        if head == "find":
            args = args[:1]
        for a in args:
            if not _looks_like_path(a):
                continue
            key = (_resolve_path(a.strip("'\""), cwd), op)
            if key in seen:
                continue
            seen.add(key)
            out.append(ArtifactInteraction(path=key[0], op=op))
    return out


def _role_hint(text: str | None) -> RoleHint:
    if not text:
        return RoleHint.user
    if _DELEGATION_TAG in text:
        return RoleHint.delegation
    stripped = text.lstrip()
    for tag in _INJECTED_TAGS:
        if stripped.startswith(tag):
            return RoleHint.injected
    return RoleHint.user


def _source_thread_id(text: str) -> str | None:
    start = text.find("<source_thread_id>")
    if start < 0:
        return None
    start += len("<source_thread_id>")
    end = text.find("</source_thread_id>", start)
    if end < 0:
        return None
    value = text[start:end].strip()
    return value or None


def _flatten_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return flatten_text(value)


_MSG_TYPE_RE = re.compile(r"^Message Type:\s*([A-Z_]+)", re.MULTILINE)


def _inter_agent_message(
    payload: dict[str, Any], ts: Any, src: SourceRef, group_id: str | None, own_path: str | None
) -> tuple[Event, bool]:
    """Decode a ``response_item/agent_message`` exchanged between threads.

    These carry ``author``/``recipient`` agent paths and a ``Message Type:``
    header (NEW_TASK, MESSAGE, FINAL_ANSWER, ...). A message *from our parent*
    (or any NEW_TASK addressed to us) is this thread's task input → ``user_msg``
    with ``role_hint=delegation``; returns ``(event, True)``. Anything else is a
    child reporting back → ``tool_result`` on ``agent.message``.
    """
    text = _flatten_output(payload.get("content")) or payload.get("text") or ""
    author = str(payload.get("author") or "")
    recipient = str(payload.get("recipient") or "")
    own = own_path or "/root"
    m = _MSG_TYPE_RE.search(text)
    msg_type = m.group(1) if m else None
    from_parent = bool(author) and own.startswith(author.rstrip("/") + "/")
    is_task = recipient == own and (from_parent or msg_type in ("NEW_TASK", "FOLLOWUP_TASK"))
    if is_task:
        ev = Event(
            kind=EventKind.user_msg,
            text=text or None,
            role_hint=RoleHint.delegation,
            tool_args={"author": author, "recipient": recipient, "message_type": msg_type},
            ts=ts,
            source=src,
            group_id=group_id,
            raw_type="agent_message",
        )
        return ev, True
    ev = Event(
        kind=EventKind.tool_result,
        tool_name="agent.message",
        op_kind="agent.message",
        ok=True,
        output=truncate_head_tail(text),
        tool_args={"author": author, "recipient": recipient, "message_type": msg_type},
        ts=ts,
        source=src,
        group_id=group_id,
        raw_type="agent_message",
    )
    return ev, False


def _count_diff_lines(diff: str) -> tuple[int, int]:
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _classify_unknown_cmd(cmd: str) -> str:
    intent = classify_command(cmd)
    head = cmd.strip().split(None, 1)[0].rsplit("/", 1)[-1] if cmd.strip() else ""
    if intent == "inspect":
        if head in {"rg", "grep", "egrep", "ugrep", "find"}:
            return "cmd.search"
        if head in {"ls", "tree", "find"}:
            return "cmd.list"
        return "cmd.read"
    return "cmd.exec"


def _primary_op_kind(classifications: list[ParsedCommand], command: str) -> str:
    kinds: list[str] = []
    for c in classifications:
        if c.type in _PARSED_TO_OP:
            kinds.append(_PARSED_TO_OP[c.type])
        else:
            kinds.append(_classify_unknown_cmd(c.cmd or command))
    if not kinds:
        return _classify_unknown_cmd(command) if command else "cmd.exec"
    # majority, ties broken by precedence
    counts: dict[str, int] = {}
    for k in kinds:
        counts[k] = counts.get(k, 0) + 1
    best = max(counts.values())
    candidates = [k for k, n in counts.items() if n == best]
    for pref in _OP_PRECEDENCE:
        if pref in candidates:
            return pref
    return candidates[0]


def _parse_args(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("arguments") or payload.get("input")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return args if isinstance(args, dict) else {}


# ---------------------------------------------------------------------------
# v2 item_completed-first path
# ---------------------------------------------------------------------------


def _command_text(item: dict[str, Any]) -> str:
    cmd = item.get("command")
    if isinstance(cmd, list):
        # drop leading shell binary; keep the -lc script if present
        if len(cmd) >= 3 and cmd[1] in ("-lc", "-c"):
            return str(cmd[2])
        return " ".join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) else ""


_FAILED_STATUSES = {"failed", "aborted", "cancelled", "canceled", "killed", "timed_out", "declined"}


def _exec_outcome(item: dict[str, Any], stdout: str) -> tuple[int | None, bool | None]:
    """``(exit_code, ok)`` for a CommandExecution item.

    ``exit_code`` may be an int or a numeric string (0.153 writes ``"0"``); when
    absent it is recovered from the wrapper's ``exited with code N`` text. A
    ``failed``/``aborted`` status is never ``ok``; ``completed`` without any exit
    evidence is treated as exit 0; anything else (in-progress, unknown) is
    ``ok=None``.
    """
    raw = item.get("exit_code")
    exit_code: int | None = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        exit_code = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        exit_code = int(raw.strip())
    else:
        exit_code = _exit_from_output(stdout)
    status = item.get("status") if isinstance(item.get("status"), str) else None
    if status in _FAILED_STATUSES:
        return exit_code, False
    if exit_code is None and status == "completed":
        exit_code = 0
    if exit_code is not None:
        return exit_code, exit_code == 0
    return None, None


def _item_command_execution(
    item: dict[str, Any], ts: Any, src: SourceRef, turn_id: str | None, group_id: str | None
) -> list[Event]:
    command = _command_text(item)
    cwd = _decode_cwd(item.get("cwd"))
    parsed = item.get("parsed_cmd") or []
    classifications = [
        ParsedCommand(type=str(p.get("type") or "unknown"), cmd=p.get("cmd"), path=p.get("path"))
        for p in parsed
        if isinstance(p, dict)
    ]
    op_kind = _primary_op_kind(classifications, command)
    artifacts: list[ArtifactInteraction] = []
    for c in classifications:
        if c.path:
            artifacts.append(
                ArtifactInteraction(
                    path=_resolve_path(c.path, cwd),
                    op=ArtifactOp.search if c.type == "search" else ArtifactOp.read,
                )
            )
        elif c.type == "unknown" and c.cmd:
            artifacts.extend(_artifacts_from_compound(c.cmd, cwd))
    if not classifications and command:
        artifacts.extend(_artifacts_from_compound(command, cwd))
    call_id = item.get("id")
    stdout = item.get("aggregated_output") or item.get("stdout") or ""
    if not isinstance(stdout, str):
        stdout = _flatten_output(stdout)
    exit_code, ok = _exec_outcome(item, stdout)
    ts_end = parse_ts(item.get("completed_at_ms")) if item.get("completed_at_ms") else ts
    orig = _original_tokens(stdout)
    call = Event(
        kind=EventKind.tool_call,
        tool_name=op_kind,
        tool_category=tool_category_of_op(op_kind),
        op_kind=op_kind,
        classifications=classifications,
        artifacts=artifacts,
        path=artifacts[0].path if artifacts else None,
        tool_args={"command": command, "cmd": command, "cwd": item.get("cwd")},
        call_id=call_id,
        ts=ts,
        ts_end=ts_end,
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="CommandExecution",
    )
    result = Event(
        kind=EventKind.tool_result,
        tool_name=op_kind,
        op_kind=op_kind,
        call_id=call_id,
        ts=ts_end or ts,
        ts_end=ts_end,
        ok=ok,
        exit_code=exit_code,
        output=truncate_head_tail(stdout),
        error_text=truncate_head_tail(stdout) if ok is False else None,
        output_tokens_original=orig,
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="CommandExecution",
    )
    return [call, result]


def _file_change_op_kind(artifacts: list[ArtifactInteraction]) -> str:
    """``file.write`` when every change creates a file, ``file.delete`` when every
    change removes one, else ``file.edit`` (update/move/mixed). Creating a file
    is not an edit of unread content, so detectors must not treat it as one."""
    ops = {a.op for a in artifacts}
    if ops and ops <= {ArtifactOp.add}:
        return "file.write"
    if ops and ops <= {ArtifactOp.delete}:
        return "file.delete"
    return "file.edit"


def _item_file_change(
    item: dict[str, Any], ts: Any, src: SourceRef, turn_id: str | None, group_id: str | None
) -> list[Event]:
    changes = item.get("changes") or {}
    artifacts: list[ArtifactInteraction] = []
    if isinstance(changes, dict):
        for path, spec in changes.items():
            if not isinstance(spec, dict):
                continue
            ctype = str(spec.get("type") or "update")
            if spec.get("move_path"):
                op = ArtifactOp.move
            elif ctype in ArtifactOp.__members__:
                op = ArtifactOp(ctype)
            else:
                op = ArtifactOp.update
            content = spec.get("content")
            diff = spec.get("unified_diff")
            added = removed = None
            diff_head = None
            if isinstance(diff, str) and diff:
                added, removed = _count_diff_lines(diff)
                diff_head = truncate(diff, 1200)
            elif isinstance(content, str):
                added = len(content.splitlines())
            artifacts.append(
                ArtifactInteraction(
                    path=str(path), op=op, lines_added=added, lines_removed=removed, diff_head=diff_head
                )
            )
    call_id = item.get("id")
    stdout = item.get("stdout") or ""
    op_kind = _file_change_op_kind(artifacts)
    added_vals = [a.lines_added for a in artifacts if a.lines_added is not None]
    removed_vals = [a.lines_removed for a in artifacts if a.lines_removed is not None]
    call = Event(
        kind=EventKind.tool_call,
        tool_name=op_kind,
        tool_category=tool_category_of_op(op_kind),
        op_kind=op_kind,
        artifacts=artifacts,
        path=artifacts[0].path if artifacts else None,
        tool_args={"paths": [a.path for a in artifacts], "ops": [a.op.value for a in artifacts]} if artifacts else {},
        lines_added=sum(added_vals) if added_vals else None,
        lines_removed=sum(removed_vals) if removed_vals else None,
        call_id=call_id,
        ts=ts,
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="FileChange",
    )
    result = Event(
        kind=EventKind.tool_result,
        tool_name=op_kind,
        op_kind=op_kind,
        call_id=call_id,
        ts=ts,
        ok=True,
        exit_code=0,
        output=truncate_head_tail(stdout if isinstance(stdout, str) else _flatten_output(stdout)),
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="FileChange",
    )
    return [call, result]


def _item_mcp(item: dict[str, Any], ts: Any, src: SourceRef, turn_id: str | None, group_id: str | None) -> list[Event]:
    server = item.get("server") or "unknown"
    tool = item.get("tool") or "unknown"
    op_kind = f"mcp.{server}.{tool}"
    result = item.get("result") or {}
    is_error = bool(result.get("isError")) if isinstance(result, dict) else False
    content = result.get("content") if isinstance(result, dict) else None
    text = _flatten_output(content) if content is not None else ""
    args = item.get("arguments")
    if isinstance(args, str):
        try:
            parsed_args = json.loads(args)
        except json.JSONDecodeError:
            parsed_args = {"raw": args}
    else:
        parsed_args = args if isinstance(args, dict) else {}
    call_id = item.get("id")
    cat = tool_category_of_op(op_kind)
    if item.get("readOnlyHint") is True:
        cat = tool_category_of("read")
    call = Event(
        kind=EventKind.tool_call,
        tool_name=op_kind,
        tool_category=cat,
        op_kind=op_kind,
        tool_args=parsed_args if isinstance(parsed_args, dict) else {"value": parsed_args},
        call_id=call_id,
        ts=ts,
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="McpToolCall",
    )
    res = Event(
        kind=EventKind.tool_result,
        tool_name=op_kind,
        op_kind=op_kind,
        call_id=call_id,
        ts=ts,
        ok=not is_error,
        output=truncate_head_tail(text),
        error_text=truncate_head_tail(text) if is_error else None,
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="McpToolCall",
    )
    return [call, res]


def _item_extension(
    item: dict[str, Any], ts: Any, src: SourceRef, turn_id: str | None, group_id: str | None
) -> list[Event]:
    kind = item.get("kind") or ""
    action = item.get("action") or {}
    atype = action.get("type") if isinstance(action, dict) else None
    op_kind = "web.open" if "open" in str(kind) or atype == "open" else "web.search"
    call_id = item.get("id")
    results = item.get("results") or []
    call = Event(
        kind=EventKind.tool_call,
        tool_name=op_kind,
        tool_category=tool_category_of_op(op_kind),
        op_kind=op_kind,
        tool_args={"query": item.get("query"), "action": action},
        call_id=call_id,
        ts=ts,
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="Extension",
    )
    res = Event(
        kind=EventKind.tool_result,
        tool_name=op_kind,
        op_kind=op_kind,
        call_id=call_id,
        ts=ts,
        ok=True,
        output=truncate_head_tail(json.dumps(results) if results else ""),
        source=src,
        turn_id=turn_id,
        group_id=group_id,
        raw_type="Extension",
    )
    return [call, res]


def _item_to_events(
    item: dict[str, Any],
    ts: Any,
    src: SourceRef,
    turn_id: str | None,
    group_id: str | None,
) -> list[Event]:
    itype = item.get("type")
    if itype == "CommandExecution":
        return _item_command_execution(item, ts, src, turn_id, group_id)
    if itype == "FileChange":
        return _item_file_change(item, ts, src, turn_id, group_id)
    if itype == "McpToolCall":
        return _item_mcp(item, ts, src, turn_id, group_id)
    if itype == "Extension":
        return _item_extension(item, ts, src, turn_id, group_id)
    if itype == "UserMessage":
        text = flatten_text(item.get("content"))
        hint = _role_hint(text)
        return [
            Event(
                kind=EventKind.user_msg,
                text=text or None,
                role_hint=hint,
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="UserMessage",
            )
        ]
    if itype == "AgentMessage":
        text = flatten_text(item.get("content"))
        phase = item.get("phase")
        return [
            Event(
                kind=EventKind.assistant_msg,
                text=text or None,
                phase=phase,
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="AgentMessage",
            )
        ]
    if itype == "Reasoning":
        return [
            Event(
                kind=EventKind.thinking,
                text=None,
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="Reasoning",
            )
        ]
    if itype == "FunctionCallOutput":
        name = item.get("name") or ""
        namespace = item.get("namespace") or ""
        op = _AGENT_FN.get(name, f"mcp.{namespace}.{name}" if namespace else "other")
        output = _flatten_output(item.get("output"))
        return [
            Event(
                kind=EventKind.tool_result,
                tool_name=op,
                op_kind=op if op in _AGENT_FN.values() or op.startswith("mcp.") else "other",
                tool_args={"name": name, "namespace": namespace},
                output=truncate_head_tail(output),
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="FunctionCallOutput",
            )
        ]
    if itype == "SubAgentActivity":
        # Lifecycle notification (started/interacted/...) about a child thread —
        # not an action the model took, so never a tool_call.
        return [
            Event(
                kind=EventKind.meta,
                tool_name="agent.activity",
                op_kind="agent.activity",
                tool_args={
                    "kind": item.get("kind"),
                    "agent_thread_id": item.get("agent_thread_id"),
                    "agent_path": item.get("agent_path"),
                },
                call_id=item.get("id"),
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="SubAgentActivity",
            )
        ]
    if itype == "CollabAgentToolCall":
        tool = item.get("tool") or "wait"
        op = _AGENT_FN.get(tool, "agent.wait")
        return [
            Event(
                kind=EventKind.tool_call,
                tool_name=op,
                tool_category=tool_category_of_op(op),
                op_kind=op,
                tool_args={
                    "tool": tool,
                    "sender_thread_id": item.get("sender_thread_id"),
                    "receiver_thread_ids": item.get("receiver_thread_ids"),
                },
                call_id=item.get("id"),
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="CollabAgentToolCall",
            )
        ]
    if itype == "ImageView":
        path = item.get("path")
        return [
            Event(
                kind=EventKind.tool_call,
                tool_name="image.view",
                tool_category=tool_category_of_op("image.view"),
                op_kind="image.view",
                path=path if isinstance(path, str) else None,
                call_id=item.get("id"),
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="ImageView",
            )
        ]
    if itype == "ContextCompaction":
        return [
            Event(
                kind=EventKind.compaction,
                ts=ts,
                source=src,
                turn_id=turn_id,
                group_id=group_id,
                raw_type="ContextCompaction",
            )
        ]
    return []


def _session_meta_fields(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source")
    spawn = None
    if isinstance(source, dict):
        spawn = (source.get("subagent") or {}).get("thread_spawn") or {}
    return {
        "id": payload.get("id"),
        "session_id_field": payload.get("session_id"),
        "cli_version": payload.get("cli_version"),
        "thread_source": payload.get("thread_source"),
        "parent_thread_id": payload.get("parent_thread_id"),
        "forked_from_id": payload.get("forked_from_id"),
        "forked_from_ordinal_exclusive": payload.get("forked_from_ordinal_exclusive"),
        "history_base": payload.get("history_base"),
        "subagent_history_start_ordinal": payload.get("subagent_history_start_ordinal"),
        "agent_nickname": payload.get("agent_nickname") or (spawn or {}).get("agent_nickname"),
        "agent_path": (spawn or {}).get("agent_path"),
        "depth": (spawn or {}).get("depth"),
        "cwd": payload.get("cwd"),
        "git": payload.get("git"),
        "originator": payload.get("originator"),
        "model_provider": payload.get("model_provider"),
    }
