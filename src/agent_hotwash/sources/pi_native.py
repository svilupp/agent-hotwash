"""Native pi session parser (format D).

Layout: ``~/.pi/agent/sessions/<project-slug>/<ISO-ts>_<uuid>.jsonl``, where the
project slug is a mangled cwd (e.g. ``--Users-jan-Documents-GitHub--``) and each
file is one session. Records are flat ``{type, ...}`` objects (no ``payload``
envelope, unlike codex; no ``uuid``/``parentUuid`` DAG, unlike claude):

- ``session`` — first record; ``id`` is the session id, plus ``version``/``cwd``.
  Persisted pi-subagents children also carry ``parentSession`` (absolute path
  of the spawner).
- ``model_change`` — ``provider`` + ``modelId`` (the active model).
- ``session_info`` — display name, including ``{type}#{agent-id-prefix}`` on
  persisted subagent sessions.
- ``thinking_level_change`` — ignored (meta).
- ``message`` — wraps a ``message`` object keyed by ``role``.
- ``custom`` / ``custom_message`` — pi-subagents history. ``subagents:record``
  and ``subagent-notification`` name spawned agents and, when the child file
  was never persisted, a lump ``totalTokens`` used for estimated spend.

A project dir is grouped into parent→child trees (like Codex). Children whose
parent is not in the input become roots with ``thread_linkage=partial``.
Missing child transcripts are reconstructed as estimated subagent sessions.

This is a close cousin of the code-bench pi stdout stream; the usage shape is
identical, so ``_pi_usage`` is reused from :mod:`agent_hotwash.sources.codebench`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_hotwash.canonical import build_turns, declared_row, observe_capabilities
from agent_hotwash.events import (
    AgentKind,
    Capabilities,
    CapLevel,
    Event,
    EventKind,
    Provenance,
    Session,
    ThreadLink,
    ThreadLinkKind,
    Trace,
    Usage,
)
from agent_hotwash.sources._common import (
    build_session,
    flatten_text,
    iter_jsonl,
    parse_ts,
    truncate,
)
from agent_hotwash.sources.codebench import _pi_usage

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import datetime

_DECLARED = declared_row(
    per_call_usage=True,
    per_turn_model=True,
    reasoning_effort=False,
    reasoning_text=True,
    reasoning_tokens=False,
    timestamps=True,
    op_timing=False,
    parsed_commands=False,
    file_diffs=False,
    full_tool_output=CapLevel.partial,
    output_size_original=False,
    thread_linkage=True,
    compaction_summaries=False,
    final_answer_marker=False,
    context_window=False,
)

# Notification ``totalTokens`` is input+output+cache with no mix. Persisted
# Pi children are cache-heavy; this split is estimated, not exact.
_EST_CACHE_READ = 0.80
_EST_OUTPUT = 0.08


def decode_pi_native(records: list[dict[str, Any]]) -> tuple[list[Event], str | None, str | None]:
    """Decode native pi session records into (events, session_id, model)."""
    events: list[Event] = []
    session_id: str | None = None
    model: str | None = None

    for r in records:
        rtype = r.get("type")
        if rtype == "session":
            session_id = r.get("id") or session_id
        elif rtype == "model_change":
            model = r.get("modelId") or model
            events.append(
                Event(
                    kind=EventKind.meta,
                    ts=parse_ts(r.get("timestamp")),
                    raw_type="model_change",
                    tool_args={"model": r.get("modelId"), "provider": r.get("provider")},
                )
            )
        elif rtype == "session_info":
            events.append(
                Event(
                    kind=EventKind.meta,
                    ts=parse_ts(r.get("timestamp")),
                    text=r.get("name") if isinstance(r.get("name"), str) else None,
                    raw_type="session_info",
                    tool_args={"name": r.get("name")},
                )
            )
        elif rtype == "message":
            msg = r.get("message")
            if isinstance(msg, dict):
                events.extend(_message(msg))
                model = model or msg.get("model")
    return events, session_id, model


def _message(msg: dict[str, Any]) -> list[Event]:
    role = msg.get("role")
    ts = parse_ts(msg.get("timestamp"))
    if role == "user":
        return [Event(kind=EventKind.user_msg, text=flatten_text(msg.get("content")) or None, ts=ts, raw_type="user")]
    if role == "toolResult":
        content = flatten_text(msg.get("content"))
        is_error = bool(msg.get("isError"))
        details = _as_dict(msg.get("details"))
        return [
            Event(
                kind=EventKind.tool_result,
                call_id=msg.get("toolCallId"),
                tool_name=msg.get("toolName"),
                ts=ts,
                ok=not is_error,
                output=truncate(content),
                error_text=truncate(content) if is_error else None,
                raw_type="toolResult",
                tool_args={k: details[k] for k in ("agentId", "modelName", "subagentType") if k in details},
            )
        ]
    if role == "assistant":
        return _assistant(msg, ts)
    return []


def _assistant(msg: dict[str, Any], ts: Any) -> list[Event]:
    """Split an assistant message's content blocks into thinking / tool_call /
    assistant_msg events. Per-response usage is attached to the first event so
    summation bills the response once."""
    usage = _pi_usage(msg.get("usage"))
    events: list[Event] = []
    for block in msg.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            events.append(
                Event(kind=EventKind.thinking, text=block.get("thinking") or None, ts=ts, raw_type="thinking")
            )
        elif btype == "toolCall":
            events.append(
                Event(
                    kind=EventKind.tool_call,
                    tool_name=block.get("name"),
                    call_id=block.get("id"),
                    tool_args=block.get("arguments") or {},
                    ts=ts,
                    raw_type="toolCall",
                )
            )
        elif btype == "text":
            events.append(Event(kind=EventKind.assistant_msg, text=block.get("text") or None, ts=ts, raw_type="text"))
    if usage is not None and events:
        events[0].usage = usage
    return events


# ---------------------------------------------------------------------------
# Subagent estimates (when the child session file was never persisted)
# ---------------------------------------------------------------------------


@dataclass
class _Estimate:
    agent_id: str
    agent_type: str | None = None
    description: str | None = None
    status: str | None = None
    total_tokens: int | None = None
    tool_uses: int | None = None
    model: str | None = None
    ts: datetime | None = None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _agent_key(agent_id: str) -> str:
    """Stable short key: ``aa966699-88f4-4ba`` and ``advisor#aa966699`` share it."""
    text = agent_id.strip()
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    return text.replace("-", "").lower()[:8]


def _estimates_from_records(records: Sequence[dict[str, Any]]) -> dict[str, _Estimate]:
    """Last-wins map of pi-subagents agent id → notification/record fields."""
    out: dict[str, _Estimate] = {}

    def bucket(agent_id: str) -> _Estimate:
        row = out.get(agent_id)
        if row is None:
            row = _Estimate(agent_id=agent_id)
            out[agent_id] = row
        return row

    for rec in records:
        rtype = rec.get("type")
        ts = parse_ts(rec.get("timestamp"))
        if rtype == "custom" and rec.get("customType") == "subagents:record":
            data = _as_dict(rec.get("data"))
            agent_id = data.get("id")
            if not isinstance(agent_id, str) or not agent_id.strip():
                continue
            row = bucket(agent_id.strip())
            if isinstance(data.get("type"), str):
                row.agent_type = data["type"]
            if isinstance(data.get("description"), str):
                row.description = data["description"]
            if isinstance(data.get("status"), str):
                row.status = data["status"]
            row.ts = ts or row.ts
        elif rtype == "custom_message" and rec.get("customType") == "subagent-notification":
            details = _as_dict(rec.get("details"))
            agent_id = details.get("id")
            if not isinstance(agent_id, str) or not agent_id.strip():
                continue
            row = bucket(agent_id.strip())
            tokens = _as_int(details.get("totalTokens"))
            if tokens is not None and (row.total_tokens is None or tokens > row.total_tokens):
                row.total_tokens = tokens
            uses = _as_int(details.get("toolUses"))
            if uses is not None:
                row.tool_uses = uses
            if isinstance(details.get("status"), str):
                row.status = details["status"]
            if isinstance(details.get("description"), str) and not row.description:
                row.description = details["description"]
            row.ts = ts or row.ts
        elif rtype == "message":
            msg = rec.get("message")
            if not isinstance(msg, dict) or msg.get("role") != "toolResult":
                continue
            details = _as_dict(msg.get("details"))
            agent_id = details.get("agentId")
            if not isinstance(agent_id, str) or not agent_id.strip():
                continue
            row = bucket(agent_id.strip())
            if isinstance(details.get("subagentType"), str) and not row.agent_type:
                row.agent_type = details["subagentType"]
            if isinstance(details.get("modelName"), str) and not row.model:
                row.model = details["modelName"]
            if isinstance(details.get("description"), str) and not row.description:
                row.description = details["description"]
    return out


def estimate_usage_from_total(total: int) -> Usage:
    """Split a lump ``totalTokens`` into billing fields (estimated mix)."""
    total = max(0, int(total))
    if total == 0:
        return Usage(input=0, output=0, cache_read=0, cache_write=0)
    cache_read = round(total * _EST_CACHE_READ)
    output = round(total * _EST_OUTPUT)
    input_tokens = total - cache_read - output
    if input_tokens < 0:
        input_tokens = 0
        cache_read = total - output
    return Usage(input=input_tokens, output=output, cache_read=cache_read, cache_write=0)


def _covered_agent_keys(sessions: Sequence[Session]) -> set[str]:
    """Keys already represented by a persisted child (``session_info`` name)."""
    keys: set[str] = set()
    for session in sessions:
        keys.add(_agent_key(session.session_id))
        for ev in session.events:
            if ev.raw_type == "session_info" and ev.text:
                keys.add(_agent_key(ev.text))
    return keys


def _synthetic_session(estimate: _Estimate, *, parent_id: str, model: str | None) -> Session:
    usage = estimate_usage_from_total(estimate.total_tokens or 0)
    events = [
        Event(
            kind=EventKind.meta,
            ts=estimate.ts,
            text=estimate.description,
            usage=usage,
            raw_type="subagent_estimate",
            tool_args={
                "agent_id": estimate.agent_id,
                "agent_type": estimate.agent_type,
                "status": estimate.status,
                "tool_uses": estimate.tool_uses,
                "total_tokens": estimate.total_tokens,
            },
        )
    ]
    session = build_session(
        events,
        AgentKind.pi,
        session_id=estimate.agent_id,
        model=model,
        parent_session_id=parent_id,
        usage_reliable=False,
    )
    session.turns = build_turns(session)
    session.capabilities = observe_capabilities(session, _DECLARED)
    session.degraded.append("usage_estimated")
    session.thread_source = "pi_subagent_estimate"
    return session


def _finish_session(
    events: list[Event],
    *,
    session_id: str,
    model: str | None,
    parent_session_id: str | None = None,
) -> Session:
    session = build_session(
        events,
        AgentKind.pi,
        session_id=session_id,
        model=model,
        parent_session_id=parent_session_id,
    )
    session.turns = build_turns(session)
    session.capabilities = observe_capabilities(session, _DECLARED)
    return session


def _parse_file(path: Path) -> tuple[Session, list[dict[str, Any]], str | None]:
    records = list(iter_jsonl(path))
    events, session_id, model = decode_pi_native(records)
    session_id = session_id or path.stem
    header = records[0] if records else {}
    parent_raw = header.get("parentSession") if isinstance(header, dict) else None
    parent_path = _resolve_parent(path, parent_raw)
    parent_id: str | None = None
    if parent_path is not None:
        parent_id = parent_path.stem.split("_")[-1] if "_" in parent_path.stem else parent_path.stem
    session = _finish_session(events, session_id=session_id, model=model, parent_session_id=parent_id)
    return session, records, None if parent_path is None else str(parent_path)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def _resolve_parent(child: Path, parent_raw: Any) -> Path | None:
    if not isinstance(parent_raw, str) or not parent_raw.strip():
        return None
    raw = Path(parent_raw.strip()).expanduser()
    candidate = raw if raw.is_absolute() else (child.parent / raw)
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def index_session_header(path: Path) -> dict[str, Any] | None:
    """Read the ``session`` header (cheap). ``parent_path`` is resolved or None."""
    path = path.expanduser()
    for rec in iter_jsonl(path):
        if rec.get("type") != "session":
            return None
        parent_path = _resolve_parent(path, rec.get("parentSession"))
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        return {
            "id": rec.get("id") or path.stem,
            "path": resolved,
            "parent_path": parent_path,
        }
    return None


def components(paths: Sequence[Path]) -> list[tuple[Path, list[Path], str | None]]:
    """Group session files into (root, descendants, parent_missing)."""
    index: list[dict[str, Any]] = []
    by_path: dict[Path, dict[str, Any]] = {}
    for path in paths:
        meta = index_session_header(path)
        if meta is None:
            continue
        index.append(meta)
        by_path[meta["path"]] = meta

    children_of: dict[Path, list[Path]] = {}
    attached: set[Path] = set()
    for meta in index:
        parent = meta["parent_path"]
        if parent in by_path and parent != meta["path"]:
            children_of.setdefault(parent, []).append(meta["path"])
            attached.add(meta["path"])

    out: list[tuple[Path, list[Path], str | None]] = []
    for meta in index:
        if meta["path"] in attached:
            continue
        descendants = _walk_descendants(meta["path"], children_of)
        missing = None
        parent = meta["parent_path"]
        if parent is not None and parent not in by_path:
            missing = str(parent)
        out.append((meta["path"], descendants, missing))
    if index and not out:
        # Cycle: every file named a parent in the input. Keep each as a root.
        for meta in index:
            out.append((meta["path"], [], None))
    return out


def _walk_descendants(root: Path, children_of: dict[Path, list[Path]]) -> list[Path]:
    out: list[Path] = []
    stack = list(children_of.get(root, []))
    seen: set[Path] = {root}
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        stack.extend(children_of.get(node, []))
    return sorted(out)


def _attach_estimates(
    root: Session,
    file_children: list[Session],
    records: Sequence[dict[str, Any]],
) -> tuple[list[Session], list[ThreadLink], list[str]]:
    estimates = _estimates_from_records(records)
    covered = _covered_agent_keys(file_children)
    subagents = list(file_children)
    links: list[ThreadLink] = []
    notes: list[str] = []
    estimated_n = 0
    estimated_tokens = 0

    for child in file_children:
        parent_id = child.parent_session_id or root.session_id
        links.append(
            ThreadLink(
                child_id=child.session_id,
                parent_id=parent_id,
                kind=ThreadLinkKind.spawn,
                evidence=["parentSession"],
            )
        )

    for estimate in estimates.values():
        if _agent_key(estimate.agent_id) in covered:
            continue
        if estimate.total_tokens is None and not estimate.status and not estimate.agent_type:
            continue
        session = _synthetic_session(estimate, parent_id=root.session_id, model=root.model)
        subagents.append(session)
        evidence = ["subagent-notification"]
        if estimate.total_tokens is None:
            evidence.append("tokens_unknown")
        links.append(
            ThreadLink(
                child_id=session.session_id,
                parent_id=root.session_id,
                kind=ThreadLinkKind.spawn,
                evidence=evidence,
                confidence="medium" if estimate.total_tokens is not None else "low",
            )
        )
        estimated_n += 1
        estimated_tokens += estimate.total_tokens or 0
        covered.add(_agent_key(estimate.agent_id))

    if estimated_n:
        notes.append(
            f"{estimated_n} subagent session(s) estimated from pi-subagents notifications "
            f"({estimated_tokens} lump tokens; mix is estimated)"
        )
    return subagents, links, notes


def _build_trace(
    root_path: Path,
    root: Session,
    *,
    files: list[Path],
    subagents: list[Session],
    links: list[ThreadLink],
    notes: list[str],
    parent_missing: str | None,
) -> Trace:
    if parent_missing:
        notes = [f"parent not in input: {parent_missing}", *notes]
        root.degraded.append("thread_linkage")
    real_children = [s for s in subagents if "usage_estimated" not in s.degraded]
    caps = (
        Capabilities.merge_min([root.capabilities, *[s.capabilities for s in real_children]])
        if real_children
        else root.capabilities
    )
    linkage: str | None = None
    if parent_missing or any("usage_estimated" in s.degraded for s in subagents):
        linkage = "partial"
    elif subagents or links:
        linkage = "full"
    provenance = Provenance(
        source_format="pi_native",
        detector_confidence="high",
        root_path=root_path,
        files=files,
        notes=notes,
        thread_linkage=linkage,
    )
    return Trace(
        trace_id=root.session_id,
        agent=AgentKind.pi,
        model=root.model,
        root=root,
        subagents=subagents,
        links=links,
        provenance=provenance,
        capabilities=caps,
    )


def load_session_file(path: Path) -> Trace:
    """Parse one native pi session file into a Trace (plus estimated children)."""
    path = path.expanduser()
    root, records, parent_missing = _parse_file(path)
    subagents, links, notes = _attach_estimates(root, [], records)
    return _build_trace(
        path,
        root,
        files=[path],
        subagents=subagents,
        links=links,
        notes=notes,
        parent_missing=parent_missing,
    )


def load_tree(root_path: Path, child_paths: Sequence[Path], *, parent_missing: str | None = None) -> Trace:
    """Load a parent session and its persisted child files as one Trace."""
    root_path = root_path.expanduser()
    root, records, header_missing = _parse_file(root_path)
    missing = parent_missing or header_missing
    file_children: list[Session] = []
    files = [root_path]
    for child_path in child_paths:
        child_path = child_path.expanduser()
        child, _child_records, _ = _parse_file(child_path)
        child.parent_session_id = root.session_id
        file_children.append(child)
        files.append(child_path)
    subagents, links, notes = _attach_estimates(root, file_children, records)
    return _build_trace(
        root_path,
        root,
        files=files,
        subagents=subagents,
        links=links,
        notes=notes,
        parent_missing=missing,
    )


def load_paths(paths: Sequence[Path]) -> Trace:
    """Load ``paths[0]`` as the root and any further paths as persisted children."""
    if not paths:
        raise ValueError("load_paths requires at least one session file")
    if len(paths) == 1:
        return load_session_file(paths[0])
    return load_tree(paths[0], paths[1:])


def iter_sessions_tree(root: Path) -> Iterator[Trace]:
    """Yield one Trace per parent→child tree under a sessions tree/project dir."""
    files = [p for p in sorted(root.rglob("*.jsonl")) if looks_like_native_pi(p)]
    for root_path, children, missing in components(files):
        yield load_tree(root_path, children, parent_missing=missing)


def looks_like_native_pi(path: Path) -> bool:
    """True when the first record is a native-pi ``session`` header.

    The signature is a flat ``{type:"session"}`` record carrying ``version`` and
    ``cwd`` and no ``payload`` envelope — this distinguishes it from a codex
    ``session_meta`` (which nests a ``payload``) and a claude session file (whose
    records carry ``uuid``/``parentUuid`` and different ``type`` values).
    """
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    for r in iter_jsonl(path):
        return r.get("type") == "session" and "version" in r and "cwd" in r and "payload" not in r
    return False
