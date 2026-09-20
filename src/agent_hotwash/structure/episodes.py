"""Atomic, immutable episodes at model-call granularity (§5.3)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.canonical import events_in_range
from agent_hotwash.config import Config
from agent_hotwash.events import Event, EventKind, ModelCall, Session, Turn, Usage
from agent_hotwash.primitives.errors import classify_impediment
from agent_hotwash.structure.facts import EDIT_OPS, READ_OPS, episode_facts, is_test_like
from agent_hotwash.structure.tasks import Task

READ_SEARCH_OPS = READ_OPS
SPAWN_WAIT_OPS = frozenset({"agent.spawn", "agent.wait"})

Trigger = Literal[
    "user_request",
    "delegated_request",
    "plan_step",
    "observed_failure",
    "prior_result",
    "recovery",
    "harness",
    "other",
]
Termination = Literal[
    "user_turn",
    "final_answer",
    "compaction",
    "spawn_or_wait",
    "phase_shift_edit",
    "phase_shift_test",
    "idle_gap",
    "cap_split",
    "end_of_file",
]


class Episode(BaseModel):
    """One atomic model-call run. Immutable for spend/provenance; phase is a label.

    Spend is carried as *tokens* (``usage``). Dollars are never stored on the
    atom: every monetary figure must name its cost view and pricing status
    (§7.1), so episode money lives in the cost view — ``PHASE_SPEND`` rows and
    ``CostViews.per_response[].episode_id`` in ``diagnostics.cost_views``.
    """

    model_config = ConfigDict(extra="ignore")

    episode_id: str
    task_id: str
    turn_id: str
    response_ids: list[str | None] = Field(default_factory=list)
    event_start: int = 0
    event_end: int = 0
    ops: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    duration_s: float | None = None
    trigger: Trigger = "other"
    termination: Termination = "end_of_file"
    facts: dict[str, Any] = Field(default_factory=dict)
    phase_activity: str | None = None
    phase_purpose: str | None = None
    parent_task: str | None = None
    spawn_op: dict[str, Any] | None = None
    # Every child thread id this atom is evidenced to have started (spawn args,
    # receiver ids, or the thread id echoed by a ``create_thread`` result).
    spawned_thread_ids: list[str] = Field(default_factory=list)


def add_usage(left: Usage, right: Usage | None) -> Usage:
    """Sum two usage rows. ``None`` fields stay ``None`` until a value is seen."""
    if right is None:
        return left

    def _add(a: int | None, b: int | None) -> int | None:
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)

    return Usage(
        input=_add(left.input, right.input),
        output=_add(left.output, right.output),
        cache_read=_add(left.cache_read, right.cache_read),
        cache_write=_add(left.cache_write, right.cache_write),
        reasoning_output=_add(left.reasoning_output, right.reasoning_output),
        cumulative=False,
    )


def sum_usage(usages: list[Usage | None]) -> Usage:
    acc = Usage()
    for u in usages:
        acc = add_usage(acc, u)
    return acc


def _call_events(session: Session, call: ModelCall) -> list[Event]:
    return events_in_range(session, call.event_start, call.event_end)


def _op_kinds(events: list[Event]) -> list[str]:
    return [ev.op_kind for ev in events if ev.kind is EventKind.tool_call and ev.op_kind]


def _is_read_search_only(ops: list[str], events: list[Event]) -> bool:
    if any(op in EDIT_OPS or op in SPAWN_WAIT_OPS or op == "cmd.exec" for op in ops):
        return False
    if any(is_test_like(ev) for ev in events):
        return False
    meaningful = [op for op in ops if op in READ_SEARCH_OPS]
    return bool(meaningful) and all(op in READ_SEARCH_OPS for op in ops)


def _contains_edit(ops: list[str]) -> bool:
    return any(op in EDIT_OPS for op in ops)


def _contains_spawn_or_wait(ops: list[str]) -> bool:
    return any(op in SPAWN_WAIT_OPS for op in ops)


def _contains_compaction(events: list[Event]) -> bool:
    return any(ev.kind is EventKind.compaction for ev in events)


def _contains_final_answer(events: list[Event]) -> bool:
    return any(ev.kind is EventKind.assistant_msg and ev.phase == "final_answer" for ev in events)


def _contains_plan(ops: list[str]) -> bool:
    return "plan.update" in ops


def _contains_failure(events: list[Event]) -> bool:
    return any(ev.ok is False or (ev.exit_code not in (0, None) and ev.kind is EventKind.tool_result) for ev in events)


def _contains_impediment(events: list[Event]) -> bool:
    for ev in events:
        text = ev.error_text or ev.output or ev.text
        if classify_impediment(text):
            return True
    return False


def _idle_gap_minutes(prev: ModelCall, nxt: ModelCall) -> float | None:
    t0 = prev.ts_end or prev.ts_start
    t1 = nxt.ts_start or nxt.ts_end
    if t0 is None or t1 is None:
        return None
    return (t1 - t0).total_seconds() / 60.0


def _all_model_calls(session: Session) -> list[tuple[Turn, ModelCall]]:
    out: list[tuple[Turn, ModelCall]] = []
    for turn in session.turns:
        for call in turn.model_calls:
            out.append((turn, call))
    return out


def _task_for_turn(turn_id: str | None, tasks: list[Task], fallback: Task | None) -> Task | None:
    if turn_id:
        for task in tasks:
            if any(t.turn_id == turn_id for t in task.turns):
                return task
            if task.task_id.endswith(turn_id):
                return task
    return fallback


def _estimate_call_tokens(session: Session, call: ModelCall) -> int:
    chars = 0
    for ev in _call_events(session, call):
        chars += len(ev.output or "") + len(ev.text or "") + len(ev.error_text or "")
        for art in ev.artifacts:
            chars += len(art.diff_head or "") + len(art.path)
    return max(1, chars // 4)


def _duration_s(calls: list[ModelCall]) -> float | None:
    starts = [c.ts_start for c in calls if c.ts_start is not None]
    ends = [c.ts_end or c.ts_start for c in calls if (c.ts_end or c.ts_start) is not None]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds()  # type: ignore[operator]


def _artifacts_of(events: list[Event]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ev in events:
        for art in ev.artifacts:
            if art.path and art.path not in seen:
                seen.add(art.path)
                out.append(art.path)
        if ev.path and ev.path not in seen:
            seen.add(ev.path)
            out.append(ev.path)
    return out


def _spawn_op(events: list[Event]) -> dict[str, Any] | None:
    for ev in events:
        if ev.op_kind == "agent.spawn":
            return {"op_kind": ev.op_kind, "tool_args": dict(ev.tool_args or {})}
    return None


_THREAD_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _is_create_thread(ev: Event) -> bool:
    return ev.kind is EventKind.tool_call and (ev.op_kind or "").endswith(".create_thread")


def _spawned_thread_ids(events: list[Event]) -> list[str]:
    """Child thread ids started within ``events`` (ordered, de-duplicated).

    ``agent.spawn`` names the child in its args/receivers; an MCP
    ``create_thread`` only reveals it in the tool *result*, so the paired
    result (by ``call_id``) is scanned too. Harness ``agent.activity`` meta
    events (0.150 has no receiver ids on the spawn call) are the fallback:
    the first atom that mentions a child is the one that started it.
    """
    out: list[str] = []
    results = {ev.call_id: ev for ev in events if ev.kind is EventKind.tool_result and ev.call_id}
    for ev in events:
        blobs: list[str] = []
        if (ev.kind is EventKind.meta and ev.op_kind == "agent.activity") or (
            ev.kind is EventKind.tool_call and ev.op_kind == "agent.spawn"
        ):
            blobs.append(str(ev.tool_args or {}))
        elif _is_create_thread(ev):
            res = results.get(ev.call_id) if ev.call_id else None
            if res is not None:
                blobs.append(str(res.output or "") + str(res.text or ""))
        for blob in blobs:
            for tid in _THREAD_ID_RE.findall(blob):
                if tid not in out:
                    out.append(tid)
    return out


def _trigger_of(
    turn: Turn,
    *,
    first_in_turn: bool,
    prev_events: list[Event],
    prev_ops: list[str],
) -> Trigger:
    if first_in_turn:
        kind = turn.user_input.kind
        if kind == "delegation":
            return "delegated_request"
        if kind == "user":
            return "user_request"
        if kind == "injected":
            return "harness"
        return "other"
    if _contains_impediment(prev_events):
        return "recovery"
    if _contains_failure(prev_events):
        return "observed_failure"
    if _contains_plan(prev_ops):
        return "plan_step"
    if prev_events:
        return "prior_result"
    return "other"


def _termination_after(
    *,
    call: ModelCall,
    nxt: tuple[Turn, ModelCall] | None,
    ops: list[str],
    events: list[Event],
    consecutive_reads: int,
    pending_edit: bool,
    idle_gap: float,
    reads_before_edit: int,
) -> Termination | None:
    if nxt is not None and nxt[1].turn_id != call.turn_id:
        return "user_turn"
    if _contains_final_answer(events):
        return "final_answer"
    if _contains_compaction(events):
        return "compaction"
    if _contains_spawn_or_wait(ops):
        return "spawn_or_wait"
    if _contains_edit(ops) and consecutive_reads >= reads_before_edit:
        return "phase_shift_edit"
    if any(is_test_like(ev) for ev in events) and (pending_edit or _contains_edit(ops)):
        return "phase_shift_test"
    if nxt is not None:
        gap = _idle_gap_minutes(call, nxt[1])
        if gap is not None and gap > idle_gap:
            return "idle_gap"
    return None


def segment_episodes(session: Session, tasks: list[Task], config: Config) -> list[Episode]:
    """Place episode boundaries only between model calls. Every call is in one atom."""
    ep_cfg = config.structure.episodes
    pairs = _all_model_calls(session)
    if not pairs:
        return []

    fallback = tasks[0] if tasks else None
    episodes: list[Episode] = []
    current: list[tuple[Turn, ModelCall]] = []
    current_tokens = 0
    consecutive_reads = 0
    pending_edit = False
    prev_events: list[Event] = []
    prev_ops: list[str] = []

    def _emit(group: list[tuple[Turn, ModelCall]], termination: Termination, trigger: Trigger) -> None:
        nonlocal episodes
        if not group:
            return
        turn, first_call = group[0]
        last_call = group[-1][1]
        events = []
        for _t, c in group:
            events.extend(_call_events(session, c))
        ops = _op_kinds(events)
        usage = sum_usage([c.usage for _t, c in group])
        task = _task_for_turn(turn.turn_id, tasks, fallback)
        task_id = task.task_id if task else f"{session.session_id}:task0"
        parent_task = task.parent_task if task else None
        n = len(episodes)
        episode = Episode(
            episode_id=f"{session.session_id}:ep{n}",
            task_id=task_id,
            turn_id=turn.turn_id,
            response_ids=[c.response_id for _t, c in group],
            event_start=first_call.event_start,
            event_end=last_call.event_end,
            ops=ops,
            artifacts=_artifacts_of(events),
            usage=usage,
            duration_s=_duration_s([c for _t, c in group]),
            trigger=trigger,
            termination=termination,
            parent_task=parent_task,
            spawn_op=_spawn_op(events),
            spawned_thread_ids=_spawned_thread_ids(events),
        )
        episode.facts = episode_facts(session, episode, task)
        episodes.append(episode)

    i = 0
    while i < len(pairs):
        turn, call = pairs[i]
        nxt = pairs[i + 1] if i + 1 < len(pairs) else None
        events = _call_events(session, call)
        ops = _op_kinds(events)
        call_tokens = _estimate_call_tokens(session, call)

        would_cap = bool(current) and (
            len(current) + 1 > ep_cfg.max_model_calls or current_tokens + call_tokens > ep_cfg.state_token_budget
        )
        if would_cap:
            trig = _trigger_of(
                current[0][0],
                first_in_turn=True,
                prev_events=prev_events,
                prev_ops=prev_ops,
            )
            _emit(current, "cap_split", trig)
            prev_events = _call_events(session, current[-1][1])
            prev_ops = _op_kinds(prev_events)
            current = []
            current_tokens = 0

        first_in_turn = (not current) and (not episodes or episodes[-1].turn_id != turn.turn_id)
        if not current:
            trigger = _trigger_of(turn, first_in_turn=first_in_turn, prev_events=prev_events, prev_ops=prev_ops)
        current.append((turn, call))
        current_tokens += call_tokens

        term = _termination_after(
            call=call,
            nxt=nxt,
            ops=ops,
            events=events,
            consecutive_reads=consecutive_reads,
            pending_edit=pending_edit,
            idle_gap=ep_cfg.idle_gap_minutes,
            reads_before_edit=ep_cfg.reads_before_edit_boundary,
        )
        is_last = nxt is None
        if is_last and term is None:
            term = "end_of_file"
        elif (
            (not is_last)
            and term is None
            and (
                len(current) >= ep_cfg.max_model_calls
                or (current_tokens > ep_cfg.state_token_budget and len(current) > 1)
            )
        ):
            term = "cap_split"

        if _is_read_search_only(ops, events):
            consecutive_reads += 1
        elif ops or any(ev.kind is EventKind.tool_call for ev in events):
            consecutive_reads = 0
        if _contains_edit(ops):
            pending_edit = True
        if any(is_test_like(ev) for ev in events):
            pending_edit = False

        if term is not None:
            _emit(current, term, trigger)
            prev_events = events
            prev_ops = ops
            current = []
            current_tokens = 0
        i += 1

    by_task: dict[str, list[Episode]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task_id, []).append(ep)
    for group in by_task.values():
        n = len(group)
        for idx, ep in enumerate(group):
            ep.facts["position"] = f"{idx + 1} of {n}"
    return episodes


def group_display_runs(episodes: list[Episode], key: Callable[[Episode], object] | None = None) -> list[list[Episode]]:
    """Group contiguous atoms that share ``key(ep)`` (default ``phase_activity``).

    Atoms themselves are unchanged. Grouped-view usage sums equal atom sums.
    """
    keyf = key or (lambda ep: ep.phase_activity)
    groups: list[list[Episode]] = []
    last_key: object = None
    for ep in episodes:
        k = keyf(ep)
        if groups and k == last_key:
            groups[-1].append(ep)
        else:
            groups.append([ep])
        last_key = k
    return groups


def _spawn_targets_thread(event: Event, spawn_thread_id: str) -> bool:
    if event.op_kind != "agent.spawn":
        return False
    args = event.tool_args or {}
    if spawn_thread_id in {
        args.get("agent_thread_id"),
        args.get("thread_id"),
        args.get("threadId"),
        args.get("child_id"),
    }:
        return True
    receivers = args.get("receiver_thread_ids") or args.get("receiver_thread_id") or []
    if isinstance(receivers, str):
        receivers = [receivers]
    if spawn_thread_id in receivers:
        return True
    nested = args.get("result") or args.get("output")
    if isinstance(nested, dict) and spawn_thread_id in {nested.get("threadId"), nested.get("thread_id")}:
        return True
    return spawn_thread_id in str(args)


def attach_delegation(child_task: Task, parent_episodes: list[Episode], spawn_thread_id: str) -> Task:
    """Bind a delegated child to the parent episode that spawned its thread (WP3b)."""
    for ep in parent_episodes:
        if spawn_thread_id in ep.spawned_thread_ids or (ep.spawn_op and spawn_thread_id in str(ep.spawn_op)):
            child_task.parent_task = ep.task_id
            child_task.spawn_episode_id = ep.episode_id
            return child_task
    # Fall back: scan spawn_op tool_args only; callers may also pass session via facts.
    for ep in parent_episodes:
        args = (ep.spawn_op or {}).get("tool_args") or {}
        fake = Event(kind=EventKind.tool_call, op_kind="agent.spawn", tool_args=args)
        if _spawn_targets_thread(fake, spawn_thread_id):
            child_task.parent_task = ep.task_id
            child_task.spawn_episode_id = ep.episode_id
            return child_task
    return child_task


__all__ = [
    "Episode",
    "add_usage",
    "attach_delegation",
    "events_in_range",
    "group_display_runs",
    "segment_episodes",
    "sum_usage",
]
