"""Structure layer: tasks, atomic episodes, digests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from agent_hotwash.config import Config, load_config
from agent_hotwash.events import (
    AgentKind,
    ArtifactInteraction,
    ArtifactOp,
    Event,
    EventKind,
    ModelCall,
    ModelConfig,
    RoleHint,
    Session,
    Turn,
    TurnStatus,
    Usage,
    UserInput,
)
from agent_hotwash.structure import (
    build_digest,
    group_display_runs,
    segment_episodes,
    segment_tasks,
)
from agent_hotwash.structure.episodes import attach_delegation, sum_usage

_TS0 = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def _usage(inp: int, out: int) -> Usage:
    return Usage(input=inp, output=out, cache_read=0, cache_write=0, reasoning_output=0)


def _call(turn_id: str, rid: str, start: int, end: int, usage: Usage, minutes: int = 0) -> ModelCall:
    ts = _TS0 + timedelta(minutes=minutes)
    return ModelCall(
        response_id=rid,
        turn_id=turn_id,
        event_start=start,
        event_end=end,
        usage=usage,
        ts_start=ts,
        ts_end=ts + timedelta(seconds=5),
    )


def _turn(
    turn_id: str,
    session_id: str,
    *,
    kind: Literal["user", "delegation", "injected", "none"],
    text: str,
    start: int,
    end: int,
    calls: list[ModelCall],
    status: TurnStatus = TurnStatus.completed,
    final: str | None = None,
) -> Turn:
    return Turn(
        turn_id=turn_id,
        session_id=session_id,
        event_start=start,
        event_end=end,
        status=status,
        user_input=UserInput(text=text, kind=kind),
        model_config_active=ModelConfig(model="gpt-5.6-luna", reasoning_effort="high"),
        model_calls=calls,
        final_message=final,
        ts_start=_TS0,
        ts_end=_TS0 + timedelta(minutes=1),
    )


def _session(events: list[Event], turns: list[Turn], session_id: str = "s1") -> Session:
    for i, ev in enumerate(events):
        ev.idx = i
    return Session(session_id=session_id, agent=AgentKind.unknown, events=events, turns=turns, model="gpt-5.6-luna")


def _two_candidate_session() -> Session:
    u1 = _usage(10, 4)
    u2 = _usage(20, 6)
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="implement src/a.py", turn_id="t1", role_hint=RoleHint.user),
        Event(kind=EventKind.tool_call, idx=1, op_kind="cmd.read", path="src/a.py", turn_id="t1", response_id="r1"),
        Event(kind=EventKind.assistant_msg, idx=2, text="done", phase="final_answer", turn_id="t1", response_id="r1"),
        Event(kind=EventKind.user_msg, idx=3, text="now rewrite billing.py", turn_id="t2", role_hint=RoleHint.user),
        Event(kind=EventKind.tool_call, idx=4, op_kind="file.edit", path="billing.py", turn_id="t2", response_id="r2"),
        Event(kind=EventKind.assistant_msg, idx=5, text="ok", phase="final_answer", turn_id="t2", response_id="r2"),
    ]
    t1 = _turn(
        "t1", "s1", kind="user", text="implement src/a.py", start=0, end=2, calls=[_call("t1", "r1", 0, 2, u1, 0)]
    )
    t2 = _turn(
        "t2",
        "s1",
        kind="user",
        text="now rewrite billing.py",
        start=3,
        end=5,
        calls=[_call("t2", "r2", 3, 5, u2, 1)],
    )
    return _session(events, [t1, t2])


def test_digest_messages_are_structured_objects() -> None:
    cfg = load_config()
    session = _two_candidate_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    assert episodes
    digest = build_digest(tasks[0], episodes[0], session, cfg)
    assert digest["digest_schema_version"] == 3
    msgs = digest["episode"]["messages"]
    assert msgs
    assert all(isinstance(m, dict) and {"kind", "text"} <= set(m) for m in msgs)
    kinds = {m["kind"] for m in msgs}
    assert kinds <= {"user", "assistant", "thinking"}
    assert any(m["kind"] == "user" for m in msgs)
    assert any(m["kind"] == "assistant" and m.get("phase") == "final_answer" for m in msgs)
    assert isinstance(digest["episode"]["omitted_messages"], int)
    ops = digest["episode"]["ops"]
    assert ops
    assert all(isinstance(op, dict) and "kind" in op for op in ops)
    counts = digest["episode"]["counts"]
    assert counts["n_ops"] == len(ops)
    assert counts["majority_kind"] == "cmd.read"
    assert counts["majority_family"] == "read"
    assert counts["n_final_answer"] == 1
    assert counts["test_after_edit"] is False
    assert digest["episode"]["instruction"] == "implement src/a.py"


def test_digest_instruction_uses_later_user_message() -> None:
    cfg = load_config()
    session = _two_candidate_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    assert len(episodes) >= 2
    first = build_digest(tasks[0], episodes[0], session, cfg)
    last = build_digest(tasks[0], episodes[-1], session, cfg)
    assert first["episode"]["instruction"] == "implement src/a.py"
    assert last["episode"]["instruction"] == "now rewrite billing.py"


def test_empty_delegation_envelope_is_not_a_request() -> None:
    from agent_hotwash.structure.ledger import Ledger, normalize_request, update_ledger

    envelope = "Message Type: NEW_TASK\nTask name: /root/bottom_up_estimate\nSender: /root\nPayload:"
    assert normalize_request(envelope) == ""
    assert normalize_request(envelope + "\n  \n") == ""
    assert normalize_request("implement src/a.py") == "implement src/a.py"

    turn = _turn(
        "t1",
        "s1",
        kind="delegation",
        text=envelope,
        start=0,
        end=2,
        calls=[_call("t1", "r1", 0, 2, _usage(1, 1))],
    )
    events = [
        Event(kind=EventKind.user_msg, idx=0, text=envelope, turn_id="t1", role_hint=RoleHint.delegation),
        Event(kind=EventKind.assistant_msg, idx=1, text="estimate", phase="final_answer", turn_id="t1"),
    ]
    session = _session(events, [turn])
    ledger = Ledger()
    update_ledger(ledger, turn, session.events)
    assert ledger.request == ""
    cfg = load_config()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    digest = build_digest(tasks[0], episodes[0], session, cfg)
    assert digest["task"]["request"] == ""
    assert digest["episode"]["instruction"] == ""


def test_off_mode_one_task_per_session() -> None:
    cfg = load_config()
    session = _two_candidate_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    assert len(tasks) == 1
    assert {t.turn_id for t in tasks[0].turns} >= {"t1", "t2"}
    assert tasks[0].edge_to_prev is None


def test_recorded_answers_produce_expected_edges() -> None:
    cfg = load_config()
    session = _two_candidate_session()
    cases = [
        (
            {
                "identity": "same_deliverable",
                "corrects": 0.9,
                "references": 0.1,
                "same_component": 0.1,
            },
            "corrects",
        ),
        (
            {
                "identity": "same_deliverable",
                "corrects": 0.1,
                "references": 0.1,
                "same_component": 0.1,
            },
            "continues",
        ),
        (
            {
                "identity": "distinct_deliverable",
                "corrects": 0.1,
                "references": 0.8,
                "same_component": 0.1,
            },
            "depends_on",
        ),
        (
            {
                "identity": "distinct_deliverable",
                "corrects": 0.1,
                "references": 0.1,
                "same_component": 0.9,
            },
            "sibling_same_area",
        ),
        (
            {
                "identity": "distinct_deliverable",
                "corrects": 0.1,
                "references": 0.1,
                "same_component": 0.1,
            },
            "unrelated",
        ),
        (
            {
                "identity": "unclear",
                "corrects": 0.9,
                "references": 0.9,
                "same_component": 0.9,
            },
            "continues",
        ),
    ]
    for answers, edge in cases:
        tasks = segment_tasks(session, cfg, semantic_mode="cached", relationship_answers={"t2": answers})
        assert len(tasks) == 2, edge
        assert tasks[0].edge_to_prev is None
        assert tasks[1].edge_to_prev == edge
        if edge == "continues" and answers["identity"] == "unclear":
            assert tasks[1].edge_confidence == "low"
        else:
            assert tasks[1].edge_confidence in {"high", "low"}


def test_delegation_initiator_parent_agent() -> None:
    cfg = load_config()
    events = [Event(kind=EventKind.user_msg, idx=0, text="<codex_delegation>do x", turn_id="d1")]
    turn = _turn("d1", "s1", kind="delegation", text="<codex_delegation>do x", start=0, end=0, calls=[])
    session = _session(events, [turn])
    tasks = segment_tasks(session, cfg, semantic_mode="cached")
    assert len(tasks) == 1
    assert tasks[0].initiator == "parent_agent"
    assert tasks[0].parent_task is None


def _multi_call_session() -> Session:
    """Two user turns, compaction, spawn, three model calls with known usage."""
    u1 = _usage(11, 3)
    u2 = _usage(7, 2)
    u3 = _usage(5, 1)
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="fix src/a.py", turn_id="t1"),
        Event(kind=EventKind.tool_call, idx=1, op_kind="cmd.read", path="src/a.py", turn_id="t1", response_id="r1"),
        Event(kind=EventKind.compaction, idx=2, text="compacted", turn_id="t1", response_id="r1"),
        Event(kind=EventKind.user_msg, idx=3, text="continue", turn_id="t2"),
        Event(
            kind=EventKind.tool_call,
            idx=4,
            op_kind="agent.spawn",
            tool_args={"agent_thread_id": "child-1"},
            turn_id="t2",
            response_id="r2",
        ),
        Event(kind=EventKind.tool_call, idx=5, op_kind="cmd.exec", turn_id="t2", response_id="r3"),
        Event(
            kind=EventKind.tool_result,
            idx=6,
            op_kind="cmd.exec",
            ok=True,
            exit_code=0,
            output="ok",
            turn_id="t2",
            response_id="r3",
        ),
    ]
    t1 = _turn("t1", "s1", kind="user", text="fix src/a.py", start=0, end=2, calls=[_call("t1", "r1", 0, 2, u1, 0)])
    t2 = _turn(
        "t2",
        "s1",
        kind="user",
        text="continue",
        start=3,
        end=6,
        calls=[
            _call("t2", "r2", 3, 4, u2, 1),
            _call("t2", "r3", 5, 6, u3, 2),
        ],
    )
    return _session(events, [t1, t2])


def test_every_model_call_in_one_episode_and_usage_sums() -> None:
    cfg = load_config()
    session = _multi_call_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    calls = [c for t in session.turns for c in t.model_calls]
    owned = [rid for ep in episodes for rid in ep.response_ids]
    assert sorted(owned) == sorted(c.response_id for c in calls)
    assert len(owned) == len(calls)
    assert sum_usage([ep.usage for ep in episodes]) == sum_usage([c.usage for c in calls])


def test_no_atom_crosses_user_turn_compaction_or_spawn() -> None:
    cfg = load_config()
    session = _multi_call_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    # r1 has compaction → boundary after; r2 has spawn → boundary after; r3 is last.
    assert len(episodes) >= 3
    by_rid = {rid: ep for ep in episodes for rid in ep.response_ids}
    assert by_rid["r1"] is not by_rid["r2"]
    assert by_rid["r2"] is not by_rid["r3"]
    assert by_rid["r1"].turn_id == "t1"
    assert by_rid["r2"].turn_id == "t2"
    assert by_rid["r1"].termination in {"compaction", "user_turn"}
    assert by_rid["r2"].termination == "spawn_or_wait"


def test_digest_fits_budget() -> None:
    cfg = Config.model_validate({"structure": {"episodes": {"state_token_budget": 200}}})
    huge = "x" * 8000
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="implement src/a.py " + huge, turn_id="t1"),
        Event(
            kind=EventKind.tool_call,
            idx=1,
            op_kind="file.edit",
            path="src/a.py",
            artifacts=[ArtifactInteraction(path="src/a.py", op=ArtifactOp.update, diff_head="+" + huge)],
            turn_id="t1",
            response_id="r1",
        ),
        Event(
            kind=EventKind.tool_result,
            idx=2,
            op_kind="file.edit",
            output=huge,
            ok=True,
            exit_code=0,
            turn_id="t1",
            response_id="r1",
        ),
        Event(
            kind=EventKind.assistant_msg, idx=3, text="All checks passed. " + huge, phase="final_answer", turn_id="t1"
        ),
    ]
    turn = _turn(
        "t1",
        "s1",
        kind="user",
        text="implement src/a.py",
        start=0,
        end=3,
        calls=[_call("t1", "r1", 0, 3, _usage(1, 1))],
        final="All checks passed.",
    )
    session = _session(events, [turn])
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    assert episodes
    digest = build_digest(tasks[0], episodes[0], session, cfg)
    raw = json.dumps(digest, ensure_ascii=False, separators=(",", ":"), default=str)
    assert len(raw) // 4 <= cfg.structure.episodes.state_token_budget
    assert digest["digest_schema_version"] == 3
    assert "counts" in digest["episode"]


def test_group_display_runs_sums_equal_atoms() -> None:
    cfg = load_config()
    session = _multi_call_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    for ep in episodes:
        ep.phase_activity = "inspect"
    groups = group_display_runs(episodes)
    assert groups
    grouped_usage = sum_usage([ep.usage for g in groups for ep in g])
    assert grouped_usage == sum_usage([ep.usage for ep in episodes])
    # split activity: two runs
    if len(episodes) >= 2:
        episodes[0].phase_activity = "inspect"
        for ep in episodes[1:]:
            ep.phase_activity = "modify"
        groups2 = group_display_runs(episodes)
        assert len(groups2) >= 2
        assert sum_usage([ep.usage for g in groups2 for ep in g]) == sum_usage([ep.usage for ep in episodes])


_FIXTURE = Path(__file__).parent / "fixtures" / "codex_native" / "v0153" / "user_multiturn.jsonl"
_FACT_KEYS = {
    "failure_signature_repeats",
    "repeat_count",
    "distinct_paths_read",
    "paths_reread_unchanged",
    "edits_without_subsequent_test",
    "error_kinds",
    "env_impediment",
    "declares_success",
    "cites_verification",
    "verification",
    "artifact_overlap",
    "after_compaction",
    "position",
}


def test_real_fixture_episodes_carry_deterministic_facts() -> None:
    from agent_hotwash.sources.detect import iter_traces

    cfg = load_config()
    trace = next(iter_traces(_FIXTURE))
    session = trace.root
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    assert episodes
    for ep in episodes:
        assert set(ep.facts) >= _FACT_KEYS, ep.episode_id
        assert isinstance(ep.facts["repeat_count"], int)
        assert isinstance(ep.facts["artifact_overlap"], list)
    # Something non-trivial is actually observed on the real trace, not just zeros.
    assert any(ep.facts["distinct_paths_read"] > 0 for ep in episodes)
    assert any(ep.facts["artifact_change"] for ep in episodes)
    assert any(ep.facts["env_impediment"] for ep in episodes)
    assert any(ep.facts["error_kinds"] for ep in episodes)
    assert any(ep.facts["after_compaction"] for ep in episodes)  # fixture carries a compaction
    assert any(ep.facts["artifact_overlap"] for ep in episodes)
    with_edit = next(ep for ep in episodes if ep.facts["artifact_change"])
    assert with_edit.facts["edits"] == with_edit.facts["edits_without_subsequent_test"]


def test_digest_and_episode_facts_agree() -> None:
    from agent_hotwash.sources.detect import iter_traces
    from agent_hotwash.structure.facts import episode_facts

    cfg = load_config()
    trace = next(iter_traces(_FIXTURE))
    session = trace.root
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    task_by_id = {t.task_id: t for t in tasks}
    for ep in episodes:
        task = task_by_id[ep.task_id]
        fresh = episode_facts(session, ep, task)
        for key, val in fresh.items():
            assert ep.facts[key] == val, (ep.episode_id, key)
        digest = build_digest(task, ep, session, cfg)
        dfacts = digest["episode"]["facts"]
        if not dfacts:  # shrunk to the floor digest
            continue
        for key in _FACT_KEYS:
            assert dfacts[key] == ep.facts[key], (ep.episode_id, key)
        assert "op_outcomes" not in dfacts


def test_env_impediment_ignores_success_text() -> None:
    from agent_hotwash.structure.facts import episode_facts

    cfg = load_config()
    u1 = _usage(10, 4)
    mention = [
        Event(kind=EventKind.user_msg, idx=0, text="check health", turn_id="t1", role_hint=RoleHint.user),
        Event(kind=EventKind.tool_call, idx=1, op_kind="cmd.exec", turn_id="t1", response_id="r1"),
        Event(
            kind=EventKind.tool_result,
            idx=2,
            op_kind="cmd.exec",
            exit_code=0,
            ok=True,
            output="200 ok",
            turn_id="t1",
            response_id="r1",
        ),
        Event(
            kind=EventKind.assistant_msg,
            idx=3,
            text="Command timed out after 30s in the docs example; live call returned 200.",
            phase="final_answer",
            turn_id="t1",
            response_id="r1",
        ),
    ]
    t1 = _turn("t1", "s1", kind="user", text="check health", start=0, end=3, calls=[_call("t1", "r1", 0, 3, u1)])
    session = _session(mention, [t1])
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    facts = episode_facts(session, episodes[0], tasks[0])
    assert facts["env_impediment"] is None

    failed = [
        Event(kind=EventKind.user_msg, idx=0, text="check health", turn_id="t1", role_hint=RoleHint.user),
        Event(kind=EventKind.tool_call, idx=1, op_kind="cmd.exec", turn_id="t1", response_id="r1"),
        Event(
            kind=EventKind.tool_result,
            idx=2,
            op_kind="cmd.exec",
            exit_code=124,
            ok=False,
            output="Command timed out after 30s",
            turn_id="t1",
            response_id="r1",
        ),
        Event(kind=EventKind.assistant_msg, idx=3, text="retrying", turn_id="t1", response_id="r1"),
    ]
    session2 = _session(failed, [t1])
    tasks2 = segment_tasks(session2, cfg, semantic_mode="off")
    episodes2 = segment_episodes(session2, tasks2, cfg)
    facts2 = episode_facts(session2, episodes2[0], tasks2[0])
    assert facts2["env_impediment"] == "timeout"


def test_episode_has_no_bare_dollar_field() -> None:
    """§7.1: dollars are never stored untagged on the atom; spend is tokens only."""
    from agent_hotwash.structure.episodes import Episode

    assert "invoice_cost" not in Episode.model_fields
    assert not any(name.endswith("cost") for name in Episode.model_fields)
    assert Episode.model_fields["usage"].annotation is Usage


def test_group_display_runs_custom_key() -> None:
    cfg = load_config()
    session = _multi_call_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    assert len(episodes) >= 3
    resolved = {episodes[0].episode_id: "inspect", episodes[1].episode_id: "inspect"}
    for ep in episodes[2:]:
        resolved[ep.episode_id] = "run"
    groups = group_display_runs(episodes, key=lambda ep: resolved[ep.episode_id])
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert sum_usage([ep.usage for g in groups for ep in g]) == sum_usage([ep.usage for ep in episodes])


def test_attach_delegation_sets_parent_and_spawn_episode() -> None:
    cfg = load_config()
    session = _multi_call_session()
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    child = segment_tasks(session, cfg, semantic_mode="cached")[0]
    attach_delegation(child, episodes, "child-1")
    assert child.parent_task == tasks[0].task_id
    assert child.spawn_episode_id is not None
    spawn_ep = next(ep for ep in episodes if ep.episode_id == child.spawn_episode_id)
    assert "agent.spawn" in spawn_ep.ops


def _child_session(session_id: str) -> Session:
    """A delegated child: one delegation turn, one model call."""
    events = [
        Event(kind=EventKind.user_msg, idx=0, text="NEW_TASK: fix it", turn_id="c1", role_hint=RoleHint.delegation),
        Event(kind=EventKind.tool_call, idx=1, op_kind="cmd.exec", turn_id="c1", response_id="cr1"),
        Event(kind=EventKind.assistant_msg, idx=2, text="done", phase="final_answer", turn_id="c1"),
    ]
    turn = _turn(
        "c1",
        session_id,
        kind="delegation",
        text="NEW_TASK: fix it",
        start=0,
        end=2,
        calls=[_call("c1", "cr1", 0, 2, _usage(3, 1))],
    )
    return _session(events, [turn], session_id=session_id)


def test_annotate_trace_segments_every_session_and_attaches_delegation() -> None:
    """Children get tasks/episodes too, and a spawned child's task binds to the
    parent episode that spawned it (PLAN WP3b) via ``trace.links``."""
    from agent_hotwash.events import Provenance, ThreadLink, ThreadLinkKind, Trace
    from agent_hotwash.semantic.pipeline import annotate_trace

    cfg = load_config()
    root = _multi_call_session()
    child = _child_session("child-1")
    trace = Trace(
        trace_id="t",
        agent=AgentKind.unknown,
        root=root,
        subagents=[child],
        links=[ThreadLink(child_id="child-1", parent_id=root.session_id, kind=ThreadLinkKind.spawn)],
        provenance=Provenance(source_format="codex_native", detector_confidence="high", root_path=Path("x")),
    )
    tasks, episodes, feature_sets, _caps = annotate_trace(trace, cfg, mode="off")
    assert feature_sets == []
    by_session = {sid: [t for t in tasks if t.session_id == sid] for sid in (root.session_id, "child-1")}
    assert by_session[root.session_id] and by_session["child-1"]
    assert {ep.episode_id.split(":")[0] for ep in episodes} == {root.session_id, "child-1"}
    # every child model call is owned by exactly one child episode
    child_owned = [rid for ep in episodes if ep.episode_id.startswith("child-1:") for rid in ep.response_ids]
    assert child_owned == ["cr1"]
    child_task = by_session["child-1"][0]
    assert child_task.initiator == "parent_agent"
    assert child_task.parent_task == by_session[root.session_id][0].task_id
    spawn_ep = next(ep for ep in episodes if ep.episode_id == child_task.spawn_episode_id)
    assert "agent.spawn" in spawn_ep.ops


def test_spawned_thread_ids_cover_create_thread_results_and_multi_spawn_atoms() -> None:
    """Real parents start children via ``agent.spawn`` (id in args/receivers),
    MCP ``create_thread`` (id only in the *result*) or, on 0.150, are only
    evidenced by ``agent.activity`` meta events; several spawns can share one
    atom. Every one of them must bind."""
    from agent_hotwash.structure.episodes import _spawned_thread_ids

    a, b, c, d = (f"0000000{i}-1111-2222-3333-444444444444" for i in range(1, 5))
    events = [
        Event(kind=EventKind.tool_call, idx=0, op_kind="agent.spawn", tool_args={"receiver_thread_ids": [a, b]}),
        Event(
            kind=EventKind.tool_call,
            idx=1,
            op_kind="mcp.codex_app.create_thread",
            call_id="k",
            tool_args={"title": "x"},
        ),
        Event(
            kind=EventKind.tool_result,
            idx=2,
            op_kind="mcp.codex_app.create_thread",
            call_id="k",
            output=f'{{"thread_id": "{c}"}}',
        ),
        Event(
            kind=EventKind.meta,
            idx=3,
            op_kind="agent.activity",
            tool_args={"agent_thread_id": d, "activity": "started"},
        ),
        Event(kind=EventKind.tool_call, idx=4, op_kind="cmd.exec", tool_args={"command": f"echo {a}"}),  # not a spawn
    ]
    assert _spawned_thread_ids(events) == [a, b, c, d]

    cfg = load_config()
    session = _multi_call_session()
    # add a create_thread call+result to the same atom as the agent.spawn
    spawn_idx = next(e.idx for e in session.events if e.op_kind == "agent.spawn")
    session.events[spawn_idx].tool_args = {"receiver_thread_ids": ["child-1", "child-2"]}
    tasks = segment_tasks(session, cfg, semantic_mode="off")
    episodes = segment_episodes(session, tasks, cfg)
    spawn_ep = next(ep for ep in episodes if "agent.spawn" in ep.ops)
    assert spawn_ep.spawned_thread_ids == []  # "child-1" is not a thread id shape...
    for tid in ("child-1", "child-2"):  # ...but attach still falls back to the raw spawn_op
        t = segment_tasks(session, cfg, semantic_mode="off")[0]
        attach_delegation(t, episodes, tid)
        assert t.spawn_episode_id == spawn_ep.episode_id
