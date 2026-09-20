"""Decoder behaviors found wrong against real 0.150-0.155 rollouts.

Each test builds a minimal record stream shaped like the real files (envelope
``ordinal``, ``file://`` cwd, ``Message Type:`` inter-agent headers, ...) and
checks the fix. See docs in ``codex_native.py`` for the record shapes.
"""

from __future__ import annotations

from typing import Any

from agent_hotwash.canonical import observe_capabilities
from agent_hotwash.events import AgentKind, CapLevel, EventKind, RoleHint
from agent_hotwash.sources._common import build_session
from agent_hotwash.sources.codex_items import _artifacts_from_compound, _decode_cwd, _resolve_path
from agent_hotwash.sources.codex_native import _DECLARED_V2, decode_codex_native_full

SID = "aaaaaaaa-0000-0000-0000-000000000001"
PARENT = "bbbbbbbb-0000-0000-0000-000000000002"
CWD_URL = "file:///Volumes/Crucial%20X9%20Pro/repo"
CWD = "/Volumes/Crucial X9 Pro/repo"


def _rec(i: int, rtype: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"timestamp": f"2026-09-01T00:00:{i:02d}.000Z", "type": rtype, "ordinal": i, "payload": payload}


def _meta(i: int, **extra: Any) -> dict[str, Any]:
    return _rec(i, "session_meta", {"id": SID, "cli_version": "0.153.4", "cwd": CWD, **extra})


def _item(i: int, item: dict[str, Any], turn: str = "t1") -> dict[str, Any]:
    return _rec(i, "event_msg", {"type": "item_completed", "thread_id": SID, "turn_id": turn, "item": item})


def _usage(i: int, rid: str, turn: str = "t1", **tok: int) -> dict[str, Any]:
    usage = {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10, **tok}
    return _rec(i, "token_usage_record", {"thread_id": SID, "turn_id": turn, "response_id": rid, "usage": usage})


def _cmd(cid: str, command: str, parsed: list[dict[str, Any]], *, status: str = "completed", exit_code: Any = "0"):
    return {
        "type": "CommandExecution",
        "id": cid,
        "command": ["/bin/zsh", "-lc", command],
        "cwd": CWD_URL,
        "parsed_cmd": parsed,
        "status": status,
        "exit_code": exit_code,
        "aggregated_output": "",
    }


def _decode(records: list[dict[str, Any]]):
    events, _sid, _model, _meta, turns, notes = decode_codex_native_full(records)
    return events, turns, notes


# --------------------------------------------------------------------------- paths


def test_cwd_url_and_relative_paths_resolve_to_absolute() -> None:
    assert _decode_cwd(CWD_URL) == CWD
    assert _decode_cwd(CWD) == CWD
    assert _resolve_path("src/app.py", CWD) == f"{CWD}/src/app.py"
    assert _resolve_path("../x/./y.py", CWD) == "/Volumes/Crucial X9 Pro/x/y.py"
    assert _resolve_path("/abs/p.py", CWD) == "/abs/p.py"


def test_compound_unknown_command_yields_read_artifacts() -> None:
    cmd = "printf '%s\\n' '--- a ---' && sed -n '1,150p' wikow/app.py && rg -n 'foo' tests src/x.py | head -20"
    arts = _artifacts_from_compound(cmd, CWD)
    paths = {(a.path, a.op.value) for a in arts}
    assert (f"{CWD}/wikow/app.py", "read") in paths
    assert (f"{CWD}/src/x.py", "search") in paths
    # the rg pattern and printf format string are not paths
    assert not any("foo" in p or "%s" in p for p, _ in paths)


def test_command_execution_artifacts_use_cwd_and_unknown_parse() -> None:
    records = [
        _meta(0),
        _item(
            1,
            _cmd(
                "c1",
                "nl -ba app/lib/x.test.ts",
                [{"type": "read", "cmd": "nl -ba app/lib/x.test.ts", "path": "app/lib/x.test.ts"}],
            ),
        ),
        _item(
            2,
            _cmd(
                "c2",
                "sed -n '1,80p' src/a.py && rg foo src",
                [{"type": "unknown", "cmd": "sed -n '1,80p' src/a.py && rg foo src"}],
            ),
        ),
        _item(
            3,
            {
                "type": "FileChange",
                "id": "f1",
                "changes": {f"{CWD}/app/lib/x.test.ts": {"type": "update", "unified_diff": "+a\n-b\n"}},
                "status": "completed",
            },
        ),
        _usage(4, "r1"),
    ]
    events, _turns, _notes = _decode(records)
    calls = [e for e in events if e.kind is EventKind.tool_call]
    assert calls[0].path == f"{CWD}/app/lib/x.test.ts"
    assert {a.path for a in calls[1].artifacts} == {f"{CWD}/src/a.py"}
    session = build_session(events, AgentKind.codex, session_id=SID)
    fs = session.file_state[f"{CWD}/app/lib/x.test.ts"]
    assert fs.read_at is not None and fs.edited_at is not None and fs.read_at < fs.edited_at


# --------------------------------------------------------------------------- file change kinds


def test_file_change_add_delete_edit_kinds_and_line_sums() -> None:
    records = [
        _meta(0),
        _item(1, {"type": "FileChange", "id": "add", "changes": {"/r/new.py": {"type": "add", "content": "a\nb\n"}}}),
        _item(2, {"type": "FileChange", "id": "del", "changes": {"/r/old.py": {"type": "delete"}}}),
        _item(
            3,
            {
                "type": "FileChange",
                "id": "multi",
                "changes": {
                    "/r/a.py": {"type": "update", "unified_diff": "+1\n+2\n-3\n"},
                    "/r/b.py": {"type": "add", "content": "x\n"},
                },
            },
        ),
        _usage(4, "r1"),
    ]
    events, _turns, _notes = _decode(records)
    by_id = {e.call_id: e for e in events if e.kind is EventKind.tool_call}
    assert by_id["add"].op_kind == "file.write" and by_id["add"].lines_added == 2
    assert by_id["del"].op_kind == "file.delete"
    assert by_id["multi"].op_kind == "file.edit"
    assert by_id["multi"].lines_added == 3 and by_id["multi"].lines_removed == 1
    assert len(by_id["multi"].artifacts) == 2


# --------------------------------------------------------------------------- exec outcome


def test_exec_status_and_string_exit_codes() -> None:
    records = [
        _meta(0),
        _item(1, _cmd("ok", "true", [], exit_code="0")),
        _item(2, _cmd("fail", "false", [], status="failed", exit_code="1")),
        _item(3, _cmd("aborted", "sleep 100", [], status="aborted", exit_code=None)),
        _item(4, _cmd("running", "sleep 1", [], status="in_progress", exit_code=None)),
        _usage(5, "r1"),
    ]
    events, _turns, _notes = _decode(records)
    res = {e.call_id: e for e in events if e.kind is EventKind.tool_result}
    assert res["ok"].ok is True and res["ok"].exit_code == 0
    assert res["fail"].ok is False and res["fail"].exit_code == 1
    assert res["aborted"].ok is False and res["aborted"].exit_code is None
    assert res["running"].ok is None and res["running"].error_text is None


# --------------------------------------------------------------------------- replay / forks


def test_fork_history_base_is_not_a_replay_cutoff() -> None:
    """Fork files restart at ordinal 0; ``history_base.end_ordinal_exclusive``
    points into the parent's file and must not drop the child's own records."""
    records = [
        _meta(
            0,
            thread_source="agent_forked_thread",
            forked_from_id=PARENT,
            history_base={"thread_id": PARENT, "end_ordinal_exclusive": 561},
        ),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(
            2,
            {
                "type": "AgentMessage",
                "id": "m",
                "content": [{"type": "output_text", "text": "hi"}],
                "phase": "final_answer",
            },
        ),
        _item(3, _cmd("c", "ls", [])),
        _usage(4, "r1"),
    ]
    events, turns, notes = _decode(records)
    assert any(e.kind is EventKind.assistant_msg for e in events)
    assert any(e.kind is EventKind.tool_call for e in events)
    assert len(turns) == 1 and turns[0].model_calls and turns[0].model_calls[0].usage is not None
    assert "no item_completed" not in " ".join(notes)


def test_spawned_child_prefix_uses_envelope_ordinal() -> None:
    records = [
        _meta(0, thread_source="subagent", parent_thread_id=PARENT, subagent_history_start_ordinal=3),
        _rec(1, "session_meta", {"id": PARENT, "cli_version": "0.153.4"}),
        _rec(2, "compacted", {"message": "parent summary"}),
        _rec(3, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(4, _cmd("c", "ls", [])),
        _usage(5, "r1"),
    ]
    events, turns, _notes = _decode(records)
    # the parent's compaction is kept as context, but not counted as ours
    replay = [e for e in events if e.raw_type == "compacted_replay"]
    assert replay and replay[0].kind is EventKind.meta
    assert not any(e.kind is EventKind.compaction for e in events)
    assert len(turns) == 1 and turns[0].compactions == 0


# --------------------------------------------------------------------------- compaction dedupe


def test_compacted_record_and_context_compaction_item_count_once() -> None:
    records = [
        _meta(0),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(2, _cmd("c", "ls", [])),
        _rec(3, "compacted", {"message": "summary", "replacement_history": [{"text": "s"}]}),
        _usage(4, "r-compact"),
        _item(5, {"type": "ContextCompaction", "id": "cc"}),
        _item(6, _cmd("c2", "ls", [])),
        _usage(7, "r2"),
    ]
    events, turns, _notes = _decode(records)
    comps = [e for e in events if e.kind is EventKind.compaction]
    assert len(comps) == 1 and comps[0].raw_type == "compacted" and comps[0].text == "s"
    assert turns[0].compactions == 1
    # a lone ContextCompaction (no compacted record nearby) still counts
    lone = [
        _meta(0),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(2, {"type": "ContextCompaction", "id": "cc"}),
        _usage(3, "r1"),
    ]
    events2, turns2, _ = _decode(lone)
    assert sum(1 for e in events2 if e.kind is EventKind.compaction) == 1 and turns2[0].compactions == 1


# --------------------------------------------------------------------------- multi-agent ops


def test_subagent_activity_is_meta_and_collab_item_dedupes_function_call() -> None:
    records = [
        _meta(0),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _rec(2, "response_item", {"type": "function_call", "name": "wait", "call_id": "call_1", "arguments": "{}"}),
        _item(
            3,
            {
                "type": "CollabAgentToolCall",
                "id": "call_1",
                "tool": "wait",
                "status": "completed",
                "receiver_thread_ids": [PARENT],
            },
        ),
        _item(
            4,
            {
                "type": "SubAgentActivity",
                "id": "call_9",
                "kind": "started",
                "agent_thread_id": PARENT,
                "agent_path": "/root/x",
            },
        ),
        _item(
            5,
            {
                "type": "CollabAgentToolCall",
                "id": "call_2",
                "tool": "wait",
                "status": "completed",
                "receiver_thread_ids": [],
            },
        ),
        _usage(6, "r1"),
    ]
    events, _turns, _notes = _decode(records)
    calls = [e for e in events if e.kind is EventKind.tool_call]
    assert [c.call_id for c in calls] == ["call_1", "call_2"]  # no duplicate for call_1
    assert calls[0].tool_args and calls[0].tool_args.get("receiver_thread_ids") == [PARENT]
    activity = [e for e in events if e.raw_type == "SubAgentActivity"]
    assert activity and activity[0].kind is EventKind.meta


def test_incoming_new_task_is_child_user_input_and_child_reply_is_result() -> None:
    def amsg(i: int, author: str, recipient: str, mtype: str, turn: str = "t1") -> dict[str, Any]:
        return _rec(
            i,
            "response_item",
            {
                "type": "agent_message",
                "id": f"amsg_{i}",
                "author": author,
                "recipient": recipient,
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Message Type: {mtype}\nTask name: {recipient}\nSender: {author}\nPayload:\n",
                    }
                ],
                "internal_chat_message_metadata_passthrough": {"turn_id": turn},
            },
        )

    child = [
        _meta(
            0,
            thread_source="subagent",
            parent_thread_id=PARENT,
            source={
                "subagent": {"thread_spawn": {"parent_thread_id": PARENT, "agent_path": "/root/worker", "depth": 1}}
            },
        ),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        amsg(2, "/root", "/root/worker", "NEW_TASK"),
        _item(3, _cmd("c", "ls", [])),
        _usage(4, "r1"),
    ]
    events, turns, _notes = _decode(child)
    msgs = [e for e in events if e.kind is EventKind.user_msg]
    assert len(msgs) == 1 and msgs[0].role_hint is RoleHint.delegation
    assert not any(e.kind is EventKind.assistant_msg for e in events)
    assert turns[0].user_input.kind == "delegation" and turns[0].user_input.source_thread_id == PARENT

    parent = [
        _meta(0),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        amsg(2, "/root/worker", "/root", "FINAL_ANSWER"),
        _item(3, _cmd("c", "ls", [])),
        _usage(4, "r1"),
    ]
    events, turns, _notes = _decode(parent)
    assert not any(e.kind is EventKind.user_msg for e in events)
    res = [e for e in events if e.kind is EventKind.tool_result]
    assert res and res[0].op_kind == "agent.message" and "FINAL_ANSWER" in (res[0].output or "")
    assert turns[0].user_input.kind == "none"


def test_created_thread_learns_parent_from_create_thread_output() -> None:
    records = [
        _meta(0, thread_source="agent_created_thread"),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(
            2,
            {
                "type": "FunctionCallOutput",
                "id": "fco",
                "namespace": "codex_app",
                "name": "create_thread",
                "output": f"<codex_delegation><source_thread_id>{PARENT}</source_thread_id></codex_delegation>",
            },
        ),
        _usage(3, "r1"),
    ]
    _events, turns, _notes = _decode(records)
    assert turns[0].user_input.kind == "delegation" and turns[0].user_input.source_thread_id == PARENT


# --------------------------------------------------------------------------- usage fallback


def test_token_count_fallback_when_no_usage_records() -> None:
    def tc(i: int, turn: str, inp: int, out: int) -> dict[str, Any]:
        return _rec(
            i,
            "event_msg",
            {
                "type": "token_count",
                "turn_id": turn,
                "info": {
                    "total_token_usage": {"input_tokens": 10**6, "output_tokens": 10**5},
                    "last_token_usage": {"input_tokens": inp, "cached_input_tokens": inp // 2, "output_tokens": out},
                },
            },
        )

    records = [
        _meta(0, cli_version="0.150.0-alpha.12.2"),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(2, _cmd("c1", "ls", [])),
        tc(3, "t1", 1000, 50),
        _item(4, _cmd("c2", "ls", [])),
        tc(5, "t1", 2000, 70),
    ]
    events, turns, notes = _decode(records)
    usages = [e.usage for e in events if e.usage is not None]
    assert [(u.input, u.cache_read, u.output) for u in usages] == [(500, 500, 50), (1000, 1000, 70)]
    assert len(turns[0].model_calls) == 2 and turns[0].model_calls[1].usage.output == 70
    assert any("token_count" in n for n in notes)

    # with real usage records, token_count stays a marker
    with_records = [*records, _usage(6, "r1", input_tokens=10, cached_input_tokens=0, output_tokens=1)]
    events, _turns, notes = _decode(with_records)
    assert next(e.usage for e in events if e.usage is not None).input == 10
    assert not any("token_count" in n for n in notes)


# --------------------------------------------------------------------------- turn ranges / capabilities


def test_turn_event_range_includes_its_last_event() -> None:
    records = [
        _meta(0),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(2, _cmd("c1", "ls", [])),  # 2 events: call + result, never terminated by usage
    ]
    events, turns, _notes = _decode(records)
    last_t1 = max(i for i, e in enumerate(events) if e.turn_id == "t1")
    assert turns[0].event_end == last_t1 == len(events) - 1


def test_truncation_marker_does_not_promote_full_output() -> None:
    records = [
        _meta(0),
        _item(
            1, {**_cmd("c", "cat big", []), "aggregated_output": "x\n[truncated output (original token count: 9999)]"}
        ),
        _usage(2, "r1"),
    ]
    events, turns, _notes = _decode(records)
    session = build_session(events, AgentKind.codex, session_id=SID)
    session.turns = turns
    caps = observe_capabilities(session, _DECLARED_V2)
    assert caps.observed.full_tool_output is CapLevel.partial
    assert caps.observed.output_size_original is CapLevel.true


# --------------------------------------------------------------------------- directory linkage


def test_created_child_links_to_parent_via_create_thread_output(tmp_path) -> None:
    """Real ``agent_created_thread`` files name their parent only inside the
    ``<codex_delegation>`` echoed by ``create_thread``; the directory index and
    the forest must pick that up (105/105 real children, 0 via UserMessage)."""
    import json

    from agent_hotwash.sources.codex_forest import components, index_rollouts
    from agent_hotwash.sources.detect import discover

    day1, day2 = tmp_path / "13", tmp_path / "14"
    day1.mkdir()
    day2.mkdir()
    parent = day1 / "rollout-parent.jsonl"
    parent.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _rec(0, "session_meta", {"id": PARENT, "cli_version": "0.153.4"}),
                _rec(1, "event_msg", {"type": "task_started", "turn_id": "p1"}),
                {
                    **_item(2, _cmd("c", "ls", []), turn="p1"),
                    "payload": {**_item(2, _cmd("c", "ls", []))["payload"], "thread_id": PARENT},
                },
                _rec(
                    3,
                    "token_usage_record",
                    {
                        "thread_id": PARENT,
                        "turn_id": "p1",
                        "response_id": "r",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    child = day2 / "rollout-child.jsonl"
    fco = {
        "type": "FunctionCallOutput",
        "id": "fco",
        "namespace": "codex_app",
        "name": "create_thread",
        "output": f"<codex_delegation><source_thread_id>{PARENT}</source_thread_id></codex_delegation>",
    }
    child.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _meta(0, thread_source="agent_created_thread"),
                _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
                _item(2, fco),
                _item(3, _cmd("c", "ls", [])),
                _usage(4, "r1"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metas = index_rollouts([parent, child])
    child_meta = next(m for m in metas if m["id"] == SID)
    assert child_meta["delegation_source_thread_id"] == PARENT
    comps = components(metas)
    assert len(comps) == 1 and comps[0].root_id == PARENT
    assert {e.kind.value for e in comps[0].edges} == {"created"}
    units = discover(tmp_path)
    assert len(units) == 1 and units[0].component is not None
    assert set(units[0].component.paths.values()) == {parent, child}


def test_send_message_output_is_not_parentage(tmp_path) -> None:
    """``send_message_to_thread`` outputs echo ``<source_thread_id>`` = the
    *sender*; only ``create_thread`` names a creator (1 134 such outputs on
    the real month must never link a thread to whoever messaged it)."""
    import json

    from agent_hotwash.sources.codex_native import index_session_meta

    sender_fco = {
        "type": "FunctionCallOutput",
        "id": "fco",
        "namespace": "codex_app",
        "name": "send_message_to_thread",
        "output": f"<codex_delegation><source_thread_id>{PARENT}</source_thread_id></codex_delegation>",
    }
    path = tmp_path / "rollout-x.jsonl"
    records = [
        _meta(0, thread_source="user"),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _rec(2, "response_item", {**sender_fco, "type": "function_call_output"}),
        _item(3, sender_fco),
        _item(4, _cmd("c", "ls", [])),
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    meta = index_session_meta(path)
    assert meta is not None and "delegation_source_thread_id" not in meta

    created = {**sender_fco, "name": "create_thread"}
    path.write_text(
        "\n".join(json.dumps(r) for r in [records[0], records[1], _item(2, created), records[4]]) + "\n",
        encoding="utf-8",
    )
    meta = index_session_meta(path)
    assert meta is not None and meta["delegation_source_thread_id"] == PARENT


def test_versionless_header_waits_for_item_completed() -> None:
    """A ``session_meta`` with no parseable ``cli_version`` must not commit to
    the legacy decoder: the first ``item_completed`` settles v2."""
    records = [
        _rec(0, "session_meta", {"id": SID, "cwd": CWD}),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        _item(2, _cmd("c", "ls", [{"type": "list_files", "cmd": "ls", "path": "."}])),
        _usage(3, "r1"),
    ]
    events, turns, notes = _decode(records)
    assert [e.kind for e in events if e.kind is EventKind.tool_call] == [EventKind.tool_call]
    assert len(turns) == 1
    assert not any("legacy" in n for n in notes)


def test_compaction_item_far_in_time_is_not_paired_with_record() -> None:
    """The record/item pairing collapses a ``ContextCompaction`` item only when
    it is close to the last ``compacted`` record in both position *and* time;
    a genuine later compaction nearby in the stream stays counted."""
    from datetime import UTC, datetime

    def at(i: int, minute: int) -> str:
        return datetime(2026, 9, 1, 0, minute, i, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    compacted = {"type": "compacted", "payload": {"message": "s", "replacement_history": []}}
    paired = [
        _meta(0),
        _rec(1, "event_msg", {"type": "task_started", "turn_id": "t1"}),
        {**compacted, "timestamp": at(2, 0), "ordinal": 2},
        {**_item(3, {"type": "ContextCompaction", "id": "cc1"}), "timestamp": at(3, 0)},
        _usage(4, "r1"),
    ]
    events, _turns, _notes = _decode(paired)
    assert sum(1 for e in events if e.kind is EventKind.compaction) == 1

    distant = [*paired[:3], {**_item(3, {"type": "ContextCompaction", "id": "cc1"}), "timestamp": at(3, 10)}, paired[4]]
    events, _turns, _notes = _decode(distant)
    assert sum(1 for e in events if e.kind is EventKind.compaction) == 2
