"""Generate compact v0153/v0150 Codex fixtures. Run: python tests/fixtures/codex_native/_gen.py"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V0153 = HERE / "v0153"
V0150 = HERE / "v0150"

SID = "aaaaaaaa-0000-0000-0000-000000000001"
TID1 = "turn-0001"
TID2 = "turn-0002"
PARENT = "bbbbbbbb-0000-0000-0000-000000000002"
CHILD = "cccccccc-0000-0000-0000-000000000003"
FORK = "dddddddd-0000-0000-0000-000000000004"
CREATED = "eeeeeeee-0000-0000-0000-000000000005"


def rec(ts: str, rtype: str, payload: dict) -> str:
    return json.dumps({"timestamp": ts, "type": rtype, "payload": payload}, separators=(",", ":"))


def session_meta(sid: str, *, cli: str = "0.153.4", **extra: object) -> dict:
    body: dict = {
        "id": sid,
        "session_id": extra.pop("session_id", sid),
        "cwd": "/work",
        "cli_version": cli,
        "originator": "cli",
        "model_provider": "openai",
        **extra,
    }
    return body


def turn_context(turn_id: str, model: str = "gpt-5.6-luna", effort: str = "high") -> dict:
    return {"turn_id": turn_id, "model": model, "effort": effort, "cwd": "/work"}


def item(itype: str, **fields: object) -> dict:
    return {"type": itype, **fields}


def item_completed(turn_id: str, item_body: dict, started: int = 1, completed: int = 2) -> dict:
    return {
        "type": "item_completed",
        "thread_id": SID,
        "turn_id": turn_id,
        "item": item_body,
        "started_at_ms": started,
        "completed_at_ms": completed,
    }


def usage(
    rid: str,
    turn_id: str,
    thread_id: str,
    inp: int,
    cached: int,
    out: int,
    reason: int,
    thread_in: int,
    thread_out: int,
) -> dict:
    u = {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": out,
        "reasoning_output_tokens": reason,
        "total_tokens": inp + out,
    }
    return {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "session_id": thread_id,
        "response_id": rid,
        "usage": u,
        "turn_token_usage": u,
        "thread_token_usage": {
            "input_tokens": thread_in,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": thread_out,
            "reasoning_output_tokens": reason,
            "total_tokens": thread_in + thread_out,
        },
    }


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gen_user_multiturn() -> None:
    """2 completed turns, 83 model calls, 2 user + 2 injected, 17 assistant."""
    lines: list[str] = []
    t0 = "2026-09-18T00:00:00.000Z"
    lines.append(rec(t0, "session_meta", session_meta(SID, thread_source="user")))
    n_calls = 83
    assistants_left = 17
    # per-call usage: input=100 (incl 40 cached) → uncached 60, output=10, reason=4
    # thread cumulative: first record thread = that usage; each adds 100 in / 10 out
    thread_in = 0
    thread_out = 0

    def add_call(i: int, turn_id: str, *, assistant: bool = False, extra_items: list[dict] | None = None) -> None:
        nonlocal thread_in, thread_out, assistants_left
        ts = f"2026-09-18T00:{i // 60:02d}:{i % 60:02d}.000Z"
        if extra_items:
            for it in extra_items:
                lines.append(rec(ts, "event_msg", item_completed(turn_id, it)))
        if assistant and assistants_left > 0:
            phase = "final_answer" if assistants_left == 1 or i == n_calls - 1 else "commentary"
            lines.append(
                rec(
                    ts,
                    "event_msg",
                    item_completed(
                        turn_id,
                        item(
                            "AgentMessage",
                            id=f"msg-{i}",
                            content=[{"type": "Text", "text": f"update {i}"}],
                            phase=phase,
                        ),
                    ),
                )
            )
            assistants_left -= 1
        lines.append(rec(ts, "event_msg", item_completed(turn_id, item("Reasoning", id=f"rs-{i}", summary_text=[]))))
        thread_in += 100
        thread_out += 10
        lines.append(
            rec(
                ts,
                "token_usage_record",
                usage(f"resp-{i:03d}", turn_id, SID, 100, 40, 10, 4, thread_in, thread_out),
            )
        )

    # turn 1
    lines.append(
        rec(
            "2026-09-18T00:00:01.000Z",
            "event_msg",
            {"type": "task_started", "turn_id": TID1, "model_context_window": 200000},
        )
    )
    lines.append(rec("2026-09-18T00:00:02.000Z", "turn_context", turn_context(TID1, effort="high")))
    lines.append(
        rec(
            "2026-09-18T00:00:03.000Z",
            "event_msg",
            item_completed(
                TID1,
                item(
                    "UserMessage",
                    id="u1",
                    content=[{"type": "text", "text": "<environment_context>cwd=/work</environment_context>"}],
                ),
            ),
        )
    )
    lines.append(
        rec(
            "2026-09-18T00:00:04.000Z",
            "event_msg",
            item_completed(
                TID1,
                item("UserMessage", id="u2", content=[{"type": "text", "text": "Fix the failing test in src/a.py"}]),
            ),
        )
    )
    # a few ops + 40 calls
    lines.append(
        rec(
            "2026-09-18T00:00:05.000Z",
            "event_msg",
            item_completed(
                TID1,
                item(
                    "CommandExecution",
                    id="exec-read-1",
                    command=["/bin/zsh", "-lc", "rg -n foo src"],
                    parsed_cmd=[{"type": "search", "cmd": "rg -n foo src", "path": "src"}],
                    status="completed",
                    exit_code=0,
                    stdout="src/a.py:1:foo",
                    aggregated_output="src/a.py:1:foo",
                ),
            ),
        )
    )
    lines.append(
        rec(
            "2026-09-18T00:00:06.000Z",
            "event_msg",
            item_completed(
                TID1,
                item(
                    "CommandExecution",
                    id="exec-err-1",
                    command=["/bin/zsh", "-lc", "cat nope.txt"],
                    parsed_cmd=[{"type": "read", "cmd": "cat nope.txt", "path": "nope.txt"}],
                    status="completed",
                    exit_code=1,
                    stdout="cat: nope.txt: No such file or directory",
                    aggregated_output="cat: nope.txt: No such file or directory",
                ),
            ),
        )
    )
    lines.append(
        rec(
            "2026-09-18T00:00:07.000Z",
            "event_msg",
            item_completed(
                TID1,
                item(
                    "FileChange",
                    id="fc-1",
                    changes={"/work/src/a.py": {"type": "update", "unified_diff": "@@\n-old\n+new\n"}},
                    status="completed",
                    stdout="Success. Updated src/a.py",
                ),
            ),
        )
    )
    for i in range(40):
        add_call(i, TID1, assistant=(i % 5 == 0))
    lines.append(
        rec(
            "2026-09-18T00:40:00.000Z",
            "event_msg",
            {"type": "task_complete", "turn_id": TID1, "last_agent_message": "Turn 1 done"},
        )
    )

    # turn 2 — repeated turn_context, compaction, settings, injected
    lines.append(
        rec(
            "2026-09-18T01:00:00.000Z",
            "event_msg",
            {"type": "task_started", "turn_id": TID2, "model_context_window": 200000},
        )
    )
    lines.append(rec("2026-09-18T01:00:01.000Z", "turn_context", turn_context(TID2, effort="high")))
    lines.append(rec("2026-09-18T01:00:02.000Z", "turn_context", turn_context(TID2, effort="max")))  # revision
    lines.append(rec("2026-09-18T01:00:03.000Z", "event_msg", {"type": "thread_settings_applied", "turn_id": TID2}))
    lines.append(
        rec(
            "2026-09-18T01:00:04.000Z",
            "event_msg",
            item_completed(
                TID2,
                item(
                    "UserMessage",
                    id="u3",
                    content=[{"type": "text", "text": "<recommended_plugins>none</recommended_plugins>"}],
                ),
            ),
        )
    )
    lines.append(
        rec(
            "2026-09-18T01:00:05.000Z",
            "event_msg",
            item_completed(
                TID2, item("UserMessage", id="u4", content=[{"type": "text", "text": "Also run the tests please"}])
            ),
        )
    )
    lines.append(
        rec("2026-09-18T01:00:06.000Z", "event_msg", item_completed(TID2, item("ContextCompaction", id="cc-1")))
    )
    for i in range(40, n_calls):
        add_call(i, TID2, assistant=(i % 4 == 0))
    lines.append(
        rec(
            "2026-09-18T02:00:00.000Z",
            "event_msg",
            {"type": "task_complete", "turn_id": TID2, "last_agent_message": "Turn 2 done"},
        )
    )
    write(V0153 / "user_multiturn.jsonl", lines)


def gen_ops_all_types() -> None:
    lines = [rec("2026-09-01T00:00:00.000Z", "session_meta", session_meta(SID, thread_source="user"))]
    lines.append(rec("2026-09-01T00:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": TID1}))
    lines.append(rec("2026-09-01T00:00:02.000Z", "turn_context", turn_context(TID1)))
    items = [
        item("UserMessage", id="u", content=[{"type": "text", "text": "hello"}]),
        item("Reasoning", id="r", summary_text=[]),
        item("AgentMessage", id="a", content=[{"type": "Text", "text": "hi"}], phase="commentary"),
        item(
            "CommandExecution",
            id="c",
            command=["ls"],
            parsed_cmd=[{"type": "list_files", "cmd": "ls", "path": None}],
            exit_code=0,
            status="completed",
            stdout="a",
        ),
        item(
            "FileChange",
            id="f",
            changes={"/work/new.py": {"type": "add", "content": "x\ny\n"}},
            status="completed",
            stdout="ok",
        ),
        item(
            "McpToolCall",
            id="m",
            server="codex_app",
            tool="list_threads",
            arguments='{"limit":1}',
            result={"content": "[]", "isError": False},
        ),
        item(
            "FunctionCallOutput",
            id="fo",
            name="create_thread",
            namespace="codex_app",
            output="<codex_delegation><source_thread_id>bbbb</source_thread_id></codex_delegation>",
        ),
        item("SubAgentActivity", id="sa", kind="interacted", agent_thread_id=CHILD, agent_path="/work"),
        item("Extension", kind="web.search", id="ex", query="q", action={"type": "search"}, results=[]),
        item("ContextCompaction", id="cc"),
        item("CollabAgentToolCall", id="ca", tool="wait", sender_thread_id=SID, receiver_thread_ids=[]),
        item("ImageView", id="iv", path="file:///tmp/x.png"),
    ]
    for it in items:
        lines.append(rec("2026-09-01T00:00:03.000Z", "event_msg", item_completed(TID1, it)))
    write(V0153 / "ops_all_types.jsonl", lines)


def gen_exec_outputs() -> None:
    lines = [rec("2026-09-01T00:00:00.000Z", "session_meta", session_meta(SID))]
    variants = [
        ("ok", 0, "Script completed\nWall time 0.5 seconds\nOutput:\nok\n"),
        ("fail", 1, "Script failed\nOutput:\nboom\n"),
        ("abort", None, "aborted by user after 977.8s"),
        ("json_exit", 3, 'text({"exit_code":3,"ok":false})'),
        ("trunc_warn", 0, "hello\nWarning: truncated output (original token count: 9999)\n"),
        ("trunc_json", 0, '{"original_token_count": 4321, "text": "x"}'),
    ]
    for name, exit_code, stdout in variants:
        body = item(
            "CommandExecution",
            id=f"exec-{name}",
            command=["echo"],
            parsed_cmd=[{"type": "unknown", "cmd": "echo"}],
            status="completed" if exit_code == 0 else "failed",
            stdout=stdout,
            aggregated_output=stdout,
        )
        if exit_code is not None:
            body["exit_code"] = exit_code
        lines.append(rec("2026-09-01T00:00:01.000Z", "event_msg", item_completed(TID1, body)))
    write(V0153 / "exec_outputs.jsonl", lines)


def gen_multi_op_exec() -> None:
    lines = [rec("2026-09-01T00:00:00.000Z", "session_meta", session_meta(SID))]
    gid = "call_wrap_1"
    lines.append(
        rec(
            "2026-09-01T00:00:01.000Z",
            "response_item",
            {
                "type": "custom_tool_call",
                "name": "exec",
                "call_id": gid,
                "input": "await tools.exec_command({cmd:'ls'}); await tools.apply_patch({});",
            },
        )
    )
    for n, itype in enumerate(("CommandExecution", "FileChange", "CommandExecution")):
        if itype == "CommandExecution":
            it = item(
                "CommandExecution",
                id=f"child-{n}",
                command=["ls"],
                parsed_cmd=[{"type": "list_files", "cmd": "ls"}],
                exit_code=0,
                status="completed",
                stdout="a",
            )
        else:
            it = item(
                "FileChange",
                id=f"child-{n}",
                changes={"/work/a.py": {"type": "update", "unified_diff": "@@\n-a\n+b\n"}},
                status="completed",
            )
        lines.append(rec("2026-09-01T00:00:02.000Z", "event_msg", item_completed(TID1, it)))
    lines.append(
        rec(
            "2026-09-01T00:00:03.000Z",
            "response_item",
            {"type": "custom_tool_call_output", "call_id": gid, "output": "Script completed"},
        )
    )
    # malformed nesting
    lines.append(
        rec(
            "2026-09-01T00:00:04.000Z",
            "response_item",
            {"type": "custom_tool_call", "name": "exec", "call_id": "call_wrap_2", "input": "x"},
        )
    )
    lines.append(
        rec(
            "2026-09-01T00:00:05.000Z",
            "response_item",
            {"type": "custom_tool_call", "name": "exec", "call_id": "call_wrap_3", "input": "y"},
        )
    )
    write(V0153 / "multi_op_exec.jsonl", lines)


def gen_tree() -> None:
    tree = V0153 / "tree"
    # orchestrator
    orch = [
        rec("2026-09-01T00:00:00.000Z", "session_meta", session_meta(PARENT, thread_source="user")),
        rec("2026-09-01T00:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": TID1}),
        rec("2026-09-01T00:00:02.000Z", "turn_context", turn_context(TID1)),
        rec(
            "2026-09-01T00:00:03.000Z",
            "event_msg",
            item_completed(
                TID1,
                item("UserMessage", id="u", content=[{"type": "text", "text": "orchestrate the work"}]),
            ),
        ),
        rec(
            "2026-09-01T00:00:04.000Z",
            "event_msg",
            item_completed(
                TID1,
                item("SubAgentActivity", id="sa", kind="spawned", agent_thread_id=CHILD, agent_path="/work"),
            ),
        ),
        rec(
            "2026-09-01T00:00:05.000Z",
            "event_msg",
            item_completed(
                TID1,
                item(
                    "McpToolCall",
                    id="ct",
                    server="codex_app",
                    tool="create_thread",
                    arguments="{}",
                    result={"content": json.dumps({"threadId": CREATED}), "isError": False},
                ),
            ),
        ),
    ]
    write(tree / "rollout-orchestrator.jsonl", orch)

    # spawned child with replay prefix
    replay = [
        rec(
            "2026-09-01T00:00:00.000Z",
            "session_meta",
            session_meta(
                CHILD,
                session_id=PARENT,
                thread_source="subagent",
                parent_thread_id=PARENT,
                forked_from_id=PARENT,
                subagent_history_start_ordinal=4,
                source={
                    "subagent": {"thread_spawn": {"parent_thread_id": PARENT, "depth": 1, "agent_nickname": "Gauss"}}
                },
                agent_nickname="Gauss",
            ),
        ),
        rec("2026-09-01T00:00:00.100Z", "session_meta", session_meta(PARENT, thread_source="user")),
        rec("2026-09-01T00:00:00.200Z", "compacted", {"message": "parent summary", "replacement_history": ["a", "b"]}),
        rec("2026-09-01T00:00:00.300Z", "turn_context", turn_context("parent-turn")),
        # ordinal 4 = start; a real child user message + usage belonging to child
        rec(
            "2026-09-01T00:00:01.000Z",
            "event_msg",
            {
                "type": "item_completed",
                "thread_id": CHILD,
                "turn_id": "child-t1",
                "item": item("UserMessage", id="cu", content=[{"type": "text", "text": "do the delegated work"}]),
            },
        ),
        rec(
            "2026-09-01T00:00:02.000Z",
            "token_usage_record",
            usage("resp-child", "child-t1", CHILD, 50, 0, 5, 2, 50, 5),
        ),
        # foreign thread_id (parent) must be dropped
        rec(
            "2026-09-01T00:00:03.000Z",
            "token_usage_record",
            {
                **usage("resp-parent", "parent-turn", PARENT, 999, 0, 9, 0, 999, 9),
                "thread_id": PARENT,
                "session_id": PARENT,
            },
        ),
    ]
    replay[-1] = rec(
        "2026-09-01T00:00:03.000Z",
        "token_usage_record",
        {
            **usage("resp-parent", "parent-turn", PARENT, 999, 0, 9, 0, 999, 9),
            "thread_id": PARENT,
            "session_id": PARENT,
        },
    )
    write(tree / "rollout-child-spawn-replay.jsonl", replay)

    created = [
        rec("2026-09-01T00:00:00.000Z", "session_meta", session_meta(CREATED, thread_source="agent_created_thread")),
        rec("2026-09-01T00:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": "ct1"}),
        rec(
            "2026-09-01T00:00:02.000Z",
            "event_msg",
            {
                "type": "item_completed",
                "thread_id": CREATED,
                "turn_id": "ct1",
                "item": item(
                    "UserMessage",
                    id="du",
                    content=[
                        {
                            "type": "text",
                            "text": (
                                f"<codex_delegation><source_thread_id>{PARENT}"
                                f"</source_thread_id><input>go</input></codex_delegation>"
                            ),
                        }
                    ],
                ),
            },
        ),
    ]
    write(tree / "rollout-child-created.jsonl", created)

    fork = [
        rec(
            "2026-09-01T00:00:00.000Z",
            "session_meta",
            session_meta(
                FORK,
                thread_source="agent_forked_thread",
                forked_from_id=PARENT,
                forked_from_ordinal_exclusive=10,
                history_base={"thread_id": PARENT, "end_ordinal_exclusive": 10},
            ),
        ),
        rec("2026-09-01T00:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": "ft1"}),
        rec(
            "2026-09-01T00:00:02.000Z",
            "token_usage_record",
            {
                **usage("resp-fork", "ft1", FORK, 200, 10, 8, 3, 5443671 + 200, 8),
                "thread_token_usage": {
                    "input_tokens": 5_443_671 + 200,
                    "cached_input_tokens": 5_443_671,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 8,
                    "reasoning_output_tokens": 3,
                    "total_tokens": 5_443_879,
                },
            },
        ),
    ]
    write(tree / "rollout-child-fork.jsonl", fork)


def gen_aborted_open() -> None:
    lines = [rec("2026-09-01T00:00:00.000Z", "session_meta", session_meta(SID))]
    # aborted turn
    lines.append(rec("2026-09-01T00:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": "ab1"}))
    lines.append(rec("2026-09-01T00:00:02.000Z", "turn_context", turn_context("ab1")))
    lines.append(
        rec(
            "2026-09-01T00:00:03.000Z",
            "event_msg",
            item_completed("ab1", item("UserMessage", id="u", content=[{"type": "text", "text": "stop"}])),
        )
    )
    lines.append(
        rec(
            "2026-09-01T00:00:04.000Z", "event_msg", {"type": "turn_aborted", "turn_id": "ab1", "reason": "interrupted"}
        )
    )
    # open turn with unterminated call (no usage record)
    lines.append(rec("2026-09-01T00:00:05.000Z", "event_msg", {"type": "task_started", "turn_id": "op1"}))
    lines.append(
        rec(
            "2026-09-01T00:00:06.000Z",
            "event_msg",
            item_completed(
                "op1", item("AgentMessage", id="a", content=[{"type": "Text", "text": "working"}], phase="commentary")
            ),
        )
    )
    # zero-model-call turn
    lines.append(rec("2026-09-01T00:00:07.000Z", "event_msg", {"type": "task_started", "turn_id": "z1"}))
    lines.append(
        rec(
            "2026-09-01T00:00:08.000Z",
            "event_msg",
            {"type": "task_complete", "turn_id": "z1", "last_agent_message": None},
        )
    )
    write(V0153 / "aborted_open_turns.jsonl", lines)


def gen_v0150_agent_message() -> None:
    lines = [
        rec(
            "2026-09-01T00:00:00.000Z", "session_meta", session_meta(SID, cli="0.150.0-alpha.12.2")
        ),  # no thread_source
        rec("2026-09-01T00:00:01.000Z", "event_msg", {"type": "task_started", "turn_id": TID1}),
        rec("2026-09-01T00:00:02.000Z", "turn_context", turn_context(TID1)),
        rec(
            "2026-09-01T00:00:03.000Z",
            "response_item",
            {"type": "agent_message", "content": [{"type": "text", "text": "legacy agent surface"}]},
        ),
        rec(
            "2026-09-01T00:00:04.000Z",
            "event_msg",
            item_completed(TID1, item("UserMessage", id="u", content=[{"type": "text", "text": "hi"}])),
        ),
    ]
    write(V0150 / "agent_message.jsonl", lines)


if __name__ == "__main__":
    gen_user_multiturn()
    gen_ops_all_types()
    gen_exec_outputs()
    gen_multi_op_exec()
    gen_tree()
    gen_aborted_open()
    gen_v0150_agent_message()
    print("fixtures written")
