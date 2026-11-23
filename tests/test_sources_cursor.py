"""Cursor CLI stream and chat-store regressions using synthetic records."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_hotwash.cli import app
from agent_hotwash.config import PriceEntry, load_config
from agent_hotwash.diagnostics.cost_views import build_cost_views, iter_billed_calls, session_invoice
from agent_hotwash.events import AgentKind, CapLevel, EventKind, TurnStatus
from agent_hotwash.memory.inputs import prepare_inputs
from agent_hotwash.runner import RunOptions, trace_matches_options
from agent_hotwash.sources import _common, cursor_stream
from agent_hotwash.sources._cursor import tool_result
from agent_hotwash.sources.detect import discover, discover_many, iter_traces, load_unit

_START = 1_790_000_000_000
_CURSOR_INIT = {
    "type": "system",
    "subtype": "init",
    "apiKeySource": "login",
    "cwd": "/work/project",
    "session_id": "test-stream-session",
    "model": "Example Model High",
    "permissionMode": "default",
}


def _conversation(trailing_result: bool = True, is_error: bool = False) -> list[dict]:
    shell = {"args": {"command": "uv run pytest", "toolCallId": "check-id"}}
    rows = [
        _CURSOR_INIT,
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Check the build."}]}},
        {"type": "thinking", "subtype": "delta", "text": "Run ", "timestamp_ms": _START},
        {"type": "thinking", "subtype": "delta", "text": "the tests.", "timestamp_ms": _START + 10},
        {"type": "thinking", "subtype": "completed", "timestamp_ms": _START + 20},
        {
            "type": "tool_call",
            "subtype": "started",
            "call_id": "check-id",
            "timestamp_ms": _START + 100,
            "model_call_id": "model-call-id",
            "tool_call": {"startedAtMs": _START + 100, "shellToolCall": shell},
        },
        {
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "check-id",
            "timestamp_ms": _START + 200,
            "tool_call": {
                "startedAtMs": _START + 100,
                "completedAtMs": _START + 200,
                "shellToolCall": {**shell, "result": {"success": {"exitCode": 2, "stderr": "No such file"}}},
            },
        },
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Tests failed."}]}},
    ]
    if trailing_result:
        rows.append(
            {
                "type": "result",
                "subtype": "success",
                "is_error": is_error,
                "request_id": "request-a",
                "duration_ms": 219,
                "result": "Tests failed.",
                "usage": {"inputTokens": 10, "outputTokens": 6, "cacheReadTokens": 8, "reasoningTokens": 2},
            }
        )
    return [dict(row, session_id=_CURSOR_INIT["session_id"]) for row in rows]


def _write_rows(path: Path, rows: list[dict], *, tail: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows) + tail)
    return path


def test_cursor_stream_pairs_tools_and_keeps_aggregate_usage(tmp_path: Path) -> None:
    path = _write_rows(tmp_path / "logs" / "cloop" / "run.jsonl", _conversation())
    (trace,) = iter_traces(path.parent)
    assert trace.agent is AgentKind.cursor
    assert trace.root.session_id == "test-stream-session"
    assert trace.provenance.source_format == "cursor_stream"
    assert trace.provenance.harness_meta["request_id"] == "request-a"
    start, user, thinking, call, result, final, terminal = trace.root.events
    assert [event.idx for event in trace.root.events] == list(range(7))
    assert start.kind is EventKind.session_start and user.text == "Check the build."
    assert thinking.text == "Run the tests." and thinking.ts_end is not None
    assert call.kind is EventKind.tool_call and call.tool_args == {"command": "uv run pytest"}
    assert call.op_kind == "cmd.exec" and call.call_id == result.call_id == "check-id"
    assert call.response_id == "model-call-id"
    assert call.ts_end == result.ts and call.ts is not None and result.ts is not None and call.ts < result.ts
    assert result.ok is False and result.exit_code == 2 and result.output == "No such file"
    assert result.output_sha256 == hashlib.sha256(b"No such file").hexdigest()
    assert result.source is not None and result.source.line_number == 7
    assert final.text == "Tests failed." and final.phase == "final_answer"
    assert terminal.kind is EventKind.session_end and terminal.usage is None
    assert terminal.invocation_usage is not None
    usage = terminal.invocation_usage
    assert usage.input == 10 and usage.cache_read == 8 and usage.output == 6
    assert usage.reasoning_output == 2
    assert all(call.usage is None for turn in trace.root.turns for call in turn.model_calls)
    assert trace.root.turns[0].status is TurnStatus.completed
    assert trace.capabilities.observed.per_call_usage is CapLevel.false


def test_cursor_stream_differentiates_claude_and_keeps_a_truncated_run(tmp_path: Path) -> None:
    claude = _write_rows(
        tmp_path / "claude.jsonl",
        [{**_CURSOR_INIT, "claude_code_version": "2.0", "tools": ["Read"]}],
    )
    assert not discover(claude)
    partial = _write_rows(
        tmp_path / "partial.ndjson",
        _conversation(trailing_result=False)[:-2],
        tail='{"type":"tool_call"',
    )
    (trace,) = iter_traces(tmp_path)
    assert trace.provenance.source_format == "cursor_stream"
    assert trace.root.events[-1].kind is EventKind.tool_call
    assert trace.root.turns[0].status is TurnStatus.open
    assert trace.root.source_diagnostics[0]["status"] == "malformed_or_truncated"
    assert trace.root.source_diagnostics[0]["line_number"] == 7
    assert trace.provenance.files == [partial]


def test_cursor_result_error_aborts_turn(tmp_path: Path) -> None:
    path = _write_rows(tmp_path / "failed.jsonl", _conversation(is_error=True))
    (trace,) = iter_traces(path)
    assert trace.root.turns[0].status is TurnStatus.aborted
    assert trace.root.events[-1].ok is False
    assert not any(event.phase == "final_answer" for event in trace.root.events)


def _varint(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _int_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


@pytest.fixture
def cursor_store(tmp_path: Path):
    path = tmp_path / "chats" / "workspace" / "db-session" / "store.db"
    path.parent.mkdir(parents=True)
    path.with_name("meta.json").write_text(json.dumps({"cwd": "/work/project", "schemaVersion": 1}))
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT); CREATE TABLE blobs(id TEXT, data BLOB)")

        def blob(data: bytes | dict) -> bytes:
            payload = json.dumps(data).encode() if isinstance(data, dict) else data
            digest = hashlib.sha256(payload).digest()
            db.execute("INSERT INTO blobs VALUES (?, ?)", (digest.hex(), payload))
            return digest

        # Intentionally write the result first. Only the checkpoint defines message order.
        file_contents = blob(b"hello from source")
        result = blob(
            {
                "role": "tool",
                "content": [{"type": "tool-result", "toolCallId": "read-id", "toolName": "Read", "result": "excerpt"}],
                "providerOptions": {
                    "cursor": {
                        "highLevelToolCallResult": {
                            "isError": False,
                            "output": {"success": {"contentBlobId": file_contents.hex()}},
                        }
                    }
                },
            }
        )
        call = blob(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "text": "Read the file.",
                        "providerOptions": {"cursor": {"modelName": "example-model"}},
                    },
                    {"type": "tool-call", "toolCallId": "read-id", "toolName": "Read", "args": {"path": "a.py"}},
                ],
            }
        )
        archived = blob({"role": "user", "content": "<user_query>Earlier question.</user_query>"})
        summary = blob({"role": "user", "content": "<summary>Earlier work.</summary>"})
        archive = blob(_field(1, archived) + _field(2, b"Earlier work.") + _field(4, summary))
        context = blob({"role": "user", "content": "<user_info>Environment context.</user_info>"})
        user = blob(
            {
                "role": "user",
                "content": [{"type": "text", "text": "<user_query>Read a.py.</user_query>"}],
                "providerOptions": {"cursor": {"requestId": "saved-request"}},
            }
        )
        answer = blob({"role": "assistant", "content": [{"type": "text", "text": "I read it."}]})
        blob({"role": "assistant", "content": "Unreachable checkpoint text."})
        root = b"".join(_field(1, ref) for ref in (context, summary, user, call, result, answer))
        prompt_detail = blob(_int_field(25, _START))
        read_step = blob(_field(2, _field(57, b"read-id") + _int_field(59, _START + 1) + _int_field(60, _START + 2)))
        answer_step = blob(_field(1, _field(1, b"I read it.") + _int_field(2, _START + 3) + _int_field(3, _START + 4)))
        turn_detail = blob(
            _field(
                1,
                _field(1, prompt_detail) + _field(2, read_step) + _field(2, answer_step) + _field(3, b"saved-request"),
            )
        )
        root_id = blob(
            root
            + _field(13, archive)
            + _int_field(26, _START)
            + _field(8, turn_detail)
            + _field(5, _int_field(2, 200_000))
        )
        meta = {"agentId": "db-session", "latestRootBlobId": root_id.hex(), "blobEncryptionKey": "opaque-private-key"}
        db.execute("INSERT INTO meta VALUES ('0', ?)", (json.dumps(meta).encode().hex(),))
        db.commit()
        # Keep the writer open so the reader must see committed WAL frames.
        yield path, root_id.hex(), call.hex()


def test_cursor_store_uses_checkpoint_order_and_blob_provenance(cursor_store) -> None:
    path, checkpoint_id, call_blob_id = cursor_store
    (trace,) = iter_traces(path.parents[2])
    assert trace.agent is AgentKind.cursor and trace.root.session_id == "db-session"
    assert trace.provenance.source_format == "cursor_native" and trace.model == "example-model"
    assert "opaque-private-key" not in trace.model_dump_json()
    assert [turn.user_input.text for turn in trace.root.turns] == ["Earlier question.", "Read a.py."]
    assert [event.text for event in trace.root.events if event.kind is EventKind.compaction] == ["Earlier work."]
    assert [event.text for event in trace.root.events if event.kind is EventKind.assistant_msg] == ["I read it."]
    call = next(event for event in trace.root.events if event.kind is EventKind.tool_call)
    result = next(event for event in trace.root.events if event.kind is EventKind.tool_result)
    assert call.idx < result.idx and call.call_id == result.call_id == "read-id"
    assert result.ok and result.output == "hello from source"
    assert trace.root.file_state["/work/project/a.py"].ever_read
    assert call.source is not None and call.source.item_id == call_blob_id and call.source.block_ordinal == 1
    assert call.source.source_sha256 == checkpoint_id and call.source.line_number is None
    assert call.ts is not None and call.ts_end == result.ts
    assert trace.root.turns[0].model_config_active.model is None
    final = trace.root.turns[-1]
    assert final.status is TurnStatus.completed and final.final_message == "I read it."
    assert final.context_window_tokens == 200_000
    assert trace.capabilities.observed.op_timing is CapLevel.true
    assert trace.capabilities.observed.per_turn_model is CapLevel.partial
    assert trace_matches_options(trace, RunOptions(model_families=("example",)))


def test_cursor_sources_reach_diagnostics_and_memory(tmp_path: Path, cursor_store) -> None:
    path, _checkpoint_id, _call_id = cursor_store
    capture = _write_rows(tmp_path / "capture.jsonl", _conversation())
    outcome = CliRunner().invoke(
        app, ["analyze", str(tmp_path), "--semantic", "off", "--format", "json", "--jobs", "1"]
    )
    assert outcome.exit_code == 0, outcome.stderr
    report = json.loads(outcome.stdout)
    assert report["aggregate"]["total_traces"] == 2
    rows = {row["analysis"]["provenance"]["source_format"]: row["analysis"]["root"] for row in report["runs"]}
    assert rows["cursor_stream"]["tokens"] == {"input": 10, "output": 6, "cache_read": 8, "cache_write": None}
    assert all(value is None for value in rows["cursor_native"]["tokens"].values())
    bundle = prepare_inputs([path, capture])
    calls = [row for row in bundle["records"] if row["kind"] == "tool_call"]
    assert {row["tool_name"] for row in calls} == {"shell", "Read"}
    assert any(row["paths"] == ["/work/project/a.py"] and row["line_number"] is None for row in calls)


def test_invocation_invoice_keeps_response_billing_unknown(tmp_path: Path, cursor_store) -> None:
    path = _write_rows(tmp_path / "metered.jsonl", _conversation())
    (stream,) = iter_traces(path)
    config = load_config()
    assert stream.model
    config = config.model_copy(update={"pricing": {stream.model: PriceEntry(input=2, output=3, as_of="2026-11-22")}})
    views = build_cost_views(stream, config)
    assert views.invoice.amount == pytest.approx((10 * 2 + 6 * 3) / 1_000_000)
    assert not iter_billed_calls(stream.root) and views.per_response == []
    assert views.origin_attributed is not None and views.origin_attributed.amount is None
    (native,) = iter_traces(cursor_store[0])
    native.root.model = stream.model
    invoice = session_invoice(native.root, config)
    assert invoice.amount is None and invoice.pricing_status == "unknown"


def test_partial_output_keeps_only_new_text_and_recovers_terminal_answer(tmp_path: Path) -> None:
    def assistant(text: str, **fields) -> dict:
        return {"type": "assistant", "message": {"content": text}, **fields}

    init, user, _, _, _, start, complete, _, result = _conversation()
    rows = [
        init,
        user,
        assistant("I will ", timestamp_ms=_START),
        assistant("", timestamp_ms=_START + 1),
        assistant("check.", timestamp_ms=_START + 2),
        {"type": "interaction_query", "subtype": "request"},
        assistant("I will check.", timestamp_ms=_START + 3, model_call_id="response-1"),
        start,
        complete,
        assistant("I will check."),
        {**result, "result": "I will check.Done."},
    ]
    (trace,) = iter_traces(_write_rows(tmp_path / "partial.jsonl", rows))
    messages = [event for event in trace.root.events if event.kind is EventKind.assistant_msg]
    assert [event.text for event in messages] == ["I will check.", "Done."]
    assert messages[0].response_id == "response-1"
    assert trace.root.turns[0].final_message == "Done."

    rows = [init, user, assistant("Hel", timestamp_ms=_START), assistant("Hello"), {**result, "result": "Hello"}]
    (recovered,) = iter_traces(_write_rows(tmp_path / "recover.jsonl", rows))
    assert [event.text for event in recovered.root.events if event.kind is EventKind.assistant_msg] == ["Hello"]


def test_edit_and_failure_payloads_preserve_evidence(tmp_path: Path) -> None:
    init, user, _, _, _, start, complete, answer, terminal = _conversation()
    args = {"path": "a.py"}
    tool_name = "strReplaceToolCall"
    start = {**start, "tool_call": {tool_name: {"args": args}}}
    complete = {
        **complete,
        "tool_call": {
            tool_name: {
                "args": args,
                "result": {
                    "success": {"linesAdded": 2, "linesRemoved": 1, "diff": "+abc"},
                    "error": None,
                },
            }
        },
    }
    (trace,) = iter_traces(_write_rows(tmp_path / "edit.jsonl", [init, user, start, complete, answer, terminal]))
    call = next(event for event in trace.root.events if event.kind is EventKind.tool_call)
    assert call.op_kind == "file.edit" and call.path == "/work/project/a.py"
    assert call.lines_added == 2 and call.lines_removed == 1 and call.artifacts[0].diff_head == "+abc"
    failure = tool_result("shell", {"failure": {"exitCode": 2, "stdout": "", "stderr": "boom"}})
    assert failure.ok is False and failure.exit_code == 2 and failure.output == "boom"


def test_append_during_load_uses_one_extent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _conversation()
    path = _write_rows(tmp_path / "active.jsonl", rows[:-1])
    snapshot = path.read_bytes()
    original = _common._hash_prefix

    def append_after_hash(stream, size):
        digest = original(stream, size)
        with path.open("a") as append:
            append.write(json.dumps(rows[-1]) + "\n")
        return digest

    monkeypatch.setattr(_common, "_hash_prefix", append_after_hash)
    trace = cursor_stream.load_stream(path)
    assert trace.root.events[-1].kind is EventKind.assistant_msg
    assert trace.root.source_artifacts[0]["bytes"] == len(snapshot)
    assert trace.root.source_artifacts[0]["source_sha256"] == hashlib.sha256(snapshot).hexdigest()


def test_verified_prefix_and_overlapping_paths_load_once(tmp_path: Path) -> None:
    rows = _conversation()
    complete = _write_rows(tmp_path / "complete.jsonl", rows)
    partial = _write_rows(tmp_path / "partial.jsonl", rows[:6])
    partial.write_bytes(partial.read_bytes().removesuffix(b"\n"))
    (unit,) = discover_many([complete, tmp_path, complete])
    (trace,) = load_unit(unit)
    assert trace.trace_id == cursor_stream.load_stream(complete).trace_id
    call = next(event for event in trace.root.events if event.kind is EventKind.tool_call)
    assert any(ref.path == str(partial) for ref in call.alternate_sources)
    outcome = CliRunner().invoke(
        app, ["analyze", str(complete), str(tmp_path), "--semantic", "off", "--format", "json"]
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert json.loads(outcome.stdout)["aggregate"]["total_traces"] == 1


def test_appended_alias_uses_the_discovered_prefix(tmp_path: Path) -> None:
    rows = _conversation()
    main = _write_rows(tmp_path / "a.jsonl", rows[:-1])
    alias = _write_rows(tmp_path / "z.jsonl", rows[:-1])
    snapshot = alias.read_bytes()
    (unit,) = discover_many([main, alias])
    with alias.open("a") as target:
        target.write(json.dumps(rows[-1]) + "\n")
    (trace,) = load_unit(unit)
    artifact = next(a for a in trace.root.source_artifacts if a["source_path"] == str(alias))
    assert artifact["bytes"] == len(snapshot)
    assert artifact["source_sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert all(event.invocation_usage is None for event in trace.root.events)


def test_saved_and_stream_turn_count_once_with_both_coordinates(tmp_path: Path, cursor_store) -> None:
    db_path = cursor_store[0]
    init = {**_CURSOR_INIT, "session_id": "db-session"}
    capture = _write_rows(
        tmp_path / "run.jsonl",
        [
            init,
            {"type": "user", "message": {"content": "Read a.py."}},
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "read-id",
                "tool_call": {"readToolCall": {"args": {"path": "a.py"}}},
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "call_id": "read-id",
                "tool_call": {"readToolCall": {"result": {"success": {"content": "hello from source"}}}},
            },
            {
                "type": "result",
                "subtype": "success",
                "request_id": "saved-request",
                "result": "I read it.",
                "usage": {"inputTokens": 10, "outputTokens": 3},
            },
        ],
    )
    (unit,) = discover_many([capture, db_path])
    (trace,) = load_unit(unit)
    assert len(trace.root.turns) == 2
    assert trace.root.turns[-1].final_message == "I read it."
    assert sum(e.kind is EventKind.tool_call for e in trace.root.events) == 1
    assert sum(e.invocation_usage is not None for e in trace.root.events) == 1
    memory = prepare_inputs([capture, db_path])
    (call,) = [record for record in memory["records"] if record["kind"] == "tool_call"]
    assert {ref["path"] for ref in call["source_refs"]} == {str(capture), str(db_path)}


def test_store_keeps_later_identified_run_and_its_usage(tmp_path: Path, cursor_store) -> None:
    path, _, _ = cursor_store
    rows = [{**row, "session_id": "db-session"} for row in _conversation()]
    capture = _write_rows(tmp_path / "later.jsonl", rows)
    (unit,) = discover_many([path, capture])
    native, stream = load_unit(unit)
    assert native.provenance.source_format == "cursor_native"
    assert stream.provenance.source_format == "cursor_stream"
    assert stream.root.events[-1].invocation_usage is not None
    assert stream.root.turns[0].final_message == "Tests failed."
    calls = [row for row in prepare_inputs([path, capture])["records"] if row["kind"] == "tool_call"]
    assert {call["tool_name"] for call in calls} == {"Read", "shell"}


def test_repeated_reply_matches_by_occurrence(tmp_path: Path, cursor_store) -> None:
    path, checkpoint_id, call_id = cursor_store
    answer_record = {"role": "assistant", "content": [{"type": "text", "text": "I read it."}]}
    answer_id = hashlib.sha256(json.dumps(answer_record).encode()).digest()
    with closing(sqlite3.connect(path)) as db:
        old = db.execute("SELECT data FROM blobs WHERE id=?", (checkpoint_id,)).fetchone()[0]
        call_ref = _field(1, bytes.fromhex(call_id))
        new = old.replace(call_ref, _field(1, answer_id) + call_ref, 1)
        digest = hashlib.sha256(new).hexdigest()
        db.execute("INSERT INTO blobs VALUES (?,?)", (digest, new))
        meta = {"agentId": "db-session", "latestRootBlobId": digest}
        db.execute("UPDATE meta SET value=? WHERE key='0'", (json.dumps(meta).encode().hex(),))
        db.commit()
    answer = {"type": "assistant", "message": {"content": "I read it."}}
    capture = _write_rows(
        tmp_path / "repeated.jsonl",
        [
            {**_CURSOR_INIT, "session_id": "db-session"},
            {"type": "user", "message": {"content": "Read a.py."}},
            answer,
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "read-id",
                "tool_call": {"readToolCall": {"args": {"path": "a.py"}}},
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "call_id": "read-id",
                "tool_call": {"readToolCall": {"result": {"success": {"content": "hello from source"}}}},
            },
            answer,
            {"type": "result", "subtype": "success", "request_id": "saved-request", "result": "I read it.I read it."},
        ],
    )
    (unit,) = discover_many([path, capture])
    (trace,) = load_unit(unit)
    replies = [ev for ev in trace.root.events if ev.kind is EventKind.assistant_msg]
    assert [ev.text for ev in replies] == ["I read it.", "I read it."]
    assert [ev.phase for ev in replies] == [None, "final_answer"]


def test_copied_checkpoint_preserves_sqlite_coordinates(tmp_path: Path, cursor_store) -> None:
    original, _, _ = cursor_store
    copy = tmp_path / "copy" / "store.db"
    copy.parent.mkdir()
    with closing(sqlite3.connect(original)) as source, closing(sqlite3.connect(copy)) as target:
        source.backup(target)
    (unit,) = discover_many([original, copy])
    (trace,) = load_unit(unit)
    assert len(trace.root.turns) == 2
    assert {artifact["source_path"] for artifact in trace.root.source_artifacts} == {str(original), str(copy)}
    call = next(e for e in trace.root.events if e.kind is EventKind.tool_call)
    assert len(call.alternate_sources) == 1 and call.source is not None
    assert call.alternate_sources[0].item_id == call.source.item_id


@pytest.mark.parametrize("out_of_line", [False, True])
def test_corrupt_archive_keeps_native_history_and_links_child(cursor_store, out_of_line) -> None:
    path, checkpoint_id, _ = cursor_store
    with closing(sqlite3.connect(path)) as db:

        def blob(raw: bytes) -> bytes:
            digest = hashlib.sha256(raw).digest()
            db.execute("INSERT INTO blobs VALUES (?,?)", (digest.hex(), raw))
            return digest

        main = db.execute("SELECT data FROM blobs WHERE id=?", (checkpoint_id,)).fetchone()[0]
        corrupt = blob(b"\x0a\xff")
        child_message = blob(json.dumps({"role": "user", "content": "Handle the delegated step."}).encode())
        child_state = _field(1, _field(1, child_message)) + _int_field(2, _START)
        if out_of_line:
            child_state = blob(child_state)
        child_field = _field(31 if out_of_line else 16, _field(1, b"child-id") + _field(2, child_state))
        checkpoint = blob(main + _field(13, corrupt) + child_field)
        meta = {"agentId": "db-session", "latestRootBlobId": checkpoint.hex()}
        db.execute("UPDATE meta SET value=? WHERE key='0'", (json.dumps(meta).encode().hex(),))
        db.commit()

    (trace,) = iter_traces(path)
    assert trace.root.turns[-1].final_message == "I read it."
    assert any(row["status"] == "unreadable_cursor_archive" for row in trace.root.source_diagnostics)
    (child,) = trace.subagents
    (link,) = trace.links
    assert child.turns[0].user_input.kind == "delegation"
    assert child.session_id == link.child_id == "child-id"
    assert child.parent_session_id == link.parent_id == "db-session"
    assert trace.capabilities.observed.thread_linkage is CapLevel.true


def test_cursor_codebench_package_preserves_result_and_scope(tmp_path: Path) -> None:
    path = tmp_path / "run"
    _write_rows(path / "stdout.jsonl", _conversation())
    (path / "run.json").write_text(json.dumps({"harness": "cursor", "run_id": "bench-cursor", "instance_id": "case"}))
    (path / "verification.json").write_text(json.dumps({"resolved": True}))
    (trace,) = iter_traces(path)
    assert trace.agent == AgentKind.cursor and trace.provenance.source_format == "codebench"
    assert trace.trace_id == "bench-cursor" and trace.instance_id == "case" and trace.resolved is True
    assert trace.root.events[-1].invocation_usage is not None and trace.root.events[-1].usage is None


def test_date_filter_reports_undated_cursor_stream(tmp_path: Path) -> None:
    rows = _conversation()
    path = _write_rows(tmp_path / "undated.jsonl", [rows[0], rows[1], rows[-2], rows[-1]])
    outcome = CliRunner().invoke(
        app, ["analyze", str(path), "--semantic", "off", "--format", "json", "--since", "2020-01-01"]
    )
    assert outcome.exit_code == 2
    assert "date filter excluded 1 trace(s) with no recorded timestamp" in outcome.stderr


def test_resumed_runs_with_equal_timestamps_get_distinct_evidence(tmp_path: Path) -> None:
    paths = []
    for number in (1, 2):
        rows = _conversation()
        rows[1] = {**rows[1], "message": {"content": f"Check part {number}."}}
        rows[-1] = {**rows[-1], "request_id": f"request-{number}"}
        paths.append(_write_rows(tmp_path / f"run-{number}.jsonl", rows))
    traces = list(iter_traces(tmp_path))
    assert len({trace.trace_id for trace in traces}) == 2
    assert len({trace.root.turns[0].turn_id for trace in traces}) == 2
    records = prepare_inputs(paths)["records"]
    assert {row["text"] for row in records if row["role"] == "user"} == {"Check part 1.", "Check part 2."}
