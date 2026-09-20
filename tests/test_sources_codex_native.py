"""Tests for the native Codex rollout parser."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.events import AgentKind, EventKind, ToolCategory
from agent_hotwash.sources.codex_legacy import _response_item
from agent_hotwash.sources.codex_native import load_rollout, looks_like_native_codex


def test_ok_trusts_exit_code_over_error_substring() -> None:
    # exit code 0 present -> ok, even if the output text mentions "Error:".
    ok0 = _response_item(
        {"type": "function_call_output", "call_id": "c", "output": "Error: handled gracefully\nexited with code 0"},
        None,
    )
    assert ok0 is not None and ok0.ok is True
    # nonzero exit code present -> not ok.
    bad = _response_item(
        {"type": "function_call_output", "call_id": "c", "output": "boom\nexited with code 2"},
        None,
    )
    assert bad is not None and bad.ok is False
    # no exit code -> fall back to the "Error:" substring heuristic.
    no_code = _response_item({"type": "function_call_output", "call_id": "c", "output": "Error: no code here"}, None)
    assert no_code is not None and no_code.ok is False


ROLLOUT = Path(__file__).parent / "fixtures" / "codex_native" / "rollout-fixture.jsonl"


def test_detects_native_codex() -> None:
    assert looks_like_native_codex(ROLLOUT)


def test_session_meta_id_and_model() -> None:
    trace = load_rollout(ROLLOUT)
    assert trace.agent is AgentKind.codex
    # session id key is `id` on session_meta; model comes from turn_context.
    assert trace.root.session_id == "dddddddd-0000-0000-0000-000000000004"
    assert trace.model == "gpt-5.5"
    assert trace.root.has_timestamps


def test_messages_from_event_msg() -> None:
    trace = load_rollout(ROLLOUT)
    kinds = [e.kind for e in trace.root.events]
    assert EventKind.user_msg in kinds
    assert EventKind.assistant_msg in kinds
    assert EventKind.thinking in kinds  # reasoning


def test_function_call_args_parsed_and_linked() -> None:
    trace = load_rollout(ROLLOUT)
    calls = [e for e in trace.root.events if e.kind is EventKind.tool_call]
    results = [e for e in trace.root.events if e.kind is EventKind.tool_result]
    exec_call = next(e for e in calls if e.call_id == "call_1")
    assert exec_call.tool_args == {"cmd": "ls"}  # arguments JSON string parsed
    assert len(results) == len(calls)  # no double-counted patch_apply_end


def test_native_tools_are_categorized() -> None:
    # Native codex names (exec_command / apply_patch) must map to the same coarse
    # categories the claude/pi decoders use, so file-op metrics and detectors fire.
    trace = load_rollout(ROLLOUT)
    calls = [e for e in trace.root.events if e.kind is EventKind.tool_call]
    exec_call = next(e for e in calls if e.call_id == "call_1")
    patch_call = next(e for e in calls if e.call_id == "call_3")
    assert exec_call.tool_category is ToolCategory.execute
    assert patch_call.tool_category is ToolCategory.write
    # apply_patch path is recovered from the patch body header line.
    assert patch_call.path == "app.py"
    assert patch_call.tool_args.get("paths") == ["app.py"]
    assert "app.py" in trace.root.file_state


def test_error_recovered_from_output_string() -> None:
    trace = load_rollout(ROLLOUT)
    failed = [e for e in trace.root.events if e.kind is EventKind.tool_result and e.ok is False]
    assert len(failed) == 1
    assert failed[0].call_id == "call_2"
    assert failed[0].exit_code == 1
    assert failed[0].error_category == "file_not_found"


def test_compaction_and_usage() -> None:
    trace = load_rollout(ROLLOUT)
    assert any(e.kind is EventKind.compaction for e in trace.root.events)
    # Usage comes from the cumulative `total_token_usage`, de-cumulated to deltas.
    # `input` excludes cache (codex input_tokens is cache-inclusive): with
    # input_tokens=5000, cached=4000 -> input delta 1000, cache_read delta 4000.
    usages = [e.usage for e in trace.root.events if e.usage]
    total_in = sum(u.input for u in usages if u.input)
    total_cache = sum(u.cache_read for u in usages if u.cache_read)
    total_out = sum(u.output for u in usages if u.output)
    assert total_in == 1000
    assert total_cache == 4000
    # input + cache_read + output reconstructs the reported total_tokens (5300).
    assert total_in + total_cache + total_out == 5300
