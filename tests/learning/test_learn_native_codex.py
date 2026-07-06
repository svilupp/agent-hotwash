"""Learning test: NATIVE Codex rollout logs.

Target: ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl

Verified facts:
- EVERY line is {timestamp, type, payload}. Top-level per-event ISO timestamp.
- Line types: session_meta, turn_context, response_item, event_msg, compacted.
- First line is session_meta.
- response_item payload.type: message | reasoning | function_call |
  function_call_output | custom_tool_call | custom_tool_call_output.
    * function_call: {name, call_id, arguments} where arguments is a JSON STRING.
    * function_call_output: {call_id, output} where output is a STRING
      (NOT structured) — contains "Process exited with code N" and "Error:".
    * custom_tool_call: e.g. apply_patch.
- event_msg payload.type: user_message, agent_message, token_count,
  task_started, task_complete, patch_apply_end, context_compacted.
    * token_count.info.total_token_usage + last_token_usage + model_context_window;
      plus rate_limits.
    * patch_apply_end: {call_id, turn_id, stdout, stderr, success, changes}.
- call_id matches function_call <-> function_call_output <-> patch_apply_end.
- Errors are NOT structured: exit codes/messages are embedded in the output STRING.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from ._helpers import NATIVE_CODEX_SESSIONS, read_jsonl


def _rollout() -> Path:
    files = sorted(Path(p) for p in glob.glob(str(NATIVE_CODEX_SESSIONS / "*" / "*" / "*" / "rollout-*.jsonl")))
    if not files:
        pytest.skip("no native Codex rollout logs found")
    # pick a large one for coverage
    return max(files, key=lambda p: p.stat().st_size)


def test_every_line_has_timestamp_type_payload():
    records = read_jsonl(_rollout())
    assert records
    for r in records:
        assert set(r) == {"timestamp", "type", "payload"}, f"drifted top-level keys: {set(r)}"
        assert r["timestamp"].endswith("Z"), "ISO-8601 Z per-event timestamp"


def test_first_line_is_session_meta():
    records = read_jsonl(_rollout())
    assert records[0]["type"] == "session_meta"
    payload = records[0]["payload"]
    # SURPRISE: the session id key is `id` here, NOT `session_id`.
    # (`session_id` only appears later in turn_context payloads.)
    assert {"id", "cwd", "cli_version"} <= set(payload), set(payload)


def test_top_level_type_inventory():
    records = read_jsonl(_rollout())
    types = {r["type"] for r in records}
    assert types <= {
        "session_meta",
        "turn_context",
        "response_item",
        "event_msg",
        "compacted",
    }, f"unexpected native codex types: {types}"


def test_response_item_payload_types():
    records = read_jsonl(_rollout())
    payload_types = {r["payload"]["type"] for r in records if r["type"] == "response_item"}
    assert payload_types <= {
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        # SURPRISE: native web search surfaces as its own response_item type.
        "web_search_call",
    }, payload_types


def test_function_call_arguments_are_json_string():
    records = read_jsonl(_rollout())
    calls = [r["payload"] for r in records if r["type"] == "response_item" and r["payload"]["type"] == "function_call"]
    assert calls
    sample = calls[0]
    assert {"name", "call_id", "arguments"} <= set(sample)
    assert isinstance(sample["arguments"], str)
    # it parses as JSON
    json.loads(sample["arguments"])


def test_function_call_output_is_plain_string():
    """SURPRISE: output is an unstructured string; exit code is embedded text."""
    records = read_jsonl(_rollout())
    outs = [
        r["payload"] for r in records if r["type"] == "response_item" and r["payload"]["type"] == "function_call_output"
    ]
    assert outs
    assert all(isinstance(o["output"], str) for o in outs)
    assert any("exited with code" in o["output"] for o in outs)


def test_call_id_links_calls_and_outputs():
    records = read_jsonl(_rollout())
    call_ids = {
        r["payload"]["call_id"]
        for r in records
        if r["type"] == "response_item" and r["payload"]["type"] == "function_call"
    }
    out_ids = {
        r["payload"]["call_id"]
        for r in records
        if r["type"] == "response_item" and r["payload"]["type"] == "function_call_output"
    }
    # outputs should reference calls we saw.
    assert out_ids & call_ids, "call_id should join function_call to its output"


def test_token_count_usage_shape():
    records = read_jsonl(_rollout())
    tcs = [r["payload"] for r in records if r["payload"].get("type") == "token_count"]
    assert tcs
    info = tcs[-1]["info"]
    assert {"total_token_usage", "last_token_usage", "model_context_window"} <= set(info)
    assert {"input_tokens", "output_tokens", "total_tokens"} <= set(info["total_token_usage"])


def test_patch_apply_end_shape_and_collect_errors():
    records = read_jsonl(_rollout())
    patches = [r["payload"] for r in records if r["payload"].get("type") == "patch_apply_end"]
    if patches:
        p = patches[0]
        assert {"call_id", "stdout", "stderr", "success"} <= set(p), set(p)

    # Collect real error strings embedded in command outputs.
    error_samples: list[str] = []
    for r in records:
        if r["type"] == "response_item" and r["payload"]["type"] == "function_call_output":
            out = r["payload"]["output"]
            if "exited with code 1" in out or "Error:" in out or "No such file" in out:
                error_samples.append(out[:200])
    print("\nNATIVE CODEX error strings (embedded in output):")
    for s in error_samples[:6]:
        print(" -", repr(s[:150]))
