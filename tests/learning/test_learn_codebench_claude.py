"""Learning test: code-bench CLAUDE stdout.jsonl (stream-json) format.

Target: runs/baseline-opus48-*/<task>/<attempt>/stdout.jsonl

Verified facts:
- Line types: system, assistant, user, result.
- system.subtype: init, PLUS (SURPRISE) thinking_tokens, task_started,
  task_notification — not in the briefing.
- assistant.message.content blocks: text | thinking | tool_use.
- user records carry tool_result blocks and a top-level `timestamp`;
  assistant records DO NOT have a timestamp (only uuid/request_id/session_id).
- Errors: tool_result.is_error == true; content is a string. Two shapes:
    * Bash failures start with "Exit code N\n..."
    * Framework errors are wrapped in <tool_use_error>...</tool_use_error>
- result: subtype success/error, is_error, usage (rich), total_cost_usd,
  permission_denials, modelUsage.
- Subagents: inline in the same stream via parent_tool_use_id (no separate file).
"""

from __future__ import annotations

from ._helpers import codebench_stdout_files, read_jsonl, require


def _first_file():
    files = codebench_stdout_files("baseline-opus48-*", limit=1)
    require(files, "code-bench claude")
    return files[0]


def test_line_type_inventory():
    records = read_jsonl(_first_file())
    types = {r["type"] for r in records}
    assert {"system", "assistant", "user", "result"} <= types, types


def test_system_subtypes_include_undocumented():
    records = read_jsonl(_first_file())
    subtypes = {r.get("subtype") for r in records if r["type"] == "system"}
    assert "init" in subtypes
    # SURPRISE: these extra system subtypes appear beyond plain init.
    assert subtypes & {"thinking_tokens", "task_started", "task_notification"}, subtypes


def test_assistant_content_block_types():
    records = read_jsonl(_first_file())
    blocks = {b["type"] for r in records if r["type"] == "assistant" for b in r["message"]["content"]}
    assert blocks <= {"text", "thinking", "tool_use"}, blocks


def test_only_user_records_have_timestamps():
    records = read_jsonl(_first_file())
    assistant_ts = [r for r in records if r["type"] == "assistant" and "timestamp" in r]
    user_ts = [r for r in records if r["type"] == "user" and "timestamp" in r]
    assert not assistant_ts, "assistant records unexpectedly gained a timestamp"
    assert user_ts, "user records should carry per-event timestamps"


def test_result_shape():
    records = read_jsonl(_first_file())
    result = next(r for r in records if r["type"] == "result")
    assert result["subtype"] in {"success", "error"}
    assert "is_error" in result
    assert "total_cost_usd" in result
    usage = result["usage"]
    assert {"input_tokens", "output_tokens", "cache_read_input_tokens"} <= set(usage)
    # permission_denials is present (empty list when none).
    assert "permission_denials" in result


def test_error_serialization_and_collect_strings():
    exit_code_samples: list[str] = []
    tool_use_error_samples: list[str] = []
    # <tool_use_error> framework errors are RARE (~5 across the whole opus48
    # corpus), so scan every run rather than a small sample.
    files = codebench_stdout_files("baseline-opus48-*")
    require(files, "claude codebench runs")
    for f in files:
        for r in read_jsonl(f):
            if r["type"] != "user":
                continue
            content = r["message"]["content"]
            if not isinstance(content, list):
                continue
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                    text = str(b.get("content", ""))
                    if text.startswith("<tool_use_error>"):
                        tool_use_error_samples.append(text[:200])
                    else:
                        exit_code_samples.append(text[:200])

    assert exit_code_samples, "expected Bash 'Exit code N' style errors"
    assert tool_use_error_samples, "expected <tool_use_error> framework errors"
    # Bash errors reliably lead with 'Exit code'.
    assert any(s.startswith("Exit code") for s in exit_code_samples)
    print("\nCLAUDE bash error strings:")
    for s in exit_code_samples[:4]:
        print(" -", repr(s[:120]))
    print("CLAUDE tool_use_error strings:")
    for s in tool_use_error_samples[:4]:
        print(" -", repr(s[:120]))


def test_subagents_via_parent_tool_use_id():
    """Subagents are inline: some records carry a non-null parent_tool_use_id."""
    found = False
    for f in codebench_stdout_files("baseline-opus48-*", limit=8):
        if any(r.get("parent_tool_use_id") for r in read_jsonl(f)):
            found = True
            break
    # Not every run spawns subagents; document that when present it's a toolu_ id.
    if not found:
        import pytest

        pytest.skip("no subagent activity in sampled claude runs")
