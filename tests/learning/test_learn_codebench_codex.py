"""Learning test: code-bench CODEX stdout.jsonl format.

Target: runs/baseline-codex-*/<task>/<attempt>/stdout.jsonl

Verified facts (see docs/research/format_findings.md):
- Line types: thread.started, turn.started, turn.completed, item.started, item.completed
- item.type: command_execution | file_change | agent_message
- Errors: command_execution.exit_code != 0 AND status == "failed";
  error text lives in item.aggregated_output.
- Usage: only on turn.completed (cumulative, single turn). Keys:
  input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens.
- SURPRISE: NO per-event timestamps anywhere in this format.
"""

from __future__ import annotations

from ._helpers import codebench_stdout_files, read_jsonl, require


def _first_file():
    files = codebench_stdout_files("baseline-codex-*", limit=1)
    require(files, "code-bench codex")
    return files[0]


def test_line_type_inventory():
    records = read_jsonl(_first_file())
    types = {r["type"] for r in records}
    assert {
        "thread.started",
        "turn.started",
        "turn.completed",
        "item.started",
        "item.completed",
    } >= types, f"unexpected extra types: {types}"

    item_types = {r["item"]["type"] for r in records if "item" in r}
    assert item_types <= {
        "command_execution",
        "file_change",
        "agent_message",
    }, f"unexpected item.type values: {item_types}"


def test_thread_started_has_thread_id():
    records = read_jsonl(_first_file())
    started = next(r for r in records if r["type"] == "thread.started")
    assert "thread_id" in started
    assert started["thread_id"].count("-") == 4  # uuid-ish


def test_turn_completed_usage_shape():
    records = read_jsonl(_first_file())
    completed = [r for r in records if r["type"] == "turn.completed"]
    assert len(completed) == 1, "expected exactly one turn per codex run"
    usage = completed[0]["usage"]
    assert set(usage) == {
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    }, f"usage keys drifted: {set(usage)}"


def test_command_execution_completed_fields():
    records = read_jsonl(_first_file())
    cmds = [
        r["item"]
        for r in records
        if r["type"] == "item.completed" and r.get("item", {}).get("type") == "command_execution"
    ]
    assert cmds
    sample = cmds[0]
    assert {"id", "command", "aggregated_output", "exit_code", "status"} <= set(sample)


def test_no_per_event_timestamps():
    """SURPRISE vs other formats: codex code-bench has zero timestamps."""
    records = read_jsonl(_first_file())
    assert not any("timestamp" in r for r in records), "codex code-bench unexpectedly has timestamps now"


def test_error_serialization_and_collect_strings():
    """Failed commands: exit_code != 0, status == 'failed', text in aggregated_output."""
    error_samples: list[str] = []
    for f in codebench_stdout_files("baseline-codex-*", limit=6):
        for r in read_jsonl(f):
            item = r.get("item", {})
            if item.get("type") == "command_execution" and item.get("exit_code") not in (0, None):
                assert item["status"] == "failed", f"nonzero exit but status={item['status']}"
                out = str(item.get("aggregated_output", ""))
                if out.strip():
                    error_samples.append(out[:200])

    # We expect at least a few real failures across the sampled runs.
    assert error_samples, "expected some failed commands across codex runs"
    print("\nCODEX real error strings (aggregated_output):")
    for s in error_samples[:6]:
        print(" -", repr(s[:160]))
