"""Canonical audit adapter preserves source authorship and exact output edges."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_hotwash.memory.extraction import eligible
from agent_hotwash.memory.inputs import prepare_inputs

FIXTURES = Path(__file__).parent / "fixtures"


def test_native_snapshots_deduplicate_reads_without_losing_tail(tmp_path):
    source = FIXTURES / "codex_native" / "rollout-fixture.jsonl"
    first = tmp_path / "rollout-first.jsonl"
    second = tmp_path / "rollout-second.jsonl"
    first.write_bytes(source.read_bytes())
    second.write_bytes(source.read_bytes())
    baseline = prepare_inputs([first])
    together = prepare_inputs([first, second])
    assert len(together["records"]) == len(baseline["records"])
    assert together["analytics"]["most_opened_files"] == baseline["analytics"]["most_opened_files"]
    assert together["analytics"]["overlapping_snapshot_events_skipped"] > 0
    assert len(together["sessions"]) == 1


def test_native_claude_injected_context_never_qualifies_as_pure_user_correction():
    prepared = prepare_inputs([FIXTURES / "claude_native" / "proj"])
    injected = [r for r in prepared["records"] if "Caveat:" in r["text"]]
    # Native decoders may discard injected messages entirely or mark them meta.
    assert all(r["is_meta"] and not eligible(r) for r in injected)
    assert any(r["role"] == "user" and eligible(r) for r in prepared["records"])
    assert all(r["role"] == "tool" for r in prepared["records"] if r["kind"] == "tool_result")


def test_native_claude_recovers_long_tool_output_tail(tmp_path):
    events = [
        json.loads(line)
        for line in (FIXTURES / "claude_native" / "proj" / "sess-fixture.jsonl").read_text().splitlines()
    ]
    long_output = "A" * 1000 + "middle" * 10_000 + "Z" * 1000
    for event in events:
        content = event.get("message", {}).get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    block["content"] = long_output
    source = tmp_path / "sess-fixture.jsonl"
    source.write_text("".join(json.dumps(event) + "\n" for event in events))
    outputs = [r for r in prepare_inputs([source])["records"] if r["kind"] == "tool_result"]
    assert outputs
    for record in outputs:
        assert record["text"].startswith("A" * 1000)
        assert record["text"].endswith("Z" * 1000)
        assert "middle" not in record["text"]
        assert record["truncated"]


def test_unrecognized_sources_are_rejected_and_decoder_errors_do_not_fall_back(tmp_path, monkeypatch):
    import agent_hotwash.memory.inputs as adapter

    source = tmp_path / "unsupported.jsonl"
    source.write_text(json.dumps({"arbitrary": "not a supported source"}) + "\n")
    with pytest.raises(ValueError, match="No recognized trace sources"):
        prepare_inputs([source])

    def broken_decoder(unit):
        raise ValueError("decoder failure")

    monkeypatch.setattr(adapter, "load_unit", broken_decoder)
    with pytest.raises(ValueError, match="Recognized source decoder failed"):
        prepare_inputs([FIXTURES / "codex_native" / "rollout-fixture.jsonl"])


def test_mixed_native_directory_includes_claude_codex_and_pi(tmp_path):
    import shutil

    for source in (
        FIXTURES / "claude_native" / "proj" / "sess-fixture.jsonl",
        FIXTURES / "codex_native" / "rollout-fixture.jsonl",
        FIXTURES / "pi_native" / "session-fixture.jsonl",
    ):
        shutil.copyfile(source, tmp_path / source.name)
    prepared = prepare_inputs([tmp_path])
    formats = {session["source_format"] for session in prepared["sessions"]}
    assert {"claude_native", "codex_native", "pi_native"} <= formats
    assert len(prepared["records"]) > 10


def test_codebench_pi_recovers_long_tool_output_tail(tmp_path):
    import shutil

    source = tmp_path / "pi-run"
    shutil.copytree(FIXTURES / "codebench" / "pi_run_zerousage", source)
    stream = source / "stdout.jsonl"
    events = []
    for line in stream.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    long_output = "A" * 1000 + "middle" * 10_000 + "Z" * 1000
    for event in events:
        if event.get("type") == "tool_execution_end":
            event["result"]["content"] = [{"type": "text", "text": long_output}]
    stream.write_text("".join(json.dumps(event) + "\n" for event in events))
    outputs = [r for r in prepare_inputs([source])["records"] if r["kind"] == "tool_result"]
    assert outputs
    for record in outputs:
        assert record["text"].startswith("A" * 1000)
        assert record["text"].endswith("Z" * 1000)
        assert "middle" not in record["text"]
        assert record["truncated"]


def test_native_codex_preserves_recorded_plaintext_reasoning():
    prepared = prepare_inputs([FIXTURES / "codex_native" / "rollout-fixture.jsonl"])
    reasoning = [r for r in prepared["records"] if r["kind"] == "reasoning"]
    assert any(r["text"] == "I should inspect the repo." for r in reasoning)
    assert prepared["sessions"][0]["plaintext_reasoning_records"] >= 1
    assert prepared["sessions"][0]["reasoning_visibility"].startswith("plaintext")


def test_attached_claude_child_preserves_tail_and_source_path(tmp_path):
    import shutil

    root = tmp_path / "proj"
    shutil.copytree(FIXTURES / "claude_native" / "proj", root)
    child = root / "sess-fixture" / "subagents" / "agent-sub-abc.jsonl"
    events = [json.loads(line) for line in child.read_text().splitlines()]
    for event in events:
        for block in event.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block["content"] = "A" * 1000 + "middle" * 10_000 + "Z" * 1000
    child.write_text("".join(json.dumps(event) + "\n" for event in events))
    prepared = prepare_inputs([root])
    output = next(r for r in prepared["records"] if r.get("native_id") == "toolu_sr1" and r["kind"] == "tool_result")
    assert output["text"].startswith("A" * 1000)
    assert output["text"].endswith("Z" * 1000)
    assert output["source_path"] == str(child.resolve())
    assert output["independence_confidence"] == "known"
    root_group = prepared["sessions"][0]["independence_group_id"]
    assert output["independence_group_id"] == root_group
    assert prepared["analytics"]["independent_group_count"] == 1


def test_inline_codebench_child_preserves_tail_and_source_path(tmp_path):
    import shutil

    root = tmp_path / "run"
    shutil.copytree(FIXTURES / "codebench" / "claude_run", root)
    stream = root / "stdout.jsonl"
    events = [json.loads(line) for line in stream.read_text().splitlines()]
    for event in events:
        if event.get("parent_tool_use_id"):
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    block["content"] = "A" * 1000 + "middle" * 10_000 + "Z" * 1000
    stream.write_text("".join(json.dumps(event) + "\n" for event in events))
    prepared = prepare_inputs([root])
    output = next(r for r in prepared["records"] if r.get("native_id") == "toolu_s1" and r["kind"] == "tool_result")
    assert output["text"].startswith("A" * 1000)
    assert output["text"].endswith("Z" * 1000)
    assert output["source_path"] == str(stream.resolve())
    assert output["independence_confidence"] == "known"
    root_group = prepared["sessions"][0]["independence_group_id"]
    assert output["independence_group_id"] == root_group
    assert prepared["analytics"]["independent_group_count"] == 1
