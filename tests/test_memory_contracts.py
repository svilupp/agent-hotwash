"""Independent plan acceptance for shared source evidence and thin projection."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from agent_hotwash.memory.extraction import eligible
from agent_hotwash.memory.inputs import prepare_inputs
from agent_hotwash.sources.detect import discover_many, load_unit

FIXTURES = Path(__file__).parent / "fixtures"
HARNESS_CASES = [
    "claude_native/proj/sess-fixture.jsonl",
    "codex_native/rollout-fixture.jsonl",
    "pi_native/session-fixture.jsonl",
    "codebench/claude_run",
    "codebench/codex_run",
    "codebench/pi_run_zerousage",
]


@pytest.mark.parametrize("relative", HARNESS_CASES)
def test_six_harnesses_have_canonical_physical_source_coordinates(relative):
    source = FIXTURES / relative
    traces = [trace for unit in discover_many([source]) for trace in load_unit(unit)]
    assert traces
    seen = 0
    for trace in traces:
        for session in (trace.root, *trace.subagents):
            for event in session.events:
                if event.source is None or event.source.path is None:
                    continue
                ref = event.source
                assert ref.path is not None
                actual = Path(ref.path)
                assert actual.is_file()
                assert ref.source_sha256 == hashlib.sha256(actual.read_bytes()).hexdigest()
                assert ref.line_number is not None and ref.line_number >= 1
                physical = actual.read_text().splitlines()[ref.line_number - 1]
                json.loads(physical)
                if ref.item_id:
                    assert ref.item_id in physical
                assert ref.record_index >= 0
                seen += 1
    assert seen
    prepared = prepare_inputs([source])
    assert prepared["schema_version"] == "memory-evidence-v2"
    assert prepared["sources"]
    assert all(session["capabilities"] for session in prepared["sessions"])
    assert all(record["source_path"] and record["line_number"] for record in prepared["records"])
    assert len({r["id"] for r in prepared["records"]}) == len(prepared["records"])


@pytest.mark.parametrize("relative", HARNESS_CASES)
def test_six_harnesses_preserve_bounded_captured_tool_edges_before_projection(tmp_path, relative):
    source = FIXTURES / relative
    destination = tmp_path / source.name
    if source.is_dir():
        shutil.copytree(source, destination)
        streams = list(destination.rglob("*.jsonl"))
    else:
        shutil.copyfile(source, destination)
        streams = [destination]
    output = "A" * 1000 + "middle" * 10_000 + "Z" * 1000
    altered = 0
    for stream in streams:
        updated = []
        for line in stream.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                updated.append(line)
                continue
            payload = event.get("payload", {})
            item = event.get("item", {})
            message = event.get("message", {})
            if payload.get("type") == "function_call_output":
                payload["output"] = output
                altered += 1
            if item.get("type") == "command_execution" and event.get("type") == "item.completed":
                item["aggregated_output"] = output
                altered += 1
            if event.get("type") == "tool_execution_end":
                event["result"]["content"] = [{"type": "text", "text": output}]
                altered += 1
            if message.get("role") == "toolResult":
                message["content"] = [{"type": "text", "text": output}]
                altered += 1
            for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                if block.get("type") == "tool_result":
                    block["content"] = output
                    altered += 1
            updated.append(json.dumps(event))
        stream.write_text("\n".join(updated) + "\n")
    assert altered
    traces = [trace for unit in discover_many([destination]) for trace in load_unit(unit)]
    captured = [
        event
        for trace in traces
        for session in (trace.root, *trace.subagents)
        for event in session.events
        if event.output_chars_original == len(output)
    ]
    assert captured
    for event in captured:
        assert event.output_excerpt is not None
        assert event.output_excerpt.startswith("A" * 1000)
        assert event.output_excerpt.endswith("Z" * 1000)
        assert "middle" not in event.output_excerpt
        assert event.output_sha256 == hashlib.sha256(output.encode()).hexdigest()
    projected = [
        r
        for r in prepare_inputs([destination])["records"]
        if r["kind"] == "tool_result" and r["original_chars"] == len(output)
    ]
    assert projected
    for record in projected:
        assert record["text"].startswith("A" * 1000) and record["text"].endswith("Z" * 1000)
        assert record["output_sha256"] == hashlib.sha256(output.encode()).hexdigest()
        assert record["truncated"]


def test_memory_source_projection_uses_no_raw_harness_decoder_or_private_source_import():
    package = Path(__file__).parents[1] / "src" / "agent_hotwash" / "memory"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agent_hotwash.sources"):
                assert node.module == "agent_hotwash.sources.detect", (path, node.module)
                assert all(not alias.name.startswith("_") for alias in node.names)
        if path.name == "inputs.py":
            text = path.read_text()
            assert "read_bytes(" not in text and "read_text(" not in text
            assert "json.loads(" not in text and "read_paths(" not in text


def test_unknown_delegated_and_injected_authorship_cannot_screen_as_user_correction():
    for authorship in ("unknown", "delegated", "injected"):
        assert not eligible(
            {"id": "e", "role": "user", "kind": "message", "authorship": authorship, "text": "No, use shared adapters."}
        )


def _native(path, session_id, text, **metadata):
    rows = [
        {
            "timestamp": "2026-09-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": "/repo", "cli_version": "0.153.4", **metadata},
        },
        {
            "timestamp": "2026-09-01T00:00:01Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": text},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_public_discovery_groups_cross_argument_child_and_missing_parent_stays_unknown(tmp_path):
    parent = _native(tmp_path / "rollout-parent.jsonl", "parent", "Found parent repository rule.")
    child = _native(
        tmp_path / "rollout-child.jsonl",
        "child",
        "Found child repository rule.",
        parent_thread_id="parent",
        thread_source="subagent",
    )
    together = prepare_inputs([parent, child])
    assert len(together["sessions"]) == 2
    assert len({session["independence_group_id"] for session in together["sessions"]}) == 1
    assert all(session["independence_confidence"] == "known" for session in together["sessions"])
    only_child = prepare_inputs([child])
    assert all(session["independence_confidence"] == "unknown" for session in only_child["sessions"])


def test_fork_and_parent_are_one_family_but_unrelated_sessions_are_two(tmp_path):
    parent = _native(tmp_path / "rollout-parent.jsonl", "parent", "Found shared repository rule.")
    fork = _native(
        tmp_path / "rollout-fork.jsonl",
        "fork",
        "Found new repository rule.",
        thread_source="agent_forked_thread",
        forked_from_id="parent",
        history_base={"thread_id": "parent", "end_ordinal_exclusive": 1},
    )
    unrelated = _native(tmp_path / "rollout-other.jsonl", "other", "Found unrelated repository rule.")
    together = prepare_inputs([parent, fork, unrelated])
    assert together["analytics"]["independent_group_count"] == 2
    groups = {session["session_id"]: session["independence_group_id"] for session in together["sessions"]}
    assert groups["codex:parent"] == groups["codex:fork"] != groups["codex:other"]


def test_identical_distinct_events_survive_and_snapshots_preserve_alternate_provenance(tmp_path):
    first = _native(tmp_path / "rollout-first.jsonl", "same", "Found the shared rule.")
    with first.open("a") as stream:
        stream.write(
            json.dumps({"type": "event_msg", "payload": {"type": "agent_message", "message": "Found the shared rule."}})
            + "\n"
        )
    duplicate = tmp_path / "rollout-duplicate.jsonl"
    shutil.copyfile(first, duplicate)
    alone = prepare_inputs([first])
    together = prepare_inputs([first, duplicate, tmp_path])
    assert len(alone["records"]) == 2
    assert len(together["records"]) == 2
    assert {r["id"] for r in alone["records"]} == {r["id"] for r in together["records"]}
    assert all(len(record["source_refs"]) == 2 for record in together["records"])


def test_proposal_recurrence_counts_known_families_and_allows_known_child_evidence():
    from agent_hotwash.memory.extraction import decode_candidates

    def record(identifier, session, group, confidence="known"):
        return {
            "id": identifier,
            "session_id": session,
            "role": "assistant",
            "kind": "reasoning",
            "text": "Found the shared parser rule.",
            "paths": [],
            "independence_group_id": group,
            "independence_confidence": confidence,
            "independent_session_known": confidence == "known",
        }

    def response():
        candidate = {
            "category": "repeated_discovery",
            "statement": "Reuse shared parser.",
            "applicability": "This repository",
            "evidence_ids": ["child", "other"],
            "paths": [],
            "intervention": "instruction",
            "draft": "Use shared parser.",
            "reason": "Repeated independently.",
        }
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": [candidate]})}]}]}

    child = record("child", "native:child", "family-parent")
    same_family = record("other", "native:parent", "family-parent")
    with pytest.raises(ValueError, match="independent"):
        decode_candidates(response(), [child, same_family])
    unrelated = record("other", "native:other", "family-other")
    result = decode_candidates(response(), [child, unrelated])[0]
    assert set(result["supporting_independence_groups"]) == {"family-parent", "family-other"}
    unrelated["independence_confidence"] = "unknown"
    with pytest.raises(ValueError, match="independent"):
        decode_candidates(response(), [child, unrelated])


def test_versioned_result_contract_rejects_unknown_status_and_schema():
    from pydantic import ValidationError

    from agent_hotwash.memory.models import AuditResult, EvidenceBundle

    with pytest.raises(ValidationError):
        AuditResult.model_validate({"schema_version": "memory-audit-v2", "status": "fictional_success"})
    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(
            {"schema_version": "memory-evidence-v1", "sessions": [], "records": [], "analytics": {}}
        )


@pytest.mark.parametrize("second_text", ["First observation.", "Second observation."])
def test_pi_multiple_text_blocks_preserve_distinct_evidence(tmp_path, second_text):
    path = tmp_path / "pi.jsonl"
    rows = [
        {"type": "session", "version": 3, "id": "pi-multi", "cwd": "/repo"},
        {
            "type": "message",
            "id": "message1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "First observation."},
                    {"type": "image", "data": "ignored"},
                    {"type": "text", "text": second_text},
                ],
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trace = next(load_unit(discover_many([path])[0]))
    text_events = [event for event in trace.root.events if event.text]
    assert [event.source.block_ordinal for event in text_events if event.source] == [0, 2]
    prepared = prepare_inputs([path])
    assert [record["text"] for record in prepared["records"]] == ["First observation.", second_text]
    assert len({record["id"] for record in prepared["records"]}) == 2


@pytest.mark.parametrize("duplicate_child", [False, True])
def test_duplicate_claude_parents_preserve_child_inventory_and_provenance(tmp_path, duplicate_child):
    first, second = tmp_path / "a", tmp_path / "b"
    for target in (first, second):
        shutil.copytree(FIXTURES / "claude_native/proj", target)
    if not duplicate_child:
        shutil.rmtree(first / "sess-fixture/subagents")
    traces = [trace for unit in discover_many([first, second]) for trace in load_unit(unit)]
    assert len(traces) == 1
    assert len(traces[0].subagents) == 1
    child = traces[0].subagents[0]
    assert len(child.events) == 3
    expected_paths = {str(second / "sess-fixture/subagents/agent-sub-abc.jsonl")}
    if duplicate_child:
        expected_paths.add(str(first / "sess-fixture/subagents/agent-sub-abc.jsonl"))
    assert {artifact["source_path"] for artifact in child.source_artifacts} == expected_paths
    for event in child.events:
        assert {ref.path for ref in [event.source, *event.alternate_sources] if ref} == expected_paths
    prepared = prepare_inputs([first, second])
    assert len(prepared["records"]) == 12
    assert len(prepared["sessions"]) == 2


@pytest.mark.parametrize("divergent", [False, True])
def test_duplicate_claude_child_snapshots_do_not_double_count(tmp_path, divergent):
    first, second = tmp_path / "a", tmp_path / "b"
    for target in (first, second):
        shutil.copytree(FIXTURES / "claude_native/proj", target)
    child_path = second / "sess-fixture/subagents/agent-sub-abc.jsonl"
    rows = child_path.read_text().splitlines()
    if divergent:
        row = json.loads(rows[0])
        row["message"]["content"] = "A different task."
        rows[0] = json.dumps(row)
    else:
        rows.append(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "final",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "Verification complete."}]},
                }
            )
        )
    child_path.write_text("\n".join(rows) + "\n")
    units = discover_many([first, second])
    if divergent:
        with pytest.raises(ValueError, match="Conflicting Claude child exports"):
            list(load_unit(units[0]))
        return
    trace = next(load_unit(units[0]))
    assert len(trace.subagents) == 1
    child = trace.subagents[0]
    assert len(child.events) == 4
    assert child.events[-1].text == "Verification complete."
    assert len(child.events[0].alternate_sources) == 1
    assert child.events[-1].alternate_sources == []
    assert len(child.source_artifacts) == 2


def test_truncated_codex_alias_only_cites_complete_records(tmp_path):
    complete = tmp_path / "rollout-full.jsonl"
    short = tmp_path / "rollout-short.jsonl"
    raw = (FIXTURES / "codex_native/rollout-fixture.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    complete.write_bytes(raw)
    short.write_bytes(b"".join(lines[:3]) + lines[3][:10])
    traces = [trace for unit in discover_many([complete, short]) for trace in load_unit(unit)]
    assert len(traces) == 1
    sessions = [traces[0].root, *traces[0].subagents]
    refs = [ref for session in sessions for event in session.events for ref in event.alternate_sources]
    assert refs
    for ref in refs:
        assert ref.line_number is not None and ref.line_number <= 3
        json.loads(short.read_bytes().splitlines()[ref.line_number - 1])
    assert any(
        d["source_path"] == str(short) and d["line_number"] == 4
        for session in sessions
        for d in session.source_diagnostics
    )
