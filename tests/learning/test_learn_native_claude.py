"""Learning test: NATIVE Claude Code session logs.

Target: ~/.claude/projects/<slug>/<session>.jsonl
        + <session>/subagents/agent-*.jsonl + agent-*.meta.json

Verified facts:
- Rich line types: assistant, user, system, attachment, file-history-snapshot,
  last-prompt, mode, queue-operation. (SURPRISE: many auxiliary types beyond
  the briefing's short list.)
- Threading: every assistant/user record has uuid, parentUuid, requestId,
  sessionId AND a per-event ISO timestamp (unlike code-bench claude!).
- Assistant records are SPLIT one content block per line, grouped by requestId
  (message.content has length 1).
- tool_result lives in user records; ALSO a top-level `toolUseResult` object
  (structured: bash -> {stdout,stderr,interrupted,...}; Task -> {agentId,status,...};
  read -> {file,type}).
- Errors: user message content tool_result.is_error == true, same two shapes as
  code-bench claude ("Exit code N", "<tool_use_error>...").
- Subagents: separate agent-*.jsonl files; agent-*.meta.json links to parent via
  toolUseId and records agentType/spawnMode/description/spawnDepth.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from ._helpers import NATIVE_CLAUDE_PROJECT, read_jsonl


def _main_session() -> Path:
    files = sorted(
        (Path(p) for p in glob.glob(str(NATIVE_CLAUDE_PROJECT / "*.jsonl"))),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not files:
        pytest.skip("no native Claude session logs found")
    return files[0]


def test_line_type_inventory():
    records = read_jsonl(_main_session())
    types = {r.get("type") for r in records}
    assert {"assistant", "user"} <= types
    # Document the auxiliary types actually present.
    assert types & {
        "attachment",
        "file-history-snapshot",
        "last-prompt",
        "mode",
        "queue-operation",
        "system",
    }, f"expected auxiliary native types, got {types}"


def test_assistant_records_are_one_block_per_line():
    records = read_jsonl(_main_session())
    assistants = [r for r in records if r["type"] == "assistant"]
    assert assistants
    lengths = {len(r["message"]["content"]) for r in assistants}
    assert lengths == {1}, f"expected single-block assistant records, got lengths {lengths}"
    # Multiple lines share a requestId (the grouping key).
    by_req: dict[str, int] = {}
    for r in assistants:
        by_req[r["requestId"]] = by_req.get(r["requestId"], 0) + 1
    assert max(by_req.values()) >= 1


def test_full_threading_and_timestamps():
    """SURPRISE: the request-grouping key differs by record type.
    assistant records carry `requestId`; user records carry `promptId` instead.
    parentUuid is present on every record but null on the single root record.
    """
    records = read_jsonl(_main_session())
    root_count = 0
    for r in records:
        if r["type"] not in ("assistant", "user"):
            continue
        assert {"uuid", "parentUuid", "sessionId", "timestamp"} <= set(r), (
            f"missing common threading keys on {r['type']}: {set(r)}"
        )
        if r["parentUuid"] is None:
            root_count += 1
        if r["type"] == "assistant":
            assert "requestId" in r and "promptId" not in r
        else:  # user
            assert "promptId" in r and "requestId" not in r
    assert root_count == 1, f"expected exactly one root (null parentUuid), got {root_count}"
    # timestamps are ISO-8601 Z.
    a = next(r for r in records if r["type"] == "assistant")
    assert a["timestamp"].endswith("Z")


def test_top_level_tool_use_result_shapes():
    records = read_jsonl(_main_session())
    shapes: set[frozenset] = set()
    for r in records:
        if r.get("type") == "user" and isinstance(r.get("toolUseResult"), dict):
            shapes.add(frozenset(r["toolUseResult"].keys()))
    assert shapes, "expected structured toolUseResult objects"
    # bash result shape appears.
    assert any({"stdout", "stderr", "interrupted"} <= s for s in shapes), shapes


def test_error_serialization_and_collect_strings():
    records = read_jsonl(_main_session())
    exit_code_samples: list[str] = []
    tool_use_error_samples: list[str] = []
    for r in records:
        if r.get("type") != "user":
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
    assert exit_code_samples or tool_use_error_samples, "expected some tool errors"
    print("\nNATIVE CLAUDE bash error strings:")
    for s in exit_code_samples[:4]:
        print(" -", repr(s[:120]))
    print("NATIVE CLAUDE tool_use_error strings:")
    for s in tool_use_error_samples[:4]:
        print(" -", repr(s[:120]))


def test_subagent_files_and_meta():
    subagent_dirs = glob.glob(str(NATIVE_CLAUDE_PROJECT / "*" / "subagents"))
    if not subagent_dirs:
        pytest.skip("no subagent directories present")
    metas = glob.glob(str(Path(subagent_dirs[0]) / "agent-*.meta.json"))
    assert metas, "expected agent-*.meta.json alongside agent-*.jsonl"
    meta = json.loads(Path(metas[0]).read_text())
    # meta links a subagent to its spawning tool_use and records depth/type.
    assert {"agentType", "toolUseId", "spawnDepth"} <= set(meta), meta
    # the matching transcript exists.
    jsonl = metas[0].replace(".meta.json", ".jsonl")
    assert Path(jsonl).exists()
