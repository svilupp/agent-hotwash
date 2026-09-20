"""Learning test: NATIVE Codex rollout logs (September 2026 / 0.150-0.155).

Target: ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl

September 2026 files (PLAN §2):
- Every line is {timestamp, type, payload}.
- First line is session_meta with id + cli_version.
- event_msg types include item_completed, token_count, task_started,
  task_complete (user_message / agent_message are no longer required).
- token_usage_record exists with usage + thread_token_usage.
- custom_tool_call name=exec whose input is a JavaScript script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._helpers import NATIVE_CODEX_SESSIONS


def _require_sessions() -> Path:
    if not NATIVE_CODEX_SESSIONS.is_dir():
        pytest.skip("no native Codex sessions directory")
    return NATIVE_CODEX_SESSIONS


def _september_rollouts() -> list[Path]:
    root = _require_sessions()
    files = sorted((root / "2026" / "09").rglob("rollout-*.jsonl")) if (root / "2026" / "09").is_dir() else []
    if not files:
        pytest.skip("no September 2026 Codex rollouts")
    return files


def _read_head(path: Path, limit: int = 4000) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _pick_rollout() -> tuple[Path, list[dict]]:
    files = _september_rollouts()
    sized = [p for p in files if 50_000 < p.stat().st_size < 40_000_000] or files
    sized.sort(key=lambda p: p.stat().st_size, reverse=True)
    for path in sized[:12]:
        records = _read_head(path)
        if records:
            return path, records
    pytest.skip("no readable September 2026 Codex rollouts")
    raise AssertionError


def test_every_line_has_timestamp_type_payload() -> None:
    _path, records = _pick_rollout()
    assert records
    for r in records:
        assert {"timestamp", "type", "payload"} <= set(r), f"drifted top-level keys: {set(r)}"


def test_session_meta_has_id_and_cli_version() -> None:
    _path, records = _pick_rollout()
    assert records[0]["type"] == "session_meta"
    payload = records[0]["payload"]
    assert "id" in payload
    assert "cli_version" in payload


def test_event_msg_types_include_v0153_surface() -> None:
    """Sep 2026 event_msg types; do not require user_message / agent_message."""
    files = _september_rollouts()
    seen: set[str] = set()
    for path in files[:20]:
        if path.stat().st_size > 80_000_000:
            continue
        records = _read_head(path, limit=8000)
        for r in records:
            if r.get("type") == "event_msg":
                payload = r.get("payload") or {}
                if isinstance(payload, dict) and payload.get("type"):
                    seen.add(str(payload["type"]))
        if {"item_completed", "token_count", "task_started", "task_complete"} <= seen:
            break
    assert "item_completed" in seen
    assert "token_count" in seen
    assert "task_started" in seen
    assert "task_complete" in seen


def test_token_usage_record_has_usage_and_thread_token_usage() -> None:
    files = _september_rollouts()
    found = None
    for path in files[:20]:
        if path.stat().st_size > 80_000_000:
            continue
        for r in _read_head(path, limit=4000):
            if r.get("type") == "token_usage_record":
                found = r.get("payload") or {}
                break
        if found:
            break
    if not found:
        pytest.skip("no token_usage_record in sampled September rollouts")
    assert "usage" in found
    assert "thread_token_usage" in found
    assert isinstance(found["usage"], dict)
    assert isinstance(found["thread_token_usage"], dict)


def test_custom_tool_call_exec_input_is_js_script() -> None:
    files = _september_rollouts()
    found = None
    for path in files[:20]:
        if path.stat().st_size > 80_000_000:
            continue
        for r in _read_head(path, limit=8000):
            payload = r.get("payload") or {}
            if (
                r.get("type") == "response_item"
                and isinstance(payload, dict)
                and payload.get("type") == "custom_tool_call"
                and payload.get("name") == "exec"
            ):
                found = payload
                break
        if found:
            break
    if not found:
        pytest.skip("no custom_tool_call name=exec in sampled September rollouts")
    inp = found.get("input")
    assert isinstance(inp, str)
    assert any(tok in inp for tok in ("tools.", "exec_command", "await ", "async ", "function "))
