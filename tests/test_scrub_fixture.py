"""Fixture scrubber CLI (PLAN §9.3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scrub_fixture.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_scrubber_strips_email_and_users_path_keeps_sentinels(tmp_path: Path) -> None:
    dirty = tmp_path / "in.jsonl"
    dirty.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-01T00:00:00.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-0001",
                    "item": {
                        "type": "UserMessage",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "please email alice@corp.example from /Users/alice/src/app.py "
                                    "after item_completed and Script completed"
                                ),
                            }
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    result = _run([str(dirty), "-o", str(out)])
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "alice@corp.example" not in text
    assert "/Users/alice" not in text
    assert "item_completed" in text
    assert "UserMessage" in text
    assert "Script completed" in text
    rec = json.loads(text.splitlines()[0])
    assert rec["timestamp"] == "2026-09-01T00:00:00.000Z"
    assert rec["payload"]["turn_id"] == "turn-0001"


def test_scrubber_canary_fails_when_string_survives(tmp_path: Path) -> None:
    dirty = tmp_path / "in.jsonl"
    dirty.write_text(
        json.dumps({"type": "event_msg", "payload": {"type": "item_completed", "note": "keep-me"}}),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    # Sentinel survives, so using it as a canary must fail.
    result = _run([str(dirty), "-o", str(out), "--canary", "item_completed"])
    assert result.returncode == 1
    assert "canary hit" in result.stderr

    # A secret that is scrubbed (email) is not a canary hit.
    dirty.write_text(
        json.dumps({"payload": {"text": "write to bob@example.com after item_completed"}}),
        encoding="utf-8",
    )
    result = _run([str(dirty), "-o", str(out), "--canary", "bob@example.com"])
    assert result.returncode == 0, result.stderr
    assert "bob@example.com" not in out.read_text(encoding="utf-8")
    assert "item_completed" in out.read_text(encoding="utf-8")
