"""Shared helpers for learning tests over real on-disk agent traces.

These tests are executable documentation of black-box trace formats. They locate
REAL sample files from optional ``HOTWASH_CODEBENCH_RUNS`` /
``HOTWASH_CLAUDE_PROJECT`` directories (plus ``~/.codex/sessions``) and assert
structural claims about them. If a sample path is missing (CI, another laptop)
the test skips rather than failing.

Nothing here mocks anything: every assertion is checked against bytes actually
written by codex / claude / pi.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest


def _env_dir(name: str) -> Path:
    """Optional on-disk sample root. Empty/unset → a path that never exists."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else Path("/nonexistent")


# Optional real-trace roots. Set HOTWASH_CODEBENCH_RUNS / HOTWASH_CLAUDE_PROJECT
# to exercise learning tests against local samples; CI leaves them unset.
CODEBENCH_RUNS = _env_dir("HOTWASH_CODEBENCH_RUNS")
NATIVE_CLAUDE_PROJECT = _env_dir("HOTWASH_CLAUDE_PROJECT")
NATIVE_CODEX_SESSIONS = Path.home() / ".codex" / "sessions"


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file, skipping blank/truncated trailing lines.

    Real traces occasionally end mid-write; we tolerate a final unparseable line
    but treat a parse error in the middle of the file as a real finding.
    """
    records: list[dict] = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # tolerate only a truncated final line
            if i == len(lines) - 1:
                continue
            raise
    return records


def codebench_stdout_files(run_glob: str, limit: int | None = None) -> list[Path]:
    """Find code-bench stdout.jsonl files matching a run-dir glob.

    Layout: runs/<run>/<task>/<attempt>/stdout.jsonl
    """
    pattern = str(CODEBENCH_RUNS / run_glob / "*" / "*" / "stdout.jsonl")
    files = sorted(Path(p) for p in glob.glob(pattern))
    if limit is not None:
        files = files[:limit]
    return files


def require(files: list[Path], what: str) -> None:
    if not files:
        pytest.skip(f"no real sample files found for {what}")
