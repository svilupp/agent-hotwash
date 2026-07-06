"""Shared helpers for learning tests over real on-disk agent traces.

These tests are executable documentation of black-box trace formats. They locate
REAL sample files on this machine and assert structural claims about them. If a
sample path is missing (e.g. running on CI or another laptop) the test skips
gracefully rather than failing.

Nothing here mocks anything: every assertion is checked against bytes actually
written by codex / claude / pi.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

# Roots for the five formats. All absolute, machine-specific.
CODEBENCH_RUNS = Path("/Users/jan/Developer/window-shop-monorepo-clean/tools/code-bench/runs")
NATIVE_CLAUDE_PROJECT = Path("/Users/jan/.claude/projects/-Users-jan-Documents-GitHub-go-training-range-logfire-trace")
NATIVE_CODEX_SESSIONS = Path("/Users/jan/.codex/sessions")


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
