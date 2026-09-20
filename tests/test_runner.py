"""Parallel runner and work-unit discovery."""

from __future__ import annotations

import json
from pathlib import Path

from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.runner import RunOptions, resolve_jobs, run_paths, run_units
from agent_hotwash.sources.detect import WorkUnit, discover

FIXTURES = Path(__file__).parent / "fixtures"
TREE = FIXTURES / "codex_native" / "v0153" / "tree"
CLAUDE_RUN = FIXTURES / "codebench" / "claude_run"


# --------------------------------------------------------------------------- discovery


def test_discover_units_are_picklable_and_labelled() -> None:
    import pickle

    units = discover(TREE)
    assert units and all(isinstance(u, WorkUnit) for u in units)
    assert all(u.kind == "codex_tree" for u in units)
    roundtrip = pickle.loads(pickle.dumps(units))
    assert [u.label for u in roundtrip] == [u.label for u in units]
    assert all(u.size_bytes > 0 for u in units)


def test_resolve_jobs() -> None:
    assert resolve_jobs(0, 0) == 1
    assert resolve_jobs(0, 1) == 1
    assert resolve_jobs(8, 3) == 3
    assert resolve_jobs(2, 10) == 2
    assert resolve_jobs(0, 10) >= 1


# --------------------------------------------------------------------------- runner


def _strip_generated(report: Report) -> dict:
    data = json.loads(report.model_dump_json())
    data["meta"].pop("generated_at", None)
    return data


def test_run_paths_parallel_equals_sequential() -> None:
    seq_runs, seq_errors, _ = run_paths([TREE, CLAUDE_RUN], RunOptions(jobs=1))
    par_runs, par_errors, _ = run_paths([TREE, CLAUDE_RUN], RunOptions(jobs=2))
    assert not seq_errors and not par_errors
    assert len(seq_runs) == len(par_runs) == 2  # one codex tree + one code-bench run
    meta = ReportMeta(tool_version="t")
    assert _strip_generated(Report.build(seq_runs, meta)) == _strip_generated(Report.build(par_runs, meta))


def test_run_units_reports_unit_error_without_aborting(tmp_path: Path) -> None:
    good = discover(TREE)
    # A claude_session unit pointing at a missing file raises inside the loader.
    units = [*good, WorkUnit(kind="claude_session", paths=[tmp_path / "missing.jsonl"])]
    outcomes = run_units(units, RunOptions(jobs=1))
    assert len(outcomes) == len(units)
    assert all(o.error is None for o in outcomes[:-1])
    last = outcomes[-1]
    assert last.runs == [] and last.error is not None
    assert "missing.jsonl" in last.error.label
    assert last.error.error and last.error.traceback
