"""Auto-detection tests (fixture-based) + real-trace integration smoke tests.

The integration tests parse ONE real trace per format from optional
``HOTWASH_*`` roots; they skip gracefully when those paths are absent.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

from agent_hotwash.events import AgentKind, EventKind, Trace
from agent_hotwash.sources.detect import iter_traces

FIXTURES = Path(__file__).parent / "fixtures"


def _env_dir(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else Path("/nonexistent")


CODEBENCH_RUNS = _env_dir("HOTWASH_CODEBENCH_RUNS")
NATIVE_CLAUDE_PROJECT = _env_dir("HOTWASH_CLAUDE_PROJECT")
NATIVE_CODEX_SESSIONS = Path.home() / ".codex" / "sessions"


def _linked_ok(trace: Trace) -> None:
    """Every tool_result references a tool_call we saw in the same session."""
    for session in [trace.root, *trace.subagents]:
        call_ids = {e.call_id for e in session.events if e.kind is EventKind.tool_call and e.call_id}
        result_ids = {e.call_id for e in session.events if e.kind is EventKind.tool_result and e.call_id}
        assert result_ids <= call_ids or not call_ids


def _usage_total(trace: Trace) -> int:
    total = 0
    for e in trace.root.events:
        if e.usage:
            total += sum(x for x in (e.usage.input, e.usage.output, e.usage.cache_read, e.usage.cache_write) if x)
    return total


# --------------------------------------------------------------------------- detection


def test_detect_codebench_run_dir() -> None:
    traces = list(iter_traces(FIXTURES / "codebench" / "codex_run"))
    assert len(traces) == 1
    assert traces[0].provenance.source_format == "codebench"


def test_detect_native_claude_file() -> None:
    traces = list(iter_traces(FIXTURES / "claude_native" / "proj" / "sess-fixture.jsonl"))
    assert len(traces) == 1
    assert traces[0].agent is AgentKind.claude


def test_detect_native_claude_project_dir() -> None:
    traces = list(iter_traces(FIXTURES / "claude_native" / "proj"))
    assert len(traces) == 1


def test_detect_native_codex_file_and_tree() -> None:
    one = list(iter_traces(FIXTURES / "codex_native" / "rollout-fixture.jsonl"))
    assert len(one) == 1 and one[0].agent is AgentKind.codex
    # The whole dir is indexed as ONE forest (rollouts in nested date dirs link
    # across days): the legacy fixture, the v0153 orchestrator tree (children
    # attached), and the fork child whose thread id collides with the legacy
    # fixture (kept standalone, never linked).
    tree = list(iter_traces(FIXTURES / "codex_native"))
    roots = {t.trace_id for t in tree}
    assert "bbbbbbbb-0000-0000-0000-000000000002" in roots
    assert sum(1 for t in tree if t.trace_id == "dddddddd-0000-0000-0000-000000000004") == 2
    assert all(t.agent is AgentKind.codex for t in tree)
    orchestrator = next(t for t in tree if t.trace_id == "bbbbbbbb-0000-0000-0000-000000000002")
    assert {s.session_id for s in orchestrator.subagents} >= {
        "cccccccc-0000-0000-0000-000000000003",
        "eeeeeeee-0000-0000-0000-000000000005",
    }


def test_codex_forest_links_across_date_dirs(tmp_path: Path) -> None:
    """Parent and child rollouts on different calendar days still form one tree."""
    import shutil

    tree = FIXTURES / "codex_native" / "v0153" / "tree"
    (tmp_path / "2026" / "09" / "01").mkdir(parents=True)
    (tmp_path / "2026" / "09" / "02").mkdir(parents=True)
    shutil.copy(tree / "rollout-orchestrator.jsonl", tmp_path / "2026" / "09" / "01" / "rollout-orchestrator.jsonl")
    shutil.copy(tree / "rollout-child-created.jsonl", tmp_path / "2026" / "09" / "02" / "rollout-child-created.jsonl")
    traces = list(iter_traces(tmp_path))
    assert len(traces) == 1
    assert [lk.kind.value for lk in traces[0].links] == ["created"]
    assert traces[0].links[0].evidence == ["delegation.source_thread_id"]
    assert traces[0].provenance.thread_linkage == "full"


def test_codebench_run_dir_internal_rollout_not_promoted(tmp_path: Path) -> None:
    """A code-bench run dir keeps a full codex rollout copy under
    ``traces/codex/sessions/…``. Detecting the experiment dir must yield exactly
    one code-bench Trace per run (parsed from stdout.jsonl), never a second
    ``codex_native`` Trace from the internal rollout copy."""
    import shutil

    exp = tmp_path / "baseline-codex-exp"
    run_dir = exp / "inst-1" / "run-1"
    run_dir.parent.mkdir(parents=True)
    shutil.copytree(FIXTURES / "codebench" / "codex_run", run_dir)
    nested = run_dir / "traces" / "codex" / "sessions" / "2026" / "06" / "30"
    nested.mkdir(parents=True)
    shutil.copy(FIXTURES / "codex_native" / "rollout-fixture.jsonl", nested / "rollout-copy.jsonl")

    traces = list(iter_traces(exp))
    assert len(traces) == 1
    assert traces[0].provenance.source_format == "codebench"
    assert traces[0].agent is AgentKind.codex


def test_detect_missing_path_yields_nothing() -> None:
    assert list(iter_traces(FIXTURES / "does-not-exist")) == []


def test_skip_suffixed_dirs(tmp_path: Path) -> None:
    for suffix in (".quarantine", ".interrupted", ".crashed-1"):
        (tmp_path / f"run{suffix}").mkdir()
    assert list(iter_traces(tmp_path / "run.quarantine")) == []
    # a container holding only skip dirs yields nothing.
    assert list(iter_traces(tmp_path)) == []


# --------------------------------------------------------------------------- integration


def _first_run_dir(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(CODEBENCH_RUNS / pattern / "*" / "*" / "run.json")))
    return Path(hits[0]).parent if hits else None


@pytest.mark.parametrize(
    ("pattern", "agent"),
    [
        ("baseline-codex-*", AgentKind.codex),
        ("baseline-opus48-*", AgentKind.claude),
        ("baseline-pi-glm52-xhigh", AgentKind.pi),
    ],
)
def test_integration_codebench(pattern: str, agent: AgentKind) -> None:
    run_dir = _first_run_dir(pattern)
    if run_dir is None:
        pytest.skip(f"no real code-bench run for {pattern}")
    traces = list(iter_traces(run_dir))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.agent is agent
    assert len(trace.root.events) > 0
    assert any(e.kind is EventKind.tool_call for e in trace.root.events)
    _linked_ok(trace)
    assert _usage_total(trace) > 0


def test_integration_native_claude() -> None:
    if not NATIVE_CLAUDE_PROJECT.is_dir():
        pytest.skip("no native Claude project dir")
    files = sorted(NATIVE_CLAUDE_PROJECT.glob("*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        pytest.skip("no native Claude session files")
    traces = list(iter_traces(files[0]))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.agent is AgentKind.claude
    assert len(trace.root.events) > 0
    _linked_ok(trace)
    assert _usage_total(trace) > 0


def test_integration_native_codex() -> None:
    hits = sorted(glob.glob(str(NATIVE_CODEX_SESSIONS / "2026" / "06" / "30" / "rollout-*.jsonl")))
    if not hits:
        hits = sorted(glob.glob(str(NATIVE_CODEX_SESSIONS / "*" / "*" / "*" / "rollout-*.jsonl")))
    if not hits:
        pytest.skip("no native Codex rollouts")
    target = max(hits, key=lambda p: Path(p).stat().st_size)
    traces = list(iter_traces(Path(target)))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.agent is AgentKind.codex
    assert len(trace.root.events) > 0
    assert any(e.kind is EventKind.tool_call for e in trace.root.events)
    _linked_ok(trace)
    assert _usage_total(trace) > 0
