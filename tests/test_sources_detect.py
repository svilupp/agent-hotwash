"""Auto-detection tests (fixture-based) + real-trace integration smoke tests.

The integration tests parse ONE real trace per format from the machine-specific
paths in the design brief; they skip gracefully when those paths are absent.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from agent_hotwash.events import AgentKind, EventKind, Trace
from agent_hotwash.sources.detect import iter_traces

FIXTURES = Path(__file__).parent / "fixtures"

# Real-trace roots (machine-specific; integration tests skip if missing).
CODEBENCH_RUNS = Path("/Users/jan/Developer/window-shop-monorepo-clean/tools/code-bench/runs")
NATIVE_CLAUDE_PROJECT = Path("/Users/jan/.claude/projects/-Users-jan-Documents-GitHub-go-training-range-logfire-trace")
NATIVE_CODEX_SESSIONS = Path("/Users/jan/.codex/sessions")


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
    tree = list(iter_traces(FIXTURES / "codex_native"))
    assert len(tree) == 1


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
