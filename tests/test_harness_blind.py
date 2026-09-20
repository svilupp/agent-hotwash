"""AST harness-blindness: structure/semantic/diagnostics must not know sources."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src" / "agent_hotwash"
_DIRS = ("structure", "semantic", "diagnostics")
_FORBIDDEN_NAMES = frozenset({"AgentKind", "source_format"})


def _py_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.rglob("*.py") if p.name != "__pycache__"]


def _module_is_sources(mod: str | None) -> bool:
    if not mod:
        return False
    return (
        mod == "agent_hotwash.sources"
        or mod.startswith("agent_hotwash.sources.")
        or mod == "sources"
        or mod.startswith("sources.")
    )


@pytest.mark.parametrize("dirname", _DIRS)
def test_no_sources_import_or_forbidden_names(dirname: str) -> None:
    directory = _ROOT / dirname
    if not directory.exists():
        pytest.skip(f"{dirname}/ not present yet")
    files = _py_files(directory)
    if dirname != "diagnostics":
        assert files, f"expected python modules under {dirname}/"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not _module_is_sources(alias.name), f"{path} imports {alias.name}"
                    name = alias.name.rsplit(".", 1)[-1]
                    assert name not in _FORBIDDEN_NAMES, f"{path} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                assert not _module_is_sources(node.module), f"{path} imports from {node.module}"
                if node.module == "agent_hotwash":
                    for alias in node.names:
                        assert alias.name != "sources", f"{path} imports sources"
                for alias in node.names:
                    assert alias.name not in _FORBIDDEN_NAMES, f"{path} imports {alias.name}"
            elif isinstance(node, ast.Name):
                assert node.id not in _FORBIDDEN_NAMES, f"{path} references {node.id}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in _FORBIDDEN_NAMES, f"{path} references .{node.attr}"


def test_identical_canonical_sessions_segment_the_same() -> None:
    """§9.6: same Events + Capabilities → same tasks/episodes regardless of origin."""
    from agent_hotwash.canonical import declared_row, observe_capabilities
    from agent_hotwash.config import load_config
    from agent_hotwash.events import AgentKind, Event, EventKind, Usage
    from agent_hotwash.sources._common import build_session
    from agent_hotwash.structure.episodes import segment_episodes
    from agent_hotwash.structure.tasks import segment_tasks

    cfg = load_config()
    events = [
        Event(kind=EventKind.user_msg, text="fix src/a.py", role_hint=None),
        Event(
            kind=EventKind.tool_call,
            tool_name="file.edit",
            op_kind="file.edit",
            path="src/a.py",
            call_id="c1",
        ),
        Event(kind=EventKind.tool_result, call_id="c1", ok=True, usage=Usage(input=10, output=4)),
    ]
    declared = declared_row(per_call_usage=True, timestamps=False, reasoning_text=True)
    a = build_session(events, AgentKind.unknown, session_id="s")
    a.capabilities = observe_capabilities(a, declared)
    b = build_session([e.model_copy(deep=True) for e in events], AgentKind.unknown, session_id="s")
    b.capabilities = observe_capabilities(b, declared)
    ta, tb = segment_tasks(a, cfg, semantic_mode="off"), segment_tasks(b, cfg, semantic_mode="off")
    ea, eb = segment_episodes(a, ta, cfg), segment_episodes(b, tb, cfg)
    assert [t.task_id for t in ta] == [t.task_id for t in tb]
    assert [e.termination for e in ea] == [e.termination for e in eb]


def test_capability_flip_only_unknowns_gated_features() -> None:
    from agent_hotwash.canonical import declared_row
    from agent_hotwash.events import Capabilities
    from agent_hotwash.semantic.bank import load_bank

    bank = load_bank()
    gated = [f for f in bank if "reasoning_text" in f.requires]
    assert gated
    present = Capabilities(declared=declared_row(reasoning_text=True))
    absent = Capabilities(declared=declared_row(reasoning_text=False))
    assert present.meets("reasoning_text")
    assert not absent.meets("reasoning_text")
    assert all(not absent.meets(req) for f in gated for req in f.requires if req == "reasoning_text")
