"""Pi native capabilities + heuristic turns (WP2 / WP8)."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.events import CapLevel
from agent_hotwash.sources.pi_native import load_session_file

SESSION = Path(__file__).parent / "fixtures" / "pi_native" / "session-fixture.jsonl"


def test_pi_fixture_yields_turns_and_declared_reasoning_effort_false() -> None:
    trace = load_session_file(SESSION)
    assert trace.root.turns
    assert trace.root.turns[0].user_input.text
    assert trace.root.capabilities.declared.reasoning_effort is CapLevel.false
    assert trace.capabilities.declared.reasoning_effort is CapLevel.false
    assert any(ev.raw_type == "model_change" for ev in trace.root.events)
    assert trace.root.turns[0].model_config_active.model == "acme-mini"
