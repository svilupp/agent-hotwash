"""Native System One bank files match the digest / builder state contract."""

from __future__ import annotations

from pathlib import Path

from systemoneprompts import check_definition, create_state_assert, errors_of, load_definition

from agent_hotwash.config import load_config
from agent_hotwash.semantic.bank import load_feature_bank, scope_requires
from agent_hotwash.semantic.pipeline import _relationship_state, _task_state
from agent_hotwash.structure.digest import build_digest
from agent_hotwash.structure.episodes import segment_episodes
from agent_hotwash.structure.tasks import segment_tasks

_FEATURES = Path(__file__).resolve().parents[1] / "src" / "agent_hotwash" / "semantic" / "features"
_FIXTURES = Path(__file__).parent / "fixtures"
_TRACES = [
    _FIXTURES / "codebench" / "claude_run",
    _FIXTURES / "codex_native" / "v0153" / "user_multiturn.jsonl",
]


def test_scope_files_check_definition_clean() -> None:
    for name in ("task.toml", "episode.toml", "turn.toml"):
        definition = load_definition(str(_FEATURES / name))
        diagnostics = check_definition(definition)
        assert errors_of(diagnostics) == [], diagnostics
        assert [d for d in diagnostics if d.code == "unguaranteed-backtick"] == []


def test_task_state_satisfies_requires() -> None:
    from agent_hotwash.structure.ledger import Ledger
    from agent_hotwash.structure.tasks import Task

    assert_state = create_state_assert(scope_requires("task"))
    task = Task(task_id="s:task0", session_id="s", ledger=Ledger(request="fix src/a.py", status="open"))
    assert_state(_task_state(task))


def test_relationship_state_satisfies_turn_requires() -> None:
    from agent_hotwash.events import (
        AgentKind,
        Event,
        EventKind,
        Session,
        Turn,
        TurnStatus,
        UserInput,
    )
    from agent_hotwash.structure.ledger import Ledger, update_ledger

    turn = Turn(
        turn_id="t2",
        session_id="s",
        event_start=2,
        event_end=3,
        status=TurnStatus.completed,
        user_input=UserInput(text="now the exporter", kind="user"),
    )
    session = Session(
        session_id="s",
        agent=AgentKind.unknown,
        events=[
            Event(kind=EventKind.user_msg, idx=0, text="implement src/a.py", turn_id="t1"),
            Event(kind=EventKind.assistant_msg, idx=1, text="done", turn_id="t1"),
        ],
        turns=[],
        model="m",
    )
    ledger = Ledger(request="implement src/a.py")
    first = Turn(
        turn_id="t1",
        session_id="s",
        event_start=0,
        event_end=1,
        status=TurnStatus.completed,
        user_input=UserInput(text="implement src/a.py", kind="user"),
    )
    update_ledger(ledger, first, session.events)
    assert_state = create_state_assert(scope_requires("turn"))
    assert_state(_relationship_state(ledger, turn))


def test_fixture_digests_satisfy_episode_requires() -> None:
    from agent_hotwash.canonical import build_turns
    from agent_hotwash.sources.detect import iter_traces

    cfg = load_config()
    assert_state = create_state_assert(scope_requires("episode"))
    checked = 0
    for path in _TRACES:
        for trace in iter_traces(path):
            session = trace.root
            if not session.turns:
                session.turns = build_turns(
                    session,
                    injected_tags=cfg.structure.injected_tags_user,
                    delegation_tag=cfg.structure.delegation_tag,
                )
            tasks = segment_tasks(session, cfg, semantic_mode="off")
            episodes = segment_episodes(session, tasks, cfg)
            task_by_id = {t.task_id: t for t in tasks}
            for ep in episodes:
                task = task_by_id.get(ep.task_id)
                if task is None:
                    continue
                digest = build_digest(task, ep, session, cfg)
                facts = (digest.get("episode") or {}).get("facts")
                if not facts:
                    continue
                assert_state(digest)
                checked += 1
    assert checked >= 1


def test_bank_routing_ids_exist() -> None:
    loaded = load_feature_bank()
    ids = {f.id for f in loaded.features}
    assert loaded.routing
    for super_id, children in loaded.routing.items():
        assert super_id in ids
        for child in children:
            assert child in ids
