"""Opt-in live System One smoke (skipped without TYPESAFE_API_KEY)."""

from __future__ import annotations

import os

import pytest
from systemoneprompts.answers import is_answer_for_question, partition_answers

from agent_hotwash.config import load_config
from agent_hotwash.semantic.bank import load_feature_bank, wire_questions
from agent_hotwash.semantic.client import SystemOneAsker

pytestmark = pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")


def test_live_task_first_round_includes_score() -> None:
    cfg = load_config()
    loaded = load_feature_bank()
    subtypes = {sid for ids in loaded.routing.values() for sid in ids}
    first = [f for f in loaded.features if f.scope == "task" and f.id not in subtypes]
    assert any(f.id == "task.scope.breadth" for f in first)
    state = {
        "task": {
            "request": "Fix the failing test in src/parser.py",
            "amendments": [],
            "deliverables": ["src/parser.py"],
            "status": "open",
        }
    }
    asker = SystemOneAsker(
        cfg.semantic.model,
        cfg.semantic.cache_dir,
        mode="live",
        secret_patterns=list(cfg.lexicons.secret),
    )
    questions = wire_questions(first)
    try:
        answers = asker.ask(state, questions)
    finally:
        asker.close()
    partitioned = partition_answers(questions, answers)
    assert not partitioned["missing"]
    assert not partitioned["malformed"]
    for qid, question in questions.items():
        assert is_answer_for_question(question, answers[qid])
    score = answers["task.scope.breadth"]
    assert score["type"] == "score"
    assert isinstance(score["score"], float)
    assert len(score["legend"]) == 4
