"""State projection onto inspect paths."""

from __future__ import annotations

from agent_hotwash.semantic.project import (
    group_wired_questions,
    inspect_paths_of,
    is_heavy_inspect,
    project_state,
    tokenize_path,
)


def test_tokenize_index_and_wildcard() -> None:
    assert tokenize_path("`episode.messages[-1].text`") == ["episode", "messages", ("idx", -1), "text"]
    assert tokenize_path("episode.ops[].kind") == ["episode", "ops", ("all", None), "kind"]
    assert tokenize_path("messages[0].text") == ["messages", ("idx", 0), "text"]


def test_project_keeps_indexed_messages_and_drops_op_bodies() -> None:
    state = {
        "task": {"request": "fix parser", "status": "open"},
        "episode": {
            "ops": [
                {"kind": "cmd.read", "cmd": "sed", "out_head": "SECRET_TAIL", "out_tail": "more"},
                {"kind": "cmd.exec", "cmd": "pytest", "out_head": "FAIL"},
            ],
            "messages": [
                {"kind": "user", "text": "start here", "phase": None},
                {"kind": "assistant", "text": "middle", "phase": "commentary"},
                {"kind": "assistant", "text": "end here", "phase": "final_answer"},
            ],
            "counts": {"by_family": {"inspect": 1}, "majority_family": "inspect", "n_ops": 2},
            "facts": {"env_impediment": "sandbox_denied", "verification": True, "artifact_change": False},
        },
    }
    out = project_state(
        state,
        [
            "task.request",
            "episode.messages[0].text",
            "episode.messages[-1].text",
            "episode.ops[].kind",
            "episode.counts.by_family",
            "episode.facts.verification",
        ],
    )
    blob = str(out)
    assert out["task"] == {"request": "fix parser"}
    assert out["episode"]["messages"][0] == {"kind": "user", "text": "start here"}
    assert out["episode"]["messages"][-1] == {"kind": "assistant", "text": "end here"}
    assert [op["kind"] for op in out["episode"]["ops"]] == ["cmd.read", "cmd.exec"]
    assert out["episode"]["ops"][0]["cmd"] == "sed"
    assert "out_head" not in out["episode"]["ops"][0]
    assert out["episode"]["counts"] == {"by_family": {"inspect": 1}}
    assert out["episode"]["facts"] == {"verification": True}
    assert "SECRET_TAIL" not in blob
    assert "majority_family" not in blob
    assert "env_impediment" not in blob
    assert "sandbox_denied" not in blob


def test_project_keeps_ledger_request_with_deliverables() -> None:
    state = {
        "messages": [{"kind": "user", "text": "finish the parser", "phase": None}],
        "ledger": {
            "request": "implement src/parser.py",
            "deliverables": ["src/parser.py"],
            "artifacts": ["src/unrelated.py"],
            "last_answer": "done",
        },
    }
    out = project_state(state, ["messages[0].text", "ledger.deliverables"])
    assert out["messages"][0] == {"kind": "user", "text": "finish the parser"}
    assert out["ledger"] == {"deliverables": ["src/parser.py"], "request": "implement src/parser.py"}
    assert "unrelated" not in str(out)


def test_project_without_paths_is_identity() -> None:
    state = {"x": 1}
    assert project_state(state, []) is state


def test_inspect_paths_of_reads_instructions() -> None:
    paths = inspect_paths_of(
        {
            "q": {
                "type": "noul",
                "instructions": {"inspect": ["`messages[0].text`"], "compare": ["`ledger.deliverables`"]},
            }
        }
    )
    assert paths == ["messages[0].text", "ledger.deliverables"]


def test_heavy_inspect_detects_whole_ops_and_tails() -> None:
    assert is_heavy_inspect("episode.ops")
    assert is_heavy_inspect("episode.facts")
    assert is_heavy_inspect("`episode.ops[].out_tail`")
    assert not is_heavy_inspect("episode.ops[].kind")
    assert not is_heavy_inspect("episode.facts.verification")
    assert not is_heavy_inspect("episode.messages[0].text")


def test_group_wired_questions_splits_inspect_sets() -> None:
    identity = {
        "type": "choice",
        "instructions": {"inspect": ["`messages[0].text`", "`ledger.deliverables`"]},
    }
    same_component = {
        "type": "noul",
        "instructions": {"inspect": ["`messages[0].text`", "`ledger.artifacts`"]},
    }
    inquire = {"type": "noul", "instructions": {"inspect": "`task.request`"}}
    groups = group_wired_questions(
        {
            "turn.relationship.task_identity": identity,
            "turn.relationship.same_component": same_component,
            "task.intent.inquire": inquire,
            "task.intent.change": inquire,
        }
    )
    assert [set(g) for g in groups] == [
        {"turn.relationship.task_identity"},
        {"turn.relationship.same_component"},
        {"task.intent.inquire", "task.intent.change"},
    ]
