"""State projection onto inspect paths."""

from __future__ import annotations

from agent_hotwash.semantic.bank import load_bank
from agent_hotwash.semantic.pipeline import questions_from_features, visible_state
from agent_hotwash.semantic.project import (
    inspect_paths_of,
    is_heavy_inspect,
    project_for_questions,
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


def test_project_for_questions_omits_unnamed_fact_flags() -> None:
    state = {
        "episode": {
            "instruction": "add the parser subtest",
            "messages": [{"kind": "assistant", "text": "done", "phase": "final_answer"}],
            "ops": [{"kind": "file.write", "paths": ["tests/test_parser.py"], "exit": 0, "out_head": "SECRET"}],
            "counts": {"n_ops": 1, "n_final_answer": 1, "majority_family": "modify", "by_family": {"modify": 1}},
            "facts": {
                "artifact_change": True,
                "verification": False,
                "env_impediment": "sandbox_denied",
                "declares_success": True,
                "cites_verification": False,
            },
            "position": "1 of 1",
            "prior_outcome": None,
        }
    }
    questions = {
        "episode.outcome.kind": {
            "type": "choice",
            "instructions": {
                "inspect": [
                    "`episode.instruction`",
                    "`episode.messages[-1].text`",
                    "`episode.ops[].kind`",
                    "`episode.ops[].paths`",
                    "`episode.ops[].exit`",
                ]
            },
        }
    }
    out = project_for_questions(state, questions)
    blob = str(out)
    assert out["episode"]["instruction"] == "add the parser subtest"
    assert "SECRET" not in blob
    assert "majority_family" not in blob
    assert "artifact_change" not in blob
    assert "env_impediment" not in blob
    assert "declares_success" not in blob
    assert "cites_verification" not in blob
    assert "facts" not in out["episode"]


def test_project_for_questions_without_inspect_is_empty() -> None:
    assert project_for_questions({"episode": {"facts": {"artifact_change": True}}}, {"q": {"type": "noul"}}) == {}


def test_visible_state_matches_live_purpose_outcome_projection() -> None:
    bank = {f.id: f for f in load_bank()}
    feats = [bank["episode.phase.purpose"], bank["episode.outcome.kind"], bank["episode.claim.declares_success"]]
    state = {
        "episode": {
            "instruction": "review src/parser.py",
            "messages": [{"kind": "user", "text": "review src/parser.py"}, {"kind": "assistant", "text": "looks fine"}],
            "ops": [{"kind": "cmd.read", "paths": ["src/parser.py"], "exit": 0, "out_head": "code"}],
            "counts": {"n_ops": 1, "n_final_answer": 1, "majority_family": "inspect", "by_family": {"inspect": 1}},
            "facts": {
                "artifact_change": False,
                "verification": False,
                "env_impediment": None,
                "declares_success": False,
                "cites_verification": False,
            },
            "position": "1 of 2",
            "prior_outcome": None,
        }
    }
    out = visible_state(state, feats)
    blob = str(out)
    assert out["episode"]["instruction"] == "review src/parser.py"
    assert "majority_family" not in blob
    assert "artifact_change" not in blob
    assert "env_impediment" not in blob
    assert "declares_success" not in blob
    assert "cites_verification" not in blob
    assert questions_from_features(feats).keys() == {f.id for f in feats}
