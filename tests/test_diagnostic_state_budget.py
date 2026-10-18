"""Evidence budgets keep huge diagnostic inputs explicit and transport-bounded."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import httpx2
import pytest
from pydantic import ValidationError
from systemoneprompts.json_values import canonical_json

from agent_hotwash.config import SemanticConfig, load_config
from agent_hotwash.events import Capabilities
from agent_hotwash.semantic.bank import FeatureDef
from agent_hotwash.semantic.client import SemanticRequestTooLarge, SystemOneAsker
from agent_hotwash.semantic.pipeline import Annotator
from agent_hotwash.semantic.project import project_for_questions
from agent_hotwash.semantic.redact import redact_state
from agent_hotwash.semantic.state_budget import (
    CLIPPING_KEY,
    DiagnosticStateTooLarge,
    bound_diagnostic_state,
    serialized_bytes,
)


def question(path: str = "task.request") -> dict:
    return {
        "type": "noul",
        "instructions": {"question": "Is the requested work explicit?", "inspect": [f"`{path}`"]},
        "criteria": {},
    }


class Transport:
    def __init__(self):
        self.calls = []

    def __call__(self, request):
        body = json.loads(request.read())
        self.calls.append(body)
        return httpx2.Response(
            200,
            json={"model": "jev-1.13.0", "answers": {qid: {"type": "noul", "noul": 0.9} for qid in body["questions"]}},
        )


def asker(tmp_path: Path, handler: Transport, **kwargs) -> SystemOneAsker:
    return SystemOneAsker(
        "jev-1.13.0", tmp_path, mode="live", max_retries=0, transport_override=httpx2.MockTransport(handler), **kwargs
    )


def test_large_unicode_state_preserves_heads_tails_structure_and_provenance():
    state = {
        "task": {"request": "HEAD " + "雪🙂" * 100_000 + " TAIL", "status": "active", "count": 3},
        "failure": {"command": "START " + "x" * 150_000 + " END"},
    }
    original = copy.deepcopy(state)
    bounded, metadata = bound_diagnostic_state(state, 16_000)
    assert state == original
    assert serialized_bytes(bounded) <= 16_000
    assert metadata is not None
    assert metadata == bounded[CLIPPING_KEY]
    assert metadata["retained_bytes"] == serialized_bytes(bounded)
    assert metadata["original_bytes"] == serialized_bytes(state)
    assert metadata["original_sha256"] == hashlib.sha256(canonical_json(state).encode()).hexdigest()
    assert metadata["clipped_paths"] == ["/task/request", "/failure/command"]
    assert bounded["task"]["request"].startswith("HEAD ")
    assert bounded["task"]["request"].endswith(" TAIL")
    assert "UTF-8 bytes omitted" in bounded["task"]["request"]
    assert bounded["failure"]["command"].startswith("START ")
    assert bounded["failure"]["command"].endswith(" END")
    assert bounded["task"]["status"] == "active"
    assert bounded["task"]["count"] == 3
    assert "\ufffd" not in canonical_json(bounded)


def test_small_state_preserves_exact_object_and_projection_cache_identity(tmp_path):
    state = {"task": {"request": "fix tests", "status": "active"}}
    bounded, metadata = bound_diagnostic_state(state, 16_000)
    assert bounded is state and metadata is None
    handler = Transport()
    client = asker(tmp_path, handler)
    config = load_config()
    feat = FeatureDef("task.test.evidence", "task", "noul", question=question())
    original = client.ask(state, {feat.id: feat.question})
    value = Annotator(client, config, Capabilities(), mode="live", allow_unredacted=False).ask(state, [feat])[feat.id]
    assert value.answer == original[feat.id]
    assert value.state_clipping is None
    assert len(handler.calls) == 1


@pytest.mark.parametrize("scope,path", [("task", "task.request"), ("failure", "failure.command")])
def test_annotator_clips_after_redaction_and_projects_before_clipping(tmp_path, scope, path):
    handler = Transport()
    client = asker(tmp_path, handler)
    config = load_config().model_copy(
        update={"lexicons": load_config().lexicons.model_copy(update={"secret": [r"SECRET\[.*?\]"]})}
    )
    feat = FeatureDef(f"{scope}.test.evidence", scope, "noul", question=question(path))
    state = {
        "task": {"request": "HEAD " + "SECRET[" + "private" * 30_000 + "]" + " snow雪 " * 50_000 + " TAIL"},
        "failure": {"command": "HEAD " + "SECRET[" + "private" * 30_000 + "]" + " snow雪 " * 50_000 + " TAIL"},
        "uninspected": "ignored" * 100_000,
    }
    original = copy.deepcopy(state)
    value = Annotator(client, config, Capabilities(), mode="live", allow_unredacted=False).ask(state, [feat])[feat.id]
    body = handler.calls[0]
    assert serialized_bytes(body["state"]) <= config.semantic.diagnostic_state_max_bytes
    assert serialized_bytes(body) <= config.semantic.diagnostic_request_max_bytes
    assert "private" not in canonical_json(body)
    assert "uninspected" not in body["state"]
    assert value.state_clipping == body["state"][CLIPPING_KEY]
    assert value.state_hash == hashlib.sha256(canonical_json(body["state"]).encode()).hexdigest()
    assert state == original
    assert value.reason is None


def test_non_diagnostic_scope_keeps_existing_evidence(tmp_path):
    handler = Transport()
    client = asker(tmp_path, handler)
    feat = FeatureDef("episode.test.evidence", "episode", "noul", question=question("episode.request"))
    state = {"episode": {"request": "x" * 20_000}}
    value = Annotator(client, load_config(), Capabilities(), mode="live", allow_unredacted=False).ask(state, [feat])[
        feat.id
    ]
    assert value.state_clipping is None
    assert handler.calls[0]["state"] == state


def test_request_bound_splits_before_transport_and_keeps_shared_default(tmp_path):
    handler = Transport()
    client = asker(tmp_path, handler, max_request_bytes=2000)
    state = {"task": {"request": "x" * 400}}
    qs = {f"task.test.q{i}": question() for i in range(6)}
    max_single = max(
        serialized_bytes({"model": client.model, "state": state, "questions": {qid: q}}) for qid, q in qs.items()
    )
    bound = max_single + 50
    answers = client.ask(state, qs, max_request_bytes=bound)
    assert set(answers) == set(qs)
    assert len(handler.calls) > 1
    assert all(serialized_bytes(body) <= bound for body in handler.calls)
    assert client.max_request_bytes == 2000
    assert client.ask(state, qs, max_request_bytes=bound) == answers
    assert client.stats()["requests"] == len(handler.calls)


def test_unfit_singleton_is_explicit_and_never_sent(tmp_path):
    handler = Transport()
    client = asker(tmp_path, handler)
    with pytest.raises(SemanticRequestTooLarge):
        client.ask({"task": {"request": "x" * 5000}}, {"task.test.evidence": question()}, max_request_bytes=1000)
    assert handler.calls == []
    config = load_config().model_copy(
        update={"semantic": load_config().semantic.model_copy(update={"diagnostic_request_max_bytes": 1024})}
    )
    feat = FeatureDef("task.test.evidence", "task", "noul", question=question())
    values = Annotator(client, config, Capabilities(), mode="live", allow_unredacted=False).ask(
        {"task": {"request": "x" * 5000}}, [feat]
    )
    assert values[feat.id].reason == "state_too_large"
    assert handler.calls == []


def test_structural_overflow_is_explicit():
    with pytest.raises(DiagnosticStateTooLarge):
        bound_diagnostic_state({"flags": {f"flag{i}": True for i in range(5000)}}, 16_000)


@pytest.mark.parametrize("field", ["diagnostic_state_max_bytes", "diagnostic_request_max_bytes"])
@pytest.mark.parametrize("value", [0, -1, 1023])
def test_invalid_budget_config_is_rejected(field, value):
    with pytest.raises(ValidationError):
        SemanticConfig.model_validate({field: value})


def test_projection_does_not_leak_uninspected_secrets():
    state = {"task": {"request": "safe"}, "private": "secret" * 100_000}
    projected = project_for_questions(redact_state(state, []), {"task.test.evidence": question()})
    bounded, metadata = bound_diagnostic_state(projected, 16_000)
    assert metadata is None
    assert bounded == {"task": {"request": "safe"}}


def test_large_collections_keep_head_tail_and_explicit_omitted_counts():
    values = [f"item-{index}" for index in range(1500)]
    state = {"task": {"request": "START " + "work " * 80_000 + " END", "deliverables": values, "status": "active"}}
    bounded, metadata = bound_diagnostic_state(state, 16_000)
    assert metadata is not None
    assert serialized_bytes(bounded) <= 16_000
    retained = bounded["task"]["deliverables"]
    assert retained[0] == "item-0"
    assert retained[-1] == "item-1499"
    omission = next(item["_diagnostic_omitted_items"] for item in retained if isinstance(item, dict))
    collection = metadata["clipped_collections"][0]
    assert collection["path"] == "/task/deliverables"
    assert collection["omitted_items"] == omission
    assert collection["original_items"] == 1500
    assert collection["retained_items"] + omission == 1500
    assert collection["original_sha256"] == hashlib.sha256(canonical_json(values).encode()).hexdigest()
    assert state["task"]["deliverables"] == values
    assert bounded["task"]["request"].startswith("START ")
    assert bounded["task"]["request"].endswith(" END")


def test_turn_relationship_classifier_bounds_ledger_and_new_message(tmp_path):
    handler = Transport()
    client = asker(tmp_path, handler)
    questions = {
        "turn.relationship.test": {
            "type": "noul",
            "instructions": {
                "question": "Does the message update this ledger?",
                "inspect": ["`ledger.request`", "`messages[].text`"],
            },
            "criteria": {},
        }
    }
    feat = FeatureDef("turn.relationship.test", "turn", "noul", question=questions["turn.relationship.test"])
    state = {
        "ledger": {"request": "HEAD " + "ledger雪 " * 50_000 + " TAIL"},
        "messages": [{"kind": "user", "text": "START " + "new🙂 " * 50_000 + " END"}],
    }
    original = copy.deepcopy(state)
    value = Annotator(client, load_config(), Capabilities(), mode="live", allow_unredacted=False).ask(state, [feat])[
        feat.id
    ]
    sent = handler.calls[0]["state"]
    assert serialized_bytes(sent) <= 16_000
    assert sent["ledger"]["request"].startswith("HEAD ")
    assert sent["ledger"]["request"].endswith(" TAIL")
    assert sent["messages"][0]["text"].startswith("START ")
    assert sent["messages"][0]["text"].endswith(" END")
    assert value.state_clipping is not None
    assert value.state_clipping == sent[CLIPPING_KEY]
    assert set(value.state_clipping["clipped_paths"]) == {"/ledger/request", "/messages/0/text"}
    assert state == original


def test_unrelated_turn_classifier_keeps_existing_state(tmp_path):
    handler = Transport()
    client = asker(tmp_path, handler)
    feat = FeatureDef("turn.test.evidence", "turn", "noul", question=question("turn.request"))
    state = {"turn": {"request": "x" * 20_000}}
    value = Annotator(client, load_config(), Capabilities(), mode="live", allow_unredacted=False).ask(state, [feat])[
        feat.id
    ]
    assert value.state_clipping is None
    assert handler.calls[0]["state"] == state
