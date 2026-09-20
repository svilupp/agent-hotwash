"""System One asker, bank overlay, redaction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from systemoneprompts.client import TypeSafeClientError, TypeSafeHttpError
from systemoneprompts.diagnostics import SystemOnePromptsError
from systemoneprompts.json_values import canonical_json

from agent_hotwash.config import load_config
from agent_hotwash.semantic.bank import FeatureDef, load_bank, validate_bank
from agent_hotwash.semantic.client import CacheMissError, SystemOneAsker
from agent_hotwash.semantic.redact import redact_state
from agent_hotwash.semantic.results import (
    FeatureValue,
    declared_success_without_observed_verification,
    investigation_then_change,
    recovered,
    stuck_window,
    thrashing_window,
)


def _synthetic_answer(qdef: dict, *, choice: str | None = None) -> dict:
    qtype = qdef.get("type") or "noul"
    if qtype == "choice":
        labels = [str(k) for k in (qdef.get("criteria") or {})]
        picked = choice if choice in labels else (labels[0] if labels else "other")
        if picked not in labels:
            labels.append(picked)
        rest = (1.0 - 0.9) / max(len(labels) - 1, 1)
        probs = {lab: (0.9 if lab == picked else rest) for lab in labels}
        if len(labels) == 1:
            probs[picked] = 1.0
        return {"type": "choice", "choice": picked, "confidence": 0.9, "probabilities": probs}
    if qtype == "score":
        raw_criteria = qdef.get("criteria")
        criteria = raw_criteria if isinstance(raw_criteria, list) else []
        legend = {
            str(i): (block.get("name") if isinstance(block, dict) else str(i)) for i, block in enumerate(criteria)
        }
        idx = 1 if len(criteria) > 1 else 0
        probs = {str(i): (1.0 if i == idx else 0.0) for i in range(len(criteria))}
        return {"type": "score", "score": float(idx), "confidence": 0.8, "legend": legend, "probabilities": probs}
    return {"type": "noul", "noul": 0.91}


class RecordingHandler:
    """httpx2 MockTransport handler that records System One bodies."""

    def __init__(self, *, fail: list[int] | None = None, identity: str | None = None) -> None:
        self.calls: list[dict] = []
        self.fail = list(fail or [])
        self.identity = identity

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        body = json.loads(request.content.decode() if request.content else "{}")
        self.calls.append(body)
        if self.fail:
            code = self.fail.pop(0)
            headers = {"retry-after": "0"} if code == 429 else {}
            return httpx2.Response(code, json={"error": f"http {code}"}, headers=headers)
        questions = body.get("questions") or {}
        answers = {
            qid: _synthetic_answer(qdef, choice=self.identity if str(qid).endswith("task_identity") else None)
            for qid, qdef in questions.items()
        }
        return httpx2.Response(
            200,
            json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}},
        )

    @property
    def question_ids(self) -> list[str]:
        return [qid for call in self.calls for qid in (call.get("questions") or {})]


def _noul_q(n: int) -> dict[str, dict]:
    return {
        f"episode.progress.q{i:02d}": {
            "type": "noul",
            "instructions": {"question": f"yes {i}?"},
            "criteria": {},
        }
        for i in range(n)
    }


def _asker(
    tmp_path: Path,
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    mode: str = "live",
    max_questions: int = 15,
    limiter: object | None = None,
    max_retries: int = 3,
) -> SystemOneAsker:
    from agent_hotwash.semantic.ratelimit import RateLimiter

    return SystemOneAsker(
        "jev-1.13.0",
        tmp_path,
        mode=mode,
        max_questions=max_questions,
        max_retries=max_retries,
        timeout_s=5.0,
        limiter=limiter if isinstance(limiter, RateLimiter) else None,
        transport_override=httpx2.MockTransport(handler),
    )


def _cache_files(tmp_path: Path) -> int:
    return len(list(tmp_path.rglob("*.json")))


@pytest.fixture
def no_client_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("systemoneprompts.client.time.sleep", lambda _s: None)


def test_batching_max_15(tmp_path: Path) -> None:
    handler = RecordingHandler()
    asker = _asker(tmp_path, handler, max_questions=15)
    out = asker.ask({"k": "v"}, _noul_q(20))
    assert len(out) == 20
    assert len(handler.calls) == 2
    assert all(len(c["questions"]) <= 15 for c in handler.calls)
    assert len(handler.calls[0]["questions"]) == 15
    assert len(handler.calls[1]["questions"]) == 5
    asker.close()


def test_asker_projects_identity_without_artifacts(tmp_path: Path) -> None:
    handler = RecordingHandler()
    asker = _asker(tmp_path, handler)
    qs = {
        "turn.relationship.task_identity": {
            "type": "choice",
            "instructions": {
                "question": "same deliverable?",
                "inspect": ["`messages[0].text`", "`ledger.deliverables`"],
            },
            "criteria": {
                "same_deliverable": {"what": "Same named output.", "examples": ["a", "b"]},
                "other": {"what": "Anything else.", "examples": ["a", "b"]},
            },
        },
    }
    state = {
        "messages": [{"kind": "user", "text": "finish the parser"}],
        "ledger": {
            "request": "implement src/parser.py",
            "deliverables": ["src/parser.py"],
            "artifacts": ["src/unrelated.py"],
        },
    }
    asker.ask(state, qs)
    asker.close()
    sent = handler.calls[0]["state"]
    blob = json.dumps(sent)
    assert sent["ledger"]["deliverables"] == ["src/parser.py"]
    assert sent["ledger"]["request"] == "implement src/parser.py"
    assert "src/unrelated.py" not in blob
    assert sent["messages"][0]["kind"] == "user"


def test_retry_on_429_then_success(tmp_path: Path, no_client_sleep: None) -> None:
    from agent_hotwash.semantic.ratelimit import RateLimiter

    handler = RecordingHandler(fail=[429, 429])
    limiter = RateLimiter(1000.0, burst=10)
    asker = _asker(tmp_path, handler, limiter=limiter)
    out = asker.ask({"k": "v"}, _noul_q(1))
    assert out["episode.progress.q00"]["noul"] == 0.91
    assert len(handler.calls) == 3
    assert asker.stats()["requests"] == 3
    assert asker.stats()["retries"] == 2
    asker.close()


def test_401_does_not_retry(tmp_path: Path, no_client_sleep: None) -> None:
    handler = RecordingHandler(fail=[401, 401, 401, 401])
    asker = _asker(tmp_path, handler)
    with pytest.raises(TypeSafeHttpError) as ei:
        asker.ask({"k": "v"}, _noul_q(1))
    assert ei.value.status == 401
    assert len(handler.calls) == 1
    asker.close()


def test_5xx_retries_then_raises(tmp_path: Path, no_client_sleep: None) -> None:
    handler = RecordingHandler(fail=[503, 503, 503, 503])
    asker = _asker(tmp_path, handler, max_retries=3)
    with pytest.raises(TypeSafeHttpError) as ei:
        asker.ask({"k": "v"}, _noul_q(1))
    assert ei.value.status == 503
    assert len(handler.calls) == 4
    asker.close()


def test_cache_hit_stable(tmp_path: Path) -> None:
    handler = RecordingHandler()
    asker = _asker(tmp_path, handler)
    state = {"episode": {"ops": [1]}}
    qs = _noul_q(2)
    a = asker.ask(state, qs)
    b = asker.ask(state, qs)
    assert a == b
    assert len(handler.calls) == 1
    asker.close()


def test_cached_mode_never_touches_transport(tmp_path: Path) -> None:
    handler = RecordingHandler()
    cached = _asker(tmp_path, handler, mode="cached")
    state = {"x": 1}
    qs = _noul_q(1)
    with pytest.raises(CacheMissError) as ei:
        cached.ask(state, qs)
    assert ei.value.ids
    assert handler.calls == []
    cached.close()

    live = _asker(tmp_path, handler, mode="live")
    live.ask(state, qs)
    assert len(handler.calls) == 1
    live.close()

    def boom(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("transport must not be called in cached mode")

    warmed = SystemOneAsker(
        "jev-1.13.0",
        tmp_path,
        mode="cached",
        transport_override=httpx2.MockTransport(boom),
    )
    warmed.ask(state, qs)
    warmed.close()


def test_omitted_answer_id_is_not_cached_and_raises(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        )

    asker = _asker(tmp_path, handler)
    with pytest.raises(SystemOnePromptsError) as ei:
        asker.ask({"x": 1}, _noul_q(1))
    assert "answers-incomplete" in str(ei.value) or "missing" in str(ei.value)
    assert _cache_files(tmp_path) == 0
    asker.close()
    cached = _asker(tmp_path, handler, mode="cached")
    with pytest.raises(CacheMissError):
        cached.ask({"x": 1}, _noul_q(1))
    cached.close()


def test_malformed_noul_raises(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        body = json.loads(request.content.decode())
        answers = {qid: {"type": "noul", "noul": "high"} for qid in body.get("questions") or {}}
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {}})

    asker = _asker(tmp_path, handler)
    with pytest.raises((TypeSafeClientError, SystemOnePromptsError)):
        asker.ask({"x": 1}, _noul_q(1))
    assert _cache_files(tmp_path) == 0
    asker.close()


def test_cold_partial_batch_does_not_cache_siblings(tmp_path: Path) -> None:
    qs = _noul_q(3)
    good, bad, missing = sorted(qs)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    good: {"type": "noul", "noul": 0.4},
                    bad: {
                        "type": "choice",
                        "choice": "nope",
                        "confidence": 0.9,
                        "probabilities": {"nope": 1.0},
                    },
                },
                "usage": {},
            },
        )

    asker = _asker(tmp_path, handler)
    with pytest.raises(SystemOnePromptsError):
        asker.ask({"x": 1}, qs)
    assert _cache_files(tmp_path) == 0
    asker.close()
    cached = _asker(tmp_path, handler, mode="cached")
    with pytest.raises(CacheMissError) as ei:
        cached.ask({"x": 1}, qs)
    assert set(ei.value.ids) >= {bad, missing}
    cached.close()


def test_valid_batch_is_cached_and_typed(tmp_path: Path) -> None:
    qs = {
        "episode.progress.n": {"type": "noul", "instructions": {"question": "n?"}, "criteria": {}},
        "episode.phase.c": {
            "type": "choice",
            "instructions": {"question": "c?"},
            "criteria": {
                "a": {"what": "A labelled situation that applies here.", "examples": ["ex one", "ex two"]},
                "other": {"what": "Anything else that does not fit.", "examples": ["ex one", "ex two"]},
            },
        },
        "episode.reasoning.s": {
            "type": "score",
            "instructions": {"question": "s?"},
            "criteria": [
                {"name": "0_none", "what": "None of the work is demanded here.", "examples": ["ex one", "ex two"]},
                {"name": "1_local", "what": "A local inference is demanded here.", "examples": ["ex one", "ex two"]},
            ],
        },
    }
    handler = RecordingHandler()
    asker = _asker(tmp_path, handler)
    out = asker.ask({"x": 1}, qs)
    assert out["episode.progress.n"]["type"] == "noul"
    assert out["episode.phase.c"]["choice"] in {"a", "other"}
    assert isinstance(out["episode.reasoning.s"]["score"], float)
    assert _cache_files(tmp_path) == 3
    asker.close()

    def boom(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("cached mode hit transport")

    warmed = SystemOneAsker("jev-1.13.0", tmp_path, mode="cached", transport_override=httpx2.MockTransport(boom))
    assert warmed.ask({"x": 1}, qs) == out
    warmed.close()


def test_live_refuses_unredacted(tmp_path: Path) -> None:
    handler = RecordingHandler()
    asker = _asker(tmp_path, handler)
    with pytest.raises(ValueError, match="unredacted"):
        asker.ask({}, _noul_q(1), redact=False)
    out = asker.ask({}, _noul_q(1), redact=False, allow_unredacted=True)
    assert out
    asker.close()


def test_no_key_without_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeClientError) as ei:
        SystemOneAsker("jev-1.13.0", tmp_path, mode="live", transport_override=None)
    assert ei.value.diagnostic.code == "missing-credentials"


def test_canonical_json_stable() -> None:
    a = canonical_json({"b": "é", "a": 1})
    b = canonical_json({"a": 1, "b": "é"})
    assert a == b
    assert " " not in a


def _valid_noul_criteria() -> dict:
    return {
        "true": {
            "what": "A yes situation that is observable.",
            "examples": ["first example here", "second example here"],
        },
        "false": {
            "what": "A no situation that is observable.",
            "not_for": "Ambiguous cases.",
            "examples": ["first counter-example", "second counter-example"],
        },
    }


def test_validator_rejects_bad_ids() -> None:
    feat = FeatureDef(
        id="Turn.Relationship.x",
        scope="turn",
        primitive="noul",
        question="Does it?",
        criteria=_valid_noul_criteria(),
    )
    with pytest.raises(ValueError, match="invalid feature id"):
        validate_bank([feat])


def test_validator_rejects_too_many_choice_options() -> None:
    criteria = {
        f"opt{i}": {"what": "A labelled situation that applies here.", "examples": ["ex one", "ex two"]}
        for i in range(8)
    }
    criteria["other"] = {"what": "Anything else that does not fit.", "examples": ["ex one", "ex two"]}
    feat = FeatureDef(
        id="episode.phase.activity",
        scope="episode",
        primitive="choice",
        question="What activity?",
        criteria=criteria,
        options=list(criteria.keys()),
    )
    with pytest.raises(ValueError, match="options"):
        validate_bank([feat])


def test_validator_rejects_missing_examples() -> None:
    feat = FeatureDef(
        id="task.intent.inquire",
        scope="task",
        primitive="noul",
        question="Does it inquire?",
        criteria={
            "true": {"what": "The request asks for knowledge.", "examples": ["only one"]},
            "false": {"what": "The request asks for a change.", "examples": ["a", "b"]},
        },
    )
    with pytest.raises(ValueError, match="examples"):
        validate_bank([feat])


def test_load_bank_smoke_relationship_features() -> None:
    feats = load_bank()
    ids = {f.id for f in feats}
    assert ids >= {
        "turn.relationship.task_identity",
        "turn.relationship.corrects_prior",
        "turn.relationship.references_prior_result",
        "turn.relationship.same_component",
    }
    identity = next(f for f in feats if f.id.endswith("task_identity"))
    assert "other" in identity.options
    assert len(identity.options) <= 8


def test_feature_question_wraps_inspect_focus_and_compare() -> None:
    from agent_hotwash.semantic.pipeline import feature_question

    feats = {f.id: f for f in load_bank()}
    execute = feature_question(feats["task.intent.execute"])
    assert isinstance(execute["instructions"], dict)
    assert execute["instructions"]["question"] == feats["task.intent.execute"].question_text
    assert "`task.request`" in str(execute["instructions"]["inspect"])
    assert "focus" in execute["instructions"]
    rel = feature_question(feats["turn.relationship.same_component"])
    blob = json.dumps(rel["instructions"])
    assert "`messages[0].text`" in blob
    assert "`ledger.artifacts`" in blob
    assert "compare" in rel["instructions"]
    claim = feature_question(feats["episode.claim.declares_success"])
    assert "`episode.messages[].text`" in str(claim["instructions"]["inspect"])
    purpose = feature_question(feats["episode.phase.purpose"])
    blob = json.dumps(purpose["instructions"])
    assert "`task.request`" in blob
    assert "`episode.messages[].text`" in blob
    assert "`episode.counts.n_final_answer`" in blob
    assert "env_impediment" not in blob
    assert "majority_family" not in blob
    outcome = feature_question(feats["episode.outcome.kind"])
    out_blob = json.dumps(outcome["instructions"])
    assert "`episode.messages[-1].text`" in out_blob
    assert "env_impediment" not in out_blob
    activity = feature_question(feats["episode.phase.activity"])
    act_blob = json.dumps(activity["instructions"])
    assert "`episode.counts.by_family`" in act_blob
    assert "majority_family" not in act_blob


def test_redaction_corpus_and_false_positive() -> None:
    secrets = load_config().lexicons.secret
    state = {
        "email": "alice@corp.example",
        "phone": "555-123-4567",
        "secret": "sk-" + "a" * 24,
        "url": "https://alice:hunter2@git.internal/repo?token=abc123xyz0000000",
        "path": "/Users/alice/src/app.py",
        "vol": "/Volumes/Data/proj/x.py",
        "home": "/home/alice/code.py",
        "sentinel": "saw item_completed then Script completed and token_usage_record",
        "diff": "const test_key = 1\nAKIA not here\n",
    }
    out = redact_state(state, secrets)
    blob = json.dumps(out)
    assert "alice@corp.example" not in blob
    assert "555-123-4567" not in blob
    assert "sk-" + "a" * 24 not in blob
    assert "hunter2" not in blob
    assert "token=abc123xyz0000000" not in blob
    assert "/Users/alice" not in blob
    assert out["path"].startswith("/home/user/")
    assert out["vol"].startswith("/home/user/")
    assert "/home/alice" not in out["home"]
    assert "item_completed" in out["sentinel"]
    assert "Script completed" in out["sentinel"]
    assert "token_usage_record" in out["sentinel"]
    assert "test_key" in out["diff"]


def _two_turn_session():
    from agent_hotwash.events import (
        AgentKind,
        Event,
        EventKind,
        ModelCall,
        Session,
        Turn,
        TurnStatus,
        Usage,
        UserInput,
    )

    events = [
        Event(kind=EventKind.user_msg, idx=0, text="implement src/a.py", turn_id="t1"),
        Event(kind=EventKind.assistant_msg, idx=1, text="done", phase="final_answer", turn_id="t1"),
        Event(kind=EventKind.user_msg, idx=2, text="now the exporter", turn_id="t2"),
        Event(kind=EventKind.assistant_msg, idx=3, text="ok", phase="final_answer", turn_id="t2"),
    ]

    def _turn(tid: str, start: int, end: int, text: str) -> Turn:
        return Turn(
            turn_id=tid,
            session_id="s",
            event_start=start,
            event_end=end,
            status=TurnStatus.completed,
            user_input=UserInput(text=text, kind="user"),
            model_calls=[
                ModelCall(response_id=f"r-{tid}", turn_id=tid, event_start=start, event_end=end, usage=Usage(input=1))
            ],
        )

    turns = [_turn("t1", 0, 1, "implement src/a.py"), _turn("t2", 2, 3, "now the exporter")]
    return Session(session_id="s", agent=AgentKind.unknown, events=events, turns=turns, model="m")


def _relationship(identity: str, tmp_path: Path) -> tuple[RecordingHandler, dict]:
    from agent_hotwash.events import Capabilities
    from agent_hotwash.semantic.pipeline import Annotator, _relationship_answers

    handler = RecordingHandler(identity=identity)
    asker = _asker(tmp_path, handler)
    cfg = load_config()
    annotator = Annotator(asker, cfg, Capabilities(), mode="live", allow_unredacted=True)
    session = _two_turn_session()
    rel = [f for f in load_bank() if f.id.startswith("turn.relationship.")]
    answers = _relationship_answers(annotator, session, session.turns, rel)
    asker.close()
    return handler, answers


def test_relationship_asks_identity_first_and_verifiers_only_when_routed(tmp_path: Path) -> None:
    handler, answers = _relationship("distinct_deliverable", tmp_path)
    ids = handler.question_ids
    assert ids[0] == "turn.relationship.task_identity"
    assert len(handler.calls[0]["questions"]) == 1
    assert set(ids) == {
        "turn.relationship.task_identity",
        "turn.relationship.corrects_prior",
        "turn.relationship.references_prior_result",
        "turn.relationship.same_component",
    }
    assert set(answers["t2"]) == set(ids)
    state = handler.calls[0]["state"]
    assert "new_message" not in state
    assert state["messages"][0]["text"] == "now the exporter"
    assert state["messages"][0]["kind"] == "user"
    assert set(state["ledger"]) == {"deliverables", "request"}
    assert "artifacts" not in state["ledger"]
    ins = handler.calls[0]["questions"]["turn.relationship.task_identity"]["instructions"]
    assert isinstance(ins, dict)
    assert "`messages[0].text`" in str(ins.get("inspect"))


@pytest.mark.parametrize("identity", ["other", "unclear"])
def test_relationship_escape_hatch_skips_verifiers(identity: str, tmp_path: Path) -> None:
    handler, answers = _relationship(identity, tmp_path)
    assert handler.question_ids == ["turn.relationship.task_identity"]
    assert len(handler.calls) == 1
    assert set(answers["t2"]) == {"turn.relationship.task_identity"}


def test_apply_phase_labels_fills_display_label_from_features() -> None:
    from agent_hotwash.semantic.pipeline import apply_phase_labels
    from agent_hotwash.semantic.results import FeatureSet, FeatureValue
    from agent_hotwash.structure.episodes import Episode

    ep = Episode(episode_id="s:ep0", task_id="s:task0", turn_id="t1")
    other = Episode(episode_id="s:ep1", task_id="s:task0", turn_id="t1")
    fs = FeatureSet(
        scope="episode",
        object_id="s:ep0",
        values={
            "episode.phase.activity": FeatureValue(id="episode.phase.activity", value="inspect"),
            "episode.phase.purpose": FeatureValue(id="episode.phase.purpose", value={"choice": "orient"}),
        },
    )
    apply_phase_labels([ep, other], [fs])
    assert (ep.phase_activity, ep.phase_purpose) == ("inspect", "orient")
    assert (other.phase_activity, other.phase_purpose) == (None, None)


def test_trajectory_groups_on_resolved_activity_without_atom_labels() -> None:
    """Display grouping keys on the FeatureSet answer even when ``phase_activity`` is unset."""
    from agent_hotwash.analytics import Analysis, SessionMetrics
    from agent_hotwash.events import AgentKind
    from agent_hotwash.report.cards import trajectory_label
    from agent_hotwash.report.model import RunResult, StructureSection
    from agent_hotwash.semantic.results import FeatureSet, FeatureValue
    from agent_hotwash.structure.episodes import Episode
    from agent_hotwash.structure.ledger import Ledger
    from agent_hotwash.structure.tasks import Task

    def _ep(n: int) -> Episode:
        return Episode(episode_id=f"s:ep{n}", task_id="s:task0", turn_id="t1")

    def _fs(n: int, activity: str, purpose: str) -> FeatureSet:
        return FeatureSet(
            scope="episode",
            object_id=f"s:ep{n}",
            values={
                "episode.phase.activity": FeatureValue(id="episode.phase.activity", value=activity),
                "episode.phase.purpose": FeatureValue(id="episode.phase.purpose", value=purpose),
            },
        )

    eps = [_ep(0), _ep(1), _ep(2)]
    assert all(ep.phase_activity is None for ep in eps)
    task = Task(task_id="s:task0", session_id="s", ledger=Ledger(request="x"))
    metrics = SessionMetrics(session_id="s", agent=AgentKind.unknown)
    run = RunResult(
        analysis=Analysis(trace_id="t", agent=AgentKind.unknown, root=metrics),
        findings=[],
        structure=StructureSection(tasks=[task], episodes=eps),
        features=[_fs(0, "inspect", "orient"), _fs(1, "inspect", "orient"), _fs(2, "modify", "produce")],
    )
    assert trajectory_label(run, "s:task0") == "inspect x orient (x2) → modify x produce"


def test_derived_formulas() -> None:
    stuck = [
        {"facts": {"failure_signature_repeats": 2, "novel_output": 0.1, "artifact_change": False}},
        {"facts": {"failure_signature_repeats": 1, "novel_output": 0.2, "ops": ["cmd.read"]}},
        {"facts": {"failure_signature_repeats": 3, "novel_output": 0.0, "edits": 0}},
    ]
    assert stuck_window(3, stuck)
    assert not stuck_window(4, stuck)
    thrash = [
        {"ops": ["file.edit"], "paths": ["a.py"]},
        {"ops": ["cmd.exec"], "paths": ["a.py"], "class": "test", "ok": False, "failing_test": True},
        {"ops": ["file.edit"], "paths": ["a.py"]},
        {"ops": ["cmd.exec"], "paths": ["a.py"], "is_test": True, "tests_failed": 1},
    ]
    assert thrashing_window(4, thrash)
    assert recovered({"op_outcomes": [{"class": "test", "ok": False}, {"class": "test", "ok": True}]})
    assert declared_success_without_observed_verification(0.8, False)
    assert not declared_success_without_observed_verification(0.8, True)
    assert investigation_then_change(0.9, 0.7, True)
    assert not investigation_then_change(0.9, 0.2, True)


def test_choice_outside_options_is_malformed(tmp_path: Path) -> None:
    q = {
        "task.intent.x": {
            "type": "choice",
            "instructions": {"question": "?"},
            "criteria": {
                "fix": {"what": "A fix request that is observable.", "examples": ["ex one", "ex two"]},
                "add": {"what": "An add request that is observable.", "examples": ["ex one", "ex two"]},
            },
        }
    }

    def bad(request: httpx2.Request) -> httpx2.Response:
        request.read()
        body = json.loads(request.content.decode())
        answers = {
            qid: {
                "type": "choice",
                "choice": "other",
                "confidence": 0.9,
                "probabilities": {"fix": 0.1, "add": 0.1, "other": 0.8},
            }
            for qid in body.get("questions") or {}
        }
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {}})

    asker = _asker(tmp_path, bad)
    with pytest.raises(SystemOnePromptsError) as ei:
        asker.ask({"x": 1}, q)
    assert "malformed" in str(ei.value)
    assert _cache_files(tmp_path) == 0
    asker.close()


def test_feature_value_abstains() -> None:
    mid = FeatureValue(id="episode.progress.novel_output", value=0.5, confidence=0.5, source="jev")
    assert mid.abstains
    low = FeatureValue(id="episode.progress.novel_output", value=0.9, confidence=0.9, source="jev")
    assert not low.abstains
    fact = FeatureValue(id="x", value=1, confidence=0.5, source="fact")
    assert not fact.abstains
    flagged = FeatureValue(id="x", reason="low_support", source="jev")
    assert flagged.abstains


def test_annotator_projects_light_episode_state(tmp_path: Path) -> None:
    from agent_hotwash.events import Capabilities
    from agent_hotwash.semantic.pipeline import Annotator

    handler = RecordingHandler()
    asker = _asker(tmp_path, handler)
    cfg = load_config()
    annotator = Annotator(asker, cfg, Capabilities(), mode="live", allow_unredacted=True)
    wanted = {
        "episode.phase.purpose",
        "episode.outcome.kind",
        "episode.impediment.kind",
    }
    feats = [f for f in load_bank() if f.id in wanted]
    state = {
        "task": {"request": "fix it", "amendments": [], "deliverables": [], "status": "open"},
        "episode": {
            "ops": [{"kind": "cmd.read", "cmd": "cat", "out_head": "LEAK", "out_tail": "LEAK", "exit": 0}],
            "messages": [
                {"kind": "user", "text": "hello"},
                {"kind": "assistant", "text": "done", "phase": "final_answer"},
            ],
            "position": "1 of 1",
            "prior_outcome": None,
            "counts": {"by_family": {"inspect": 1}, "majority_family": "inspect", "n_ops": 1, "n_final_answer": 1},
            "facts": {
                "env_impediment": "sandbox_denied",
                "verification": False,
                "artifact_change": False,
                "outstanding_failure": False,
                "error_kinds": [],
            },
        },
    }
    annotator.ask(state, feats)
    asker.close()
    assert len(handler.calls) == 2
    light_ids = set(handler.calls[0]["questions"])
    heavy_ids = set(handler.calls[1]["questions"])
    assert light_ids == {"episode.phase.purpose", "episode.outcome.kind"}
    assert heavy_ids == {"episode.impediment.kind"}
    light_blob = json.dumps(handler.calls[0]["state"])
    assert "LEAK" not in light_blob
    assert "majority_family" not in light_blob
    assert "env_impediment" not in light_blob
    assert "sandbox_denied" not in light_blob
    assert handler.calls[0]["state"]["episode"]["messages"][0]["text"] == "hello"
    assert handler.calls[0]["state"]["episode"]["messages"][0]["kind"] == "user"
    assert handler.calls[0]["state"]["episode"]["ops"][0]["cmd"] == "cat"
    heavy_blob = json.dumps(handler.calls[1]["state"])
    assert "LEAK" in heavy_blob


def test_annotator_5xx_degrades_to_api_error(tmp_path: Path, no_client_sleep: None) -> None:
    from agent_hotwash.events import Capabilities
    from agent_hotwash.semantic.pipeline import Annotator

    handler = RecordingHandler(fail=[503, 503, 503, 503])
    asker = _asker(tmp_path, handler, max_retries=3)
    cfg = load_config()
    annotator = Annotator(asker, cfg, Capabilities(), mode="live", allow_unredacted=True)
    feat = FeatureDef(
        id="task.intent.inquire",
        scope="task",
        primitive="noul",
        question="Does it inquire?",
        criteria=_valid_noul_criteria(),
    )
    out = annotator.ask({"task": {"request": "why?"}}, [feat])
    assert out[feat.id].reason == "api_error"
    asker.close()
