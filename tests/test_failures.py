"""Failed-result identity, literal facts, ordered leaves, and reconciliation."""

from __future__ import annotations

from agent_hotwash.analytics import analyze, apply_failure_features
from agent_hotwash.config import load_config
from agent_hotwash.events import AgentKind, Event, EventKind, Session, SourceRef, ToolCategory
from agent_hotwash.primitives.failures import build_failure_records, classify_failure
from agent_hotwash.semantic.results import FeatureSet, FeatureValue


def _session(command: str, output: str, *, exit_code: int = 1) -> Session:
    return Session(
        session_id="s",
        agent=AgentKind.codex,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                call_id="c1",
                tool_name="exec_command",
                tool_category=ToolCategory.execute,
                op_kind="cmd.exec",
                tool_args={"command": command},
                source=SourceRef(record_index=10, ordinal=0),
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                call_id="c1",
                tool_name="cmd.exec",
                ok=False,
                exit_code=exit_code,
                output=output,
                error_text=output,
                error_category="other",
                source=SourceRef(record_index=11, ordinal=1),
            ),
        ],
    )


def _noul(feature_id: str, value: float) -> FeatureValue:
    return FeatureValue(id=feature_id, value=value, source="jev")


def test_failed_result_has_stable_source_identity_and_named_permission_leaf() -> None:
    session = _session("bq query 'select 1'", "Permission bigquery.jobs.create denied")
    first = build_failure_records("trace", session)[0]
    second = build_failure_records("trace", session)[0]
    assert first.provenance.stable_id == second.provenance.stable_id
    assert first.provenance.record_index == 11
    assert first.signals["iam_permission_named"] is True
    assert first.leaf == "iam_permission_missing"


def test_plain_filesystem_permission_denial_is_not_iam() -> None:
    record = build_failure_records("trace", _session("cat /root/private", "Permission denied: /root/private"))[0]
    assert record.signals["iam_permission_named"] is False
    assert record.leaf == "unresolved"


def test_recovery_outcome_distinguishes_stop_from_continued_work() -> None:
    stopped = build_failure_records("trace", _session("false", "failed"))[0]
    assert stopped.recovery.outcome == "terminal_stop"

    continued_session = _session("false", "failed")
    continued_session.events.append(
        Event(
            kind=EventKind.tool_call,
            idx=2,
            call_id="c2",
            tool_name="exec_command",
            tool_category=ToolCategory.execute,
            tool_args={"command": "pwd"},
        )
    )
    continued = build_failure_records("trace", continued_session)[0]
    assert continued.recovery.outcome == "unresolved"


def test_expected_no_match_requires_semantic_intent_evidence() -> None:
    record = build_failure_records("trace", _session("rg policy_tag models", ""))[0]
    assert record.signals["no_match"] is True
    assert record.leaf == "unresolved"
    classify_failure(
        record,
        {
            "failure.intent.presence_probe": _noul("failure.intent.presence_probe", 0.95),
            "failure.intent.absence_acceptable": _noul("failure.intent.absence_acceptable", 0.9),
        },
    )
    assert record.leaf == "expected_no_match"
    assert record.observation is True


def test_low_support_noul_does_not_force_leaf() -> None:
    record = build_failure_records("trace", _session("rg policy_tag models", ""))[0]
    classify_failure(
        record,
        {
            "failure.intent.presence_probe": FeatureValue(
                id="failure.intent.presence_probe", value=0.5, reason="low_support", source="jev"
            ),
            "failure.intent.absence_acceptable": _noul("failure.intent.absence_acceptable", 0.9),
        },
    )
    assert record.leaf == "unresolved"


def test_truncated_result_preserves_unknown_instead_of_false() -> None:
    session = _session("make test", "partial output")
    session.events[1].output_truncated = True
    session.events[1].output_chars_original = 10_000
    record = build_failure_records("trace", session)[0]
    assert record.output_truncated is True
    assert record.original_size == 10_000
    assert record.signals["test_assertion"] is None
    assert record.signals["iam_permission_named"] is None


def test_apply_failure_features_reconciles_session_metrics(tf) -> None:
    session = _session("rg policy_tag models", "")
    trace = tf.trace(session, trace_id="trace", agent=AgentKind.codex)
    analysis = analyze(trace, load_config())
    stable_id = analysis.failures[0].provenance.stable_id
    features = FeatureSet(
        scope="failure",
        object_id=stable_id,
        values={
            "failure.intent.presence_probe": _noul("failure.intent.presence_probe", 0.95),
            "failure.intent.absence_acceptable": _noul("failure.intent.absence_acceptable", 0.95),
        },
    )
    apply_failure_features(analysis, [features])
    assert analysis.root.tool_error_count == 1
    assert analysis.root.expected_observation_count == 1
    assert analysis.root.unresolved_failure_count == 0
    assert analysis.root.failure_leaves == {"expected_no_match": 1}


def test_edit_match_and_tool_contract_are_distinct_from_missing_file() -> None:
    session = _session("", "Failed to find expected lines in src/app.py")
    session.events[0].tool_name = "edit"
    record = build_failure_records("trace", session)[0]
    assert record.leaf == "edit_match_missing"
    assert record.signals["missing_target"] is False
    assert record.owner == "agent"
    assert record.disposition == "terminal"

    session.events[1].error_text = "Validation error: missing required parameter old_string for src/app.py"
    record = build_failure_records("trace", session)[0]
    assert record.leaf == "tool_contract_rejected"


def test_repeated_same_diagnostic_groups_one_incident() -> None:
    session = _session("make check", "Formatter failed: src/app.py")
    session.events.extend(
        [
            session.events[0].model_copy(update={"idx": 2, "call_id": "c2"}),
            session.events[1].model_copy(update={"idx": 3, "call_id": "c2"}),
        ]
    )
    records = build_failure_records("trace", session)
    assert len(records) == 2
    assert len({r.incident_id for r in records}) == 1


def test_success_closes_failure_incident_span() -> None:
    session = _session("make check", "Formatter failed: src/app.py")
    successful_call = session.events[0].model_copy(update={"idx": 2, "call_id": "ok"})
    successful_result = session.events[1].model_copy(update={"idx": 3, "call_id": "ok", "ok": True})
    repeated_call = session.events[0].model_copy(update={"idx": 4, "call_id": "again"})
    repeated_result = session.events[1].model_copy(update={"idx": 5, "call_id": "again"})
    session.events.extend([successful_call, successful_result, repeated_call, repeated_result])
    rows = build_failure_records("trace", session)
    assert len(rows) == 2
    assert rows[0].incident_id != rows[1].incident_id


def test_long_dotted_identifier_does_not_backtrack_as_iam_permission() -> None:
    from agent_hotwash.primitives.failures import _IAM

    # The former nested class included '.', allowing exponentially many ways
    # to split a dotted identifier when the trailing permission text is absent.
    identifier = ".".join(["segment"] * 200)
    assert _IAM.search(identifier + ": ordinary build output") is None
    assert _IAM.search(identifier + " permission denied") is not None
    assert _IAM.search("Permission 'storage.objects.get' denied") is not None
