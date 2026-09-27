"""Action grouping keeps failed-result facts and conservative decisions."""

from agent_hotwash.primitives.failures import FailureProvenance, FailureRecord
from agent_hotwash.report.error_actions import ErrorActionAccumulator, group_failure_actions


def _failure(
    leaf: str,
    *,
    trace: str = "trace",
    owner: str = "agent",
    disposition: str = "terminal",
    observation: bool = False,
) -> FailureRecord:
    return FailureRecord(
        provenance=FailureProvenance(stable_id=f"{trace}-{leaf}", trace_id=trace, session_id="s", event_idx=1),
        tool="exec_command",
        leaf=leaf,
        owner=owner,
        disposition=disposition,
        observation=observation,
        action="Repair the named issue.",
        diagnostic="The captured diagnostic.",
        result_excerpt="The captured diagnostic.",
    )


def test_groups_preserve_leaf_and_owner_counts_without_model_inference() -> None:
    failures = [
        _failure("test_assertion_failure", owner="product"),
        _failure("static_lint_finding"),
        _failure("cli_invocation_rejected"),
        _failure("required_target_missing"),
        _failure("edit_match_missing"),
        _failure("tool_contract_rejected"),
        _failure("unresolved", disposition="unresolved", owner="unknown"),
    ]
    groups = group_failure_actions([("trace", failures)])
    by_id = {group.id: group for group in groups}
    assert sum(group.count for group in groups) == len(failures)
    assert by_id["expected_iteration"].urgency == "low"
    assert by_id["expected_iteration"].leaf_counts == {
        "test_assertion_failure": 1,
        "static_lint_finding": 1,
    }
    assert by_id["expected_iteration"].owner_counts == {"product": 1, "agent": 1}
    assert by_id["context_cli"].leaf_counts == {
        "cli_invocation_rejected": 1,
        "required_target_missing": 1,
    }
    assert by_id["agent_tool_use"].count == 2
    assert by_id["unknown"].count == 1
    assert "model_choice" not in by_id


def test_expected_observation_and_bounded_samples() -> None:
    acc = ErrorActionAccumulator(examples_per_group=2)
    for index in range(5):
        acc.add(_failure("expected_no_match", trace=f"trace-{index}", disposition="expected", observation=True))
    group = acc.groups()[0]
    assert group.id == "expected_observation"
    assert group.count == 5
    assert group.disposition_counts == {"expected": 5}
    assert len(group.examples) == 2
    assert group.examples[0].trace_id == "trace-0"


def test_expected_flag_does_not_reclassify_check_as_task_scope() -> None:
    groups = group_failure_actions(
        [("trace", [_failure("static_lint_finding", disposition="expected", observation=True)])]
    )
    assert [group.id for group in groups] == ["expected_iteration"]


def test_product_failures_are_separate_from_routine_checks() -> None:
    groups = group_failure_actions(
        [("trace", [_failure("product_runtime_failure"), _failure("prompt_definition_finding")])]
    )
    by_id = {group.id: group for group in groups}
    assert by_id["product_defect"].urgency == "review"
    assert by_id["context_cli"].leaf_counts == {"prompt_definition_finding": 1}


def test_compact_run_dict_streams_without_full_validation() -> None:
    acc = ErrorActionAccumulator(examples_per_group=1)
    acc.add_run(
        {
            "analysis": {
                "trace_id": "fallback-trace",
                "failures": [
                    {
                        "leaf": "edit_match_missing",
                        "owner": "agent",
                        "disposition": "terminal",
                        "provenance": {"trace_id": "original-trace", "stable_id": "id-1"},
                        "diagnostic": "bad edit" * 100,
                        "action": "read current text",
                    },
                    {"leaf": "unresolved", "disposition": "unresolved"},
                ],
            }
        }
    )
    by_id = {group.id: group for group in acc.groups()}
    sample = by_id["agent_tool_use"].examples[0]
    assert sample.trace_id == "original-trace"
    assert sample.stable_id == "id-1"
    assert len(sample.diagnostic) == 240
    assert by_id["unknown"].examples[0].trace_id == "fallback-trace"
    summary = acc.summary()
    assert summary.total == 2
    assert summary.unknown_count == 1
    assert summary.unknown_share == 0.5
