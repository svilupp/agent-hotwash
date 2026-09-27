"""Failed tool-result facts, stable identity, and primary-leaf decisions.

The unit is one ``tool_result`` whose ``ok`` value is false.  Shell command
intent and failure cause intentionally remain separate: a validator written in
Python is still an ``other`` command even when it raises a data-shape error.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.events import Event, EventKind, Session, Usage
from agent_hotwash.primitives.commands import (
    classify_command,
    exit1_is_signal_free,
    is_compound,
    split_segments,
    strip_shell_wrapper,
)

TriState = bool | None


class FailureProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stable_id: str
    trace_id: str
    session_id: str
    call_id: str | None = None
    record_index: int | None = None
    ordinal: int | None = None
    event_idx: int


class FailureRecovery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attempts: int = 0
    outcome: Literal["success", "terminal_stop", "unresolved"] = "unresolved"
    elapsed_seconds: float | None = None
    tokens: dict[str, int | None] | None = None


class FailureRecord(BaseModel):
    """Report-facing classification of one failed result."""

    model_config = ConfigDict(extra="ignore")

    provenance: FailureProvenance
    tool: str
    op_kind: str | None = None
    target_path: str | None = None
    command: str | None = None
    command_type: str = "other"
    exit_code: int | None = None
    diagnostic: str
    result_excerpt: str
    original_size: int | None = None
    original_tokens: int | None = None
    output_truncated: bool = False
    legacy_category: str = "other"
    leaf: str = "unresolved"
    observation: bool = False
    action: str = "Inspect the captured diagnostic and add a supported cause boundary."
    signals: dict[str, TriState] = Field(default_factory=dict)
    feature_answers: dict[str, Any] = Field(default_factory=dict)
    flags: dict[str, bool] = Field(default_factory=dict)
    abstention_reason: str | None = None
    shell_segments: list[str] = Field(default_factory=list)
    failing_segment: str | None = None
    attribution_reliable: bool = False
    owner: str = "unknown"
    disposition: str = "unresolved"
    evidence_status: str = "insufficient"
    incident_id: str | None = None
    recovery: FailureRecovery = Field(default_factory=FailureRecovery)


_DIAGNOSTIC = re.compile(
    r"(?:error|exception|traceback|failed|denied|not found|no such|unknown (?:flag|option)|"
    r"unrecognized argument|invalid (?:argument|option|offset)|timed? out|deadline|"
    r"command not found|cannot find|missing|expected|assert)",
    re.I,
)
_AUTH_REFRESH = re.compile(
    r"(?:reauth|re-auth|refresh).*(?:interactive|stdin|tty)|interactive.*(?:unavailable|required)", re.I | re.S
)
_IAM = re.compile(
    r"(?:permission|role)\s+['\"]?[\w-]++(?:\.[\w-]++)++|"
    r"(?<![\w.-])[\w-]++(?:\.[\w-]++)++.*(?:permission|denied)|"
    r"(?:missing|required)\s+(?:permission|role)\s+['\"]?[\w.-]+",
    re.I,
)
_CLI_REJECT = re.compile(
    r"unknown (?:flag|option)|unrecognized arguments?|invalid (?:flag|option|argument|path)|unexpected argument", re.I
)
_READ_RANGE = re.compile(
    r"(?:offset|line|range).*(?:beyond|exceeds|past).*(?:length|end|file)|offset.*out of range", re.I | re.S
)
_EXEC_MISSING = re.compile(
    r"command not found|executable file not found|not recognized as an internal|"
    r"No such file or directory: ['\"][^'\"]+['\"]",
    re.I,
)
_FORMAT = re.compile(r"(?:would reformat|formatting|formatter|format --check|biome format|ruff format)", re.I)
_LINT = re.compile(r"\b(?:ruff|eslint|biome lint|clippy)\b|\b[A-Z]{1,4}\d{3,4}\b", re.I)
_TYPE = re.compile(r"\b(?:mypy|pyright|type error|typecheck|type-check|\bty check)\b", re.I)
_PROMPT = re.compile(r"prompt(?:[-_ ]definition| checker| bank).*(?:error|invalid|failed)|check_prompts", re.I | re.S)
_TEST = re.compile(r"(?:assertionerror|assert .*==|tests? failed|FAILURES|expected .+ (?:got|received))", re.I | re.S)
_COMPILE = re.compile(r"(?:compile error|compilation failed|cannot compile|syntaxerror|tsc.*error)", re.I)
_RUNTIME = re.compile(r"(?:traceback \(most recent call last\)|uncaught exception|runtimeerror|panic:)", re.I)
_DATA_SHAPE = re.compile(
    r"(?:keyerror|indexerror|typeerror:.*(?:subscript|indices|index)|cannot read propert)", re.I | re.S
)
_SQL_SCHEMA = re.compile(
    r"(?:no such (?:table|column)|unknown column|relation .+ does not exist|invalid identifier)", re.I
)
_DEPENDENCY = re.compile(
    r"(?:no module named|modulenotfounderror|cannot find module|package .+ not found|command not found)", re.I
)
_TIMEOUT = re.compile(r"timed? out|deadline exceeded|ETIMEDOUT", re.I)
_TRANSPORT = re.compile(
    r"ECONN(?:REFUSED|RESET)|network is unreachable|transport error|connection (?:closed|failed)|DNS", re.I
)
_MISSING_TARGET = re.compile(r"no such file|ENOENT|not found|does not exist", re.I)
_TOOL_CONTRACT = re.compile(
    r"(?:invalid (?:tool |function )?(?:arguments?|parameters?|schema)|"
    r"validation error|missing required (?:argument|parameter)|"
    r"unexpected keyword argument|failed to parse (?:function|tool) (?:arguments|call))",
    re.I,
)
_EDIT_MATCH = re.compile(
    r"(?:failed to find expected lines|could not find (?:the )?(?:expected|exact|old) (?:text|lines)|"
    r"patch (?:context|hunk) (?:does not match|failed)|old_string.*not found)",
    re.I,
)
_CHECK_SEGMENT = re.compile(r"\b(?:format-check|format|lint|typecheck|type-check|test|check|build)\b", re.I)
_DIAGNOSTIC_PATH = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+")
_NAMED_ARGUMENT = re.compile(r"(?:parameter|argument|field|schema)\s+['\"`]?\w+", re.I)


def _command(call: Event | None) -> str:
    if call is None:
        return ""
    for key in ("command", "cmd", "script"):
        value = call.tool_args.get(key)
        if isinstance(value, str):
            return value
    return ""


def _arg_path(args: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filename", "target"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _diagnostic(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip() and line.strip() != "Script completed"]
    selected = [line for line in lines if _DIAGNOSTIC.search(line)]
    if not selected:
        selected = lines[-3:]
    return "\n".join(selected[:6])[:1200] or "no error text captured; inspect the command and event span"


def _stable_id(trace_id: str, session: Session, result: Event) -> str:
    source = result.source
    key = "\x1f".join(
        [
            trace_id,
            session.session_id,
            str(source.record_index if source else ""),
            str(source.ordinal if source else ""),
            result.call_id or "",
        ]
    )
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _truth(
    pattern: re.Pattern[str], text: str, *, unknown_when_truncated: bool = True, truncated: bool = False
) -> TriState:
    if pattern.search(text):
        return True
    if unknown_when_truncated and truncated:
        return None
    return False


def _literal_signals(result: Event, command: str, raw: str, truncated: bool) -> dict[str, TriState]:
    diagnostic_present = bool(raw.strip()) and raw.strip() not in {"Script completed", "Command exited with code 1"}
    no_match: TriState = False
    if result.exit_code == 1 and command and exit1_is_signal_free(command) and not diagnostic_present and not truncated:
        no_match = True
    elif result.exit_code == 1 and command and (is_compound(command) or truncated):
        no_match = None
    test = lambda pattern: _truth(pattern, raw, truncated=truncated)  # noqa: E731
    signals: dict[str, TriState] = {
        "no_match": no_match,
        "auth_refresh_blocked": test(_AUTH_REFRESH),
        "iam_permission_named": test(_IAM),
        "cli_argument_rejected": test(_CLI_REJECT),
        "read_range_invalid": test(_READ_RANGE),
        "executable_missing": test(_EXEC_MISSING),
        "static_format": test(_FORMAT),
        "static_lint": test(_LINT),
        "static_type": test(_TYPE),
        "prompt_definition": test(_PROMPT),
        "test_assertion": test(_TEST),
        "compile_error": test(_COMPILE),
        "product_runtime_error": test(_RUNTIME),
        "data_shape_exception": test(_DATA_SHAPE),
        "sql_schema_exception": test(_SQL_SCHEMA),
        "missing_dependency": test(_DEPENDENCY),
        "timeout": result.exit_code == 143 or test(_TIMEOUT),
        "transport_failure": test(_TRANSPORT),
        "missing_target": test(_MISSING_TARGET),
    }
    return signals


_ACTIONS = {
    "credential_refresh_blocked": "Ask for an interactive credential refresh, then stop local credential searches.",
    "iam_permission_missing": "Show the denied permission, identity, and project; request authorized access.",
    "cli_invocation_rejected": "Use the accepted CLI syntax or the repository's documented command recipe.",
    "read_range_invalid": "Read a valid range or the whole file.",
    "command_unavailable": "Use an available compatible command or install the documented dependency.",
    "static_format_finding": "Run the repository formatter before the final check.",
    "static_lint_finding": "Fix the named lint rule or configure generated-file treatment.",
    "static_type_finding": "Fix the named type diagnostic and rerun the repository check.",
    "prompt_definition_finding": "Fix the named prompt entry and rerun the prompt checker.",
    "test_assertion_failure": "Show the test name and expected versus observed result, then repair the changed code.",
    "product_build_failure": "Show the failing build target and compiler diagnostic.",
    "product_runtime_failure": "Show the product exception and its normal-run call path.",
    "external_dependency_or_transport": "Show the blocked dependency and the next operational step.",
    "agent_validator_data_shape": "Inspect returned types and keys before indexing; document the output contract.",
    "agent_validator_sql_schema": "Inspect the schema before writing validation joins.",
    "expected_no_match": "Report the negative finding without spending a repair turn.",
    "expected_stop_precondition": "State the missing prerequisite and stop as requested.",
    "wrong_requested_identifier": "Copy the exact identifier from the task; check CLI syntax if needed.",
    "required_target_missing": "Repair the path or identifier, or surface the missing prerequisite as a blocker.",
    "tool_contract_rejected": "Correct the rejected tool arguments and retry the intended operation.",
    "edit_match_missing": "Read the current target text and rebase the edit on an exact matching context.",
    "compound_check_attributed": "Repair the named failing check segment and rerun the compound check.",
    "post_action_verifier_mismatch": "Repair the verifier contract while keeping the main action's success visible.",
}


def _sum_usage(events: list[Event]) -> dict[str, int | None] | None:
    rows: list[Usage] = [e.usage for e in events if e.usage is not None]
    if not rows:
        return None
    return {
        name: sum(v for row in rows if (v := getattr(row, name)) is not None)
        if any(getattr(row, name) is not None for row in rows)
        else None
        for name in ("input", "output", "cache_read")
    }


def _recovery(session: Session, result: Event, call: Event | None) -> FailureRecovery:
    later = [e for e in session.events if e.idx > result.idx]
    command = _command(call)
    attempts = 0
    success: Event | None = None
    calls = {e.call_id: e for e in later if e.kind is EventKind.tool_call and e.call_id}
    for ev in later:
        if ev.kind is not EventKind.tool_result:
            continue
        candidate = calls.get(ev.call_id or "")
        if command and _command(candidate).strip() == command.strip():
            attempts += 1
            if ev.ok is True:
                success = ev
                break
    end = success or (later[-1] if later else result)
    elapsed = None
    if result.ts is not None and end.ts is not None:
        elapsed = max(0.0, (end.ts - result.ts).total_seconds())
    window = [e for e in later if e.idx <= end.idx]
    later_tool_calls = [event for event in later if event.kind is EventKind.tool_call]
    outcome: Literal["success", "terminal_stop", "unresolved"] = (
        "success" if success else ("terminal_stop" if not later_tool_calls else "unresolved")
    )
    return FailureRecovery(attempts=attempts, outcome=outcome, elapsed_seconds=elapsed, tokens=_sum_usage(window))


def build_failure_records(trace_id: str, session: Session) -> list[FailureRecord]:
    """Join every failed result to its call and compute literal facts."""
    calls = {e.call_id: e for e in session.events if e.kind is EventKind.tool_call and e.call_id}
    records: list[FailureRecord] = []
    for result in session.events:
        if result.kind is not EventKind.tool_result or result.ok is not False:
            continue
        call = calls.get(result.call_id or "")
        command = _command(call)
        raw = result.error_text or result.output or ""
        truncated = result.output_truncated or result.output_tokens_original is not None
        segments = split_segments(strip_shell_wrapper(command)) if command else []
        source = result.source
        record = FailureRecord(
            provenance=FailureProvenance(
                stable_id=_stable_id(trace_id, session, result),
                trace_id=trace_id,
                session_id=session.session_id,
                call_id=result.call_id,
                record_index=source.record_index if source else None,
                ordinal=source.ordinal if source else None,
                event_idx=result.idx,
            ),
            tool=(call.tool_name if call else result.tool_name) or "?",
            op_kind=(call.op_kind if call else result.op_kind),
            target_path=(call.path or _arg_path(call.tool_args)) if call else None,
            command=command or None,
            command_type=classify_command(command),
            exit_code=result.exit_code,
            diagnostic=result.diagnostic_excerpt or _diagnostic(raw),
            result_excerpt=raw[:4000],
            original_size=result.output_chars_original,
            original_tokens=result.output_tokens_original,
            output_truncated=truncated,
            legacy_category=result.error_category or "other",
            signals=_literal_signals(result, command, raw, truncated),
            flags={
                "compound_status_ambiguous": len(segments) > 1,
                "multiple_signals": False,
                "partial_lookup": bool(
                    _EXEC_MISSING.search(raw) and re.search(r"\b(?:or|which|where)\b", command, re.I)
                ),
            },
            abstention_reason="semantic evidence unavailable",
            shell_segments=segments,
            attribution_reliable=len(segments) <= 1,
            recovery=_recovery(session, result, call),
        )
        record.signals["tool_contract_rejected"] = _truth(_TOOL_CONTRACT, raw, truncated=truncated)
        record.signals["edit_match_missing"] = _truth(_EDIT_MATCH, raw, truncated=truncated)
        if len(segments) > 1:
            matched = [segment for segment in segments if _CHECK_SEGMENT.search(segment) and _CHECK_SEGMENT.search(raw)]
            if len(matched) == 1:
                record.failing_segment = matched[0]
                record.attribution_reliable = True
        true_signals = sum(value is True for value in record.signals.values())
        record.flags["multiple_signals"] = true_signals > 1
        classify_failure(record, {})
        records.append(record)
    incidents: dict[tuple[str, str, str], str] = {}
    by_event = {record.provenance.event_idx: record for record in records}
    for event in session.events:
        if event.kind is not EventKind.tool_result:
            continue
        call = calls.get(event.call_id or "")
        tool = (call.tool_name if call else event.tool_name) or "?"
        command = _command(call)
        if event.ok is True:
            incidents = {key: value for key, value in incidents.items() if key[:2] != (tool, command)}
            continue
        record = by_event.get(event.idx)
        if record is None:
            continue
        signature = (record.tool, record.command or "", record.diagnostic.strip().lower())
        incident = incidents.get(signature)
        if incident is None:
            key = f"{trace_id}\x1f{session.session_id}\x1f{record.provenance.stable_id}"
            incident = hashlib.sha256(key.encode()).hexdigest()[:24]
            incidents[signature] = incident
        record.incident_id = incident
    return records


def _feature_bool(features: dict[str, Any], key: str) -> TriState:
    value = features.get(key)
    if value is None:
        return None
    reason = getattr(value, "reason", None)
    if reason is not None or bool(getattr(value, "abstains", False)):
        return None
    raw = getattr(value, "value", value)
    if isinstance(raw, dict):
        raw = raw.get("noul", raw.get("value"))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        positive = getattr(value, "positive_threshold", None) or 0.7
        negative = getattr(value, "negative_threshold", None)
        if negative is None:
            negative = 0.3
        if raw >= positive:
            return True
        if raw <= negative:
            return False
    return None


def classify_failure(record: FailureRecord, features: dict[str, Any]) -> FailureRecord:
    """Apply ordered primary-leaf rules. Unknown evidence never becomes false."""
    record.feature_answers = {
        key: (value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
        for key, value in features.items()
    }
    s = record.signals
    f = lambda suffix: _feature_bool(features, f"failure.{suffix}")  # noqa: E731
    leaf = "unresolved"
    observation = False

    if (
        f("intent.user_named_identifier") is True
        and f("intent.identifier_matches") is False
        and s.get("missing_target") is True
    ):
        leaf = "wrong_requested_identifier"
    elif f("intent.stop_if_missing") is True and s.get("missing_target") is True:
        leaf, observation = "expected_stop_precondition", True
    elif (
        (
            s.get("no_match") is True
            or (
                record.failing_segment is not None
                and record.attribution_reliable
                and record.exit_code == 1
                and classify_command(record.failing_segment) == "inspect"
            )
        )
        and f("intent.presence_probe") is True
        and f("intent.absence_acceptable") is True
    ):
        leaf, observation = "expected_no_match", True
    elif s.get("auth_refresh_blocked") is True and s.get("iam_permission_named") is True:
        record.abstention_reason = "combined authentication and permission diagnostics have no reliable order"
    elif s.get("auth_refresh_blocked") is True:
        leaf = "credential_refresh_blocked"
    elif s.get("iam_permission_named") is True:
        leaf = "iam_permission_missing"
    elif (
        s.get("tool_contract_rejected") is True
        and not record.command
        and (record.target_path or _NAMED_ARGUMENT.search(record.diagnostic))
    ):
        leaf = "tool_contract_rejected"
    elif (
        s.get("edit_match_missing") is True
        and not record.command
        and (record.target_path or _DIAGNOSTIC_PATH.search(record.diagnostic))
    ):
        leaf = "edit_match_missing"
    elif record.failing_segment and record.attribution_reliable and f("intent.project_check") is True:
        leaf = "compound_check_attributed"
    elif s.get("cli_argument_rejected") is True or f("check.invocation_rejected") is True:
        leaf = "cli_invocation_rejected"
    elif s.get("read_range_invalid") is True:
        leaf = "read_range_invalid"
    elif s.get("executable_missing") is True or s.get("missing_dependency") is True:
        leaf = "command_unavailable"
    elif f("intent.post_action_verifier") is True and f("intent.main_action_completed") is True:
        leaf = "post_action_verifier_mismatch"
    elif f("intent.agent_validator") is True and (
        s.get("data_shape_exception") is True or f("context.agent_data_assumption") is True
    ):
        leaf = "agent_validator_data_shape"
    elif f("intent.agent_validator") is True and (
        s.get("sql_schema_exception") is True or f("context.agent_sql_assumption") is True
    ):
        leaf = "agent_validator_sql_schema"
    elif f("intent.project_check") is True and (
        s.get("prompt_definition") is True or f("check.prompt_finding") is True
    ):
        leaf = "prompt_definition_finding"
    elif f("intent.project_check") is True and (s.get("static_format") is True or f("check.format_finding") is True):
        leaf = "static_format_finding"
    elif f("intent.project_check") is True and (s.get("static_lint") is True or f("check.lint_finding") is True):
        leaf = "static_lint_finding"
    elif f("intent.project_check") is True and (s.get("static_type") is True or f("check.type_finding") is True):
        leaf = "static_type_finding"
    elif f("intent.project_check") is True and (s.get("test_assertion") is True or f("check.test_assertion") is True):
        leaf = "test_assertion_failure"
    elif f("intent.project_check") is True and (s.get("compile_error") is True or f("check.product_compile") is True):
        leaf = "product_build_failure"
    elif s.get("product_runtime_error") is True and f("check.product_runtime") is True:
        leaf = "product_runtime_failure"
    elif s.get("timeout") is True or s.get("transport_failure") is True:
        leaf = "external_dependency_or_transport"
    elif (
        s.get("missing_target") is True
        and f("intent.user_named_identifier") is True
        and f("intent.absence_acceptable") is False
    ):
        leaf = "required_target_missing"

    record.leaf = leaf
    record.observation = observation
    record.disposition = "expected" if observation else ("unresolved" if leaf == "unresolved" else "terminal")
    record.evidence_status = "supported" if leaf != "unresolved" else "insufficient"
    record.owner = (
        "expected"
        if observation
        else "environment"
        if leaf in {"credential_refresh_blocked", "iam_permission_missing", "command_unavailable"}
        else "harness/provider"
        if leaf == "external_dependency_or_transport"
        else "product"
        if leaf in {"test_assertion_failure", "product_build_failure", "product_runtime_failure"}
        else "agent"
        if leaf != "unresolved"
        else "unknown"
    )
    record.action = _ACTIONS.get(leaf, "Inspect the captured diagnostic and add a supported cause boundary.")
    if leaf != "unresolved":
        record.abstention_reason = None
    elif record.flags.get("compound_status_ambiguous"):
        record.abstention_reason = "compound command status cannot be attributed to a segment"
    elif any(value is None for value in s.values()):
        record.abstention_reason = "required literal evidence is truncated or ambiguous"
    return record


__all__ = ["FailureProvenance", "FailureRecord", "FailureRecovery", "build_failure_records", "classify_failure"]
