"""Registry + run_detectors behaviour: registration, ordering, determinism,
enable/disable, and severity overrides."""

from __future__ import annotations

import copy

from agent_hotwash.config import Config, load_config
from agent_hotwash.detectors import get_registry, run_detectors
from agent_hotwash.detectors.registry import Severity, severity_rank
from agent_hotwash.events import AgentKind, Provenance, Trace


def _cfg(**over) -> Config:
    base = load_config()
    if not over:
        return base
    data = base.model_dump()
    for k, v in over.items():
        data.setdefault("detectors", {})
        data["detectors"][k] = v
    return Config.model_validate(data)


def test_registry_has_12_smells_and_28_taxonomies():
    reg = get_registry()
    smells = [s for s in reg.values() if s.kind == "smell"]
    tax = [s for s in reg.values() if s.kind == "taxonomy"]
    assert len(smells) == 12
    assert len(tax) == 28


def test_five_fuzzy_taxonomies_are_low_confidence_llm_candidates():
    reg = get_registry()
    fuzzy = {k for k, v in reg.items() if v.default_confidence == "low"}
    assert fuzzy == {
        "CONTEXT_ROT",
        "ASSUMING_NOT_OBSERVING",
        "OVER_ENGINEERING",
        "GOAL_DRIFT",
        "STYLE_IMPOSITION",
    }
    assert all(reg[k].llm_candidate for k in fuzzy)
    assert all(v.tier == "rule" for v in reg.values())


def test_credential_leak_fires_and_is_high_severity(dt):
    sess = dt.make([dt.assistant("here is the key AKIA1234567890ABCDEF done")])
    findings = run_detectors(sess, load_config())
    leak = [f for f in findings if f.id == "CREDENTIAL_LEAK"]
    assert leak and leak[0].severity == Severity.high
    assert leak[0].session_id == sess.session_id


def test_is_edit_tool_includes_codex_file_change(dt):
    from agent_hotwash.detectors.registry import is_edit_tool
    from agent_hotwash.events import ToolCategory

    fc = dt.call("file_change", call_id="c1", args={"path": "a.py"}, category=ToolCategory.write)
    edit = dt.call("Edit", call_id="c2", args={"file_path": "a.py"}, category=ToolCategory.write)
    write = dt.call("Write", call_id="c3", args={"file_path": "a.py"}, category=ToolCategory.write)
    assert is_edit_tool(fc)
    assert is_edit_tool(edit)
    assert not is_edit_tool(write)  # full Write is not a targeted edit


def _fc(ops, *, op_kind="file.edit"):
    from agent_hotwash.events import ArtifactInteraction, Event, EventKind, ToolCategory

    return Event(
        kind=EventKind.tool_call,
        tool_name=op_kind,
        op_kind=op_kind,
        tool_category=ToolCategory.write,
        artifacts=[ArtifactInteraction(path=f"/r/{i}.py", op=op) for i, op in enumerate(ops)],
    )


def test_is_edit_tool_false_for_pure_add_or_delete_artifacts():
    from agent_hotwash.detectors.registry import edited_paths, is_edit_tool
    from agent_hotwash.events import ArtifactOp

    assert not is_edit_tool(_fc([ArtifactOp.add]))
    assert not is_edit_tool(_fc([ArtifactOp.delete]))
    assert not is_edit_tool(_fc([ArtifactOp.add, ArtifactOp.delete]))
    assert not is_edit_tool(_fc([ArtifactOp.add], op_kind="file.write"))
    assert not is_edit_tool(_fc([ArtifactOp.delete], op_kind="file.delete"))
    mixed = _fc([ArtifactOp.add, ArtifactOp.update])
    assert is_edit_tool(mixed)
    assert edited_paths(mixed) == ["/r/1.py"]  # only the in-place update
    assert edited_paths(_fc([ArtifactOp.move])) == []


def test_benign_failure_and_terminal_msgs_helpers(dt):
    from agent_hotwash.detectors.registry import is_benign_failure, is_killed_result, terminal_assistant_msgs
    from agent_hotwash.events import Event, EventKind

    sess = dt.make(
        [
            dt.bash("rg needle src", call_id="p"),
            dt.result(call_id="p", ok=False, exit_code=1),
            dt.bash("pytest", call_id="t"),
            dt.result(call_id="t", ok=False, exit_code=1, error_text="1 failed"),
            dt.bash("uv run srv", call_id="k"),
            dt.result(call_id="k", ok=False, exit_code=143),
            dt.bash("cmp a b", call_id="d"),
            dt.result(call_id="d", ok=False, exit_code=1, output="a b differ: byte 3"),
        ]
    )
    calls = {e.call_id: e for e in sess.events if e.kind is EventKind.tool_call}
    results = {e.call_id: e for e in sess.events if e.kind is EventKind.tool_result}
    assert is_benign_failure(results["p"], calls["p"])
    assert not is_benign_failure(results["t"], calls["t"])
    assert is_benign_failure(results["k"], calls["k"]) and is_killed_result(results["k"])
    assert is_benign_failure(results["d"], calls["d"])
    assert not is_benign_failure(dt.result(call_id="ok", ok=True))

    def msg(text, phase=None):
        return Event(kind=EventKind.assistant_msg, text=text, phase=phase)

    codex = dt.make(
        [
            dt.user("t1"),
            msg("c1", "commentary"),
            msg("f1", "final_answer"),
            dt.user("t2"),
            msg("c2", "commentary"),
            Event(kind=EventKind.meta, raw_type="task_complete"),
        ]
    )
    assert [e.text for e in terminal_assistant_msgs(codex)] == ["f1"]
    claude = dt.make([dt.user("t1"), msg("a"), msg("b"), dt.user("t2"), msg("c")])
    assert [e.text for e in terminal_assistant_msgs(claude)] == ["b", "c"]


def test_disabled_detector_does_not_run(dt):
    sess = dt.make([dt.assistant("key AKIA1234567890ABCDEF")])
    cfg = _cfg(disabled=["CREDENTIAL_LEAK"])
    ids = {f.id for f in run_detectors(sess, cfg)}
    assert "CREDENTIAL_LEAK" not in ids


def test_enabled_allowlist_runs_only_those(dt):
    sess = dt.make([dt.assistant("key AKIA1234567890ABCDEF")])
    cfg = _cfg(enabled=["CREDENTIAL_LEAK"])
    ids = {f.id for f in run_detectors(sess, cfg)}
    assert ids == {"CREDENTIAL_LEAK"}


def test_severity_override_applied(dt):
    sess = dt.make([dt.assistant("key AKIA1234567890ABCDEF")])
    cfg = _cfg(severity={"CREDENTIAL_LEAK": "low"})
    leak = [f for f in run_detectors(sess, cfg) if f.id == "CREDENTIAL_LEAK"]
    assert leak and leak[0].severity == Severity.low


def test_run_is_deterministic_and_nonmutating(dt):
    events = [
        dt.user("please fix the bug"),
        dt.read("a.py", call_id="r1"),
        dt.edit("a.py", call_id="e1"),
        dt.edit("a.py", call_id="e2"),
        dt.edit("a.py", call_id="e3"),
        dt.assistant("all done, should work"),
    ]
    sess = dt.make(events)
    snapshot = copy.deepcopy(sess.model_dump())
    cfg = load_config()
    first = [f.model_dump() for f in run_detectors(sess, cfg)]
    second = [f.model_dump() for f in run_detectors(sess, cfg)]
    assert first == second
    assert sess.model_dump() == snapshot  # detectors never mutate their input


def test_run_over_trace_tags_subagent_findings(dt):
    root = dt.make([dt.user("hi")], session_id="root")
    child = dt.make([dt.assistant("leaking AKIA1234567890ABCDEF")], session_id="child")
    trace = Trace(
        trace_id="t",
        agent=AgentKind.claude,
        root=root,
        subagents=[child],
        provenance=Provenance(
            source_format="claude_native", detector_confidence="high", root_path=__import__("pathlib").Path(".")
        ),
    )
    findings = run_detectors(trace, load_config())
    leaks = [f for f in findings if f.id == "CREDENTIAL_LEAK"]
    assert leaks and leaks[0].session_id == "child"


def test_severity_rank_ordering():
    assert (
        severity_rank(Severity.info)
        < severity_rank(Severity.low)
        < severity_rank(Severity.medium)
        < severity_rank(Severity.high)
    )
