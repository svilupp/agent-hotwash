"""One positive + one negative synthetic Session per taxonomy detector (28)."""

from __future__ import annotations

from agent_hotwash.config import load_config
from agent_hotwash.detectors import taxonomy as tx
from agent_hotwash.events import ArtifactInteraction, ArtifactOp, Event, EventKind, ToolCategory

CFG = load_config()


def _codex_msg(text: str, phase: str) -> Event:
    return Event(kind=EventKind.assistant_msg, text=text, phase=phase)


def _task_complete() -> Event:
    return Event(kind=EventKind.meta, raw_type="task_complete")


def _bash_pair(cmd, call_id, dt, *, ok=True, exit_code=None, error_text=None):
    return [
        dt.bash(cmd, call_id=call_id),
        dt.result(call_id=call_id, ok=ok, exit_code=exit_code, error_text=error_text),
    ]


# 1
def test_context_rot(dt):
    pos = [
        dt.user("go"),
        dt.edit("a.py", call_id="e0"),
        dt.assistant("m"),
        dt.assistant("m"),
        dt.assistant("m"),
        dt.assistant("m"),
        dt.assistant("m"),
        dt.assistant("m"),
        dt.edit("a.py", call_id="e1"),
        dt.edit("a.py", call_id="e2"),
        dt.edit("a.py", call_id="e3"),
        dt.result(call_id="x", ok=False, error_text="boom"),
    ]
    assert tx.context_rot(dt.make(pos), CFG)
    neg = [dt.assistant("m") for _ in range(12)]
    assert not tx.context_rot(dt.make(neg), CFG)


# 2
def test_kitchen_sink(dt):
    pos = dt.make([dt.user("implement the parser"), dt.user("now switch to the exporter")])
    assert tx.kitchen_sink(pos, CFG)
    neg = dt.make([dt.user("implement the parser"), dt.user("thanks")])
    assert not tx.kitchen_sink(neg, CFG)


# 3
def test_correction_loop(dt):
    pos = dt.make([dt.user("no"), dt.user("wrong"), dt.user("still broken")])
    assert tx.correction_loop(pos, CFG)
    neg = dt.make([dt.user("no"), dt.user("wrong")])
    assert not tx.correction_loop(neg, CFG)


# 4
def test_edit_thrash(dt):
    pos = dt.make([dt.edit("a.py", call_id="e1"), dt.edit("a.py", call_id="e2"), dt.edit("a.py", call_id="e3")])
    assert tx.edit_thrash(pos, CFG)
    neg = dt.make(
        [
            dt.edit("a.py", call_id="e1"),
            dt.read("a.py", call_id="r1"),
            dt.edit("a.py", call_id="e2"),
            dt.read("a.py", call_id="r2"),
            dt.edit("a.py", call_id="e3"),
        ]
    )
    assert not tx.edit_thrash(neg, CFG)


# 5
def test_edit_without_read(dt):
    # The session must have read *something* (guard against read-less traces).
    pos = dt.make([dt.user("go"), dt.read("b.py", call_id="r0"), dt.edit("a.py", call_id="e1")])
    found = tx.edit_without_read(pos, CFG)
    assert found and found[0].evidence["path"] == "a.py"
    neg = dt.make([dt.read("a.py", call_id="r1"), dt.edit("a.py", call_id="e1")])
    assert not tx.edit_without_read(neg, CFG)
    # create-then-edit: a prior Write grounds the edit, so it must NOT fire.
    neg_created = dt.make([dt.write("a.py", call_id="w1"), dt.edit("a.py", call_id="e1")])
    assert not tx.edit_without_read(neg_created, CFG)


def test_edit_without_read_guard_when_session_never_reads(dt):
    # No reads and no read-capable calls at all: nothing to ground against -> silent.
    assert not tx.edit_without_read(dt.make([dt.user("go"), dt.edit("a.py", call_id="e1")]), CFG)


def _file_change(paths_ops: list[tuple[str, ArtifactOp]], *, call_id: str) -> Event:
    arts = [ArtifactInteraction(path=p, op=op) for p, op in paths_ops]
    return Event(
        kind=EventKind.tool_call,
        tool_name="file.edit",
        op_kind="file.edit",
        tool_category=ToolCategory.write,
        artifacts=arts,
        call_id=call_id,
    )


def _cmd_read(paths: list[str], *, call_id: str, op: ArtifactOp = ArtifactOp.read, kind: str = "cmd.read") -> Event:
    return Event(
        kind=EventKind.tool_call,
        tool_name=kind,
        op_kind=kind,
        tool_category=ToolCategory.read,
        artifacts=[ArtifactInteraction(path=p, op=op) for p in paths],
        tool_args={"command": f"cat {' '.join(paths)}", "cwd": "/repo"},
        call_id=call_id,
    )


def test_edit_without_read_skips_add_delete_move_artifacts(dt):
    # Pure adds / deletes are not edits of existing content.
    sess = dt.make(
        [
            _cmd_read(["/repo/z.py"], call_id="r0"),
            _file_change([("/repo/new.py", ArtifactOp.add)], call_id="fc1"),
            _file_change([("/repo/old.py", ArtifactOp.delete)], call_id="fc2"),
            _file_change([("/repo/moved.py", ArtifactOp.move)], call_id="fc3"),
        ]
    )
    assert not tx.edit_without_read(sess, CFG)


def test_edit_without_read_checks_every_artifact(dt):
    # A multi-file FileChange: the first path was read, the second was not.
    sess = dt.make(
        [
            _cmd_read(["/repo/a.py"], call_id="r0"),
            _file_change([("/repo/a.py", ArtifactOp.update), ("/repo/b.py", ArtifactOp.update)], call_id="fc1"),
        ]
    )
    found = tx.edit_without_read(sess, CFG)
    assert [f.evidence["path"] for f in found] == ["/repo/b.py"]


def test_edit_without_read_grounded_by_compound_read_naming_the_file(dt):
    # Codex leaves `sed … && rg …` as parsed type=unknown with no artifacts; the
    # command text still names the file the agent looked at.
    compound = Event(
        kind=EventKind.tool_call,
        tool_name="cmd.read",
        op_kind="cmd.read",
        tool_category=ToolCategory.read,
        tool_args={"command": "sed -n '1,200p' src/bootstrap.py && rg -n 'foo' tests", "cwd": "/repo"},
        call_id="r0",
    )
    sess = dt.make(
        [
            _cmd_read(["/repo/other.py"], call_id="rz"),
            compound,
            _file_change([("/repo/src/bootstrap.py", ArtifactOp.update)], call_id="fc1"),
        ]
    )
    assert not tx.edit_without_read(sess, CFG)


def test_edit_without_read_grounded_by_parent_dir_search(dt):
    # `rg foo src/` then editing src/x.py is grounded (search over a parent dir).
    sess = dt.make(
        [
            _cmd_read(["/repo/src"], call_id="r0", op=ArtifactOp.search, kind="cmd.search"),
            _file_change([("/repo/src/x.py", ArtifactOp.update)], call_id="fc1"),
        ]
    )
    assert not tx.edit_without_read(sess, CFG)
    # ...but only within the read window.
    filler = [dt.assistant(f"m{i}") for i in range(60)]
    far = dt.make(
        [
            _cmd_read(["/repo/src"], call_id="r0", op=ArtifactOp.search, kind="cmd.search"),
            *filler,
            _file_change([("/repo/src/x.py", ArtifactOp.update)], call_id="fc1"),
        ]
    )
    assert tx.edit_without_read(far, CFG)


# 6
def test_full_file_rewrite(dt):
    big = "line\n" * 60
    pos = dt.make([dt.read("a.py", call_id="r1"), dt.write("a.py", content=big, call_id="w1")])
    assert tx.full_file_rewrite(pos, CFG)
    neg = dt.make([dt.write("new.py", content=big, call_id="w1")])  # brand new file
    assert not tx.full_file_rewrite(neg, CFG)


# 7
def _storm(dt, n: int, *, fail_first: bool = True, cmd: str = "ls -la") -> list[Event]:
    evs: list[Event] = []
    for i in range(n):
        evs += _bash_pair(
            cmd, f"c{i}", dt=dt, ok=not (fail_first and i == 0), exit_code=1 if fail_first and i == 0 else 0
        )
    return evs


def test_retry_storm(dt):
    pos = dt.make(_storm(dt, 4))
    assert tx.retry_storm(pos, CFG)
    neg = dt.make(_storm(dt, 3))
    assert not tx.retry_storm(neg, CFG)


def test_retry_storm_requires_a_real_failure(dt):
    # Four identical successful calls (`git status` between edits) is not a storm.
    assert not tx.retry_storm(dt.make(_storm(dt, 4, fail_first=False)), CFG)
    # A benign no-match probe does not count as the failure.
    probes = []
    for i in range(4):
        probes += _bash_pair("rg needle src", f"p{i}", dt=dt, ok=False, exit_code=1)
    assert not tx.retry_storm(dt.make(probes), CFG)


def test_retry_storm_requires_dense_repeats(dt):
    # Same failing command 4x but spread over 200 events (an edit->test loop).
    evs: list[Event] = []
    for i in range(4):
        evs += _bash_pair("pytest", f"c{i}", dt=dt, ok=False, exit_code=1)
        evs += [dt.assistant(f"m{i}-{j}") for j in range(60)]
    assert not tx.retry_storm(dt.make(evs), CFG)


def test_retry_storm_edit_between_repeats_is_a_fix_loop(dt):
    evs: list[Event] = []
    for i in range(4):
        evs += _bash_pair("bun run typecheck", f"c{i}", dt=dt, ok=False, exit_code=1, error_text="TS2322 boom")
        evs.append(dt.edit("x.ts", call_id=f"e{i}"))
    assert not tx.retry_storm(dt.make(evs), CFG)


def test_retry_storm_ignores_empty_args_and_polling_ops(dt):
    # FileChange events share (`file.edit`, '') — never a storm.
    fcs = [_file_change([(f"/repo/f{i}.py", ArtifactOp.update)], call_id=f"fc{i}") for i in range(6)]
    assert not tx.retry_storm(dt.make([_cmd_read(["/repo"], call_id="r"), *fcs]), CFG)
    # Subagent waits and MCP wait/list polling repeat by design.
    waits = [
        dt.call("agent.wait", call_id=f"w{i}", args={"tool": "wait"}, category=ToolCategory.subagent) for i in range(6)
    ]
    waits += [dt.result(call_id="w0", ok=False, error_text="timeout")]
    assert not tx.retry_storm(dt.make(waits), CFG)
    polls = [dt.call("mcp.codex_app.wait_threads", call_id=f"m{i}", args={"ids": [1]}) for i in range(6)]
    for p in polls:
        p.op_kind = "mcp.codex_app.wait_threads"
    polls += [dt.result(call_id="m0", ok=False, error_text="boom")]
    assert not tx.retry_storm(dt.make(polls), CFG)


# 8
def test_no_adapt_retry(dt):
    pos = dt.make(
        _bash_pair("pytest foo", "c1", ok=False, exit_code=1, dt=dt)
        + _bash_pair("pytest foo", "c2", ok=False, exit_code=1, dt=dt)
    )
    assert tx.no_adapt_retry(pos, CFG)
    neg = dt.make(_bash_pair("pytest foo", "c1", ok=False, exit_code=1, dt=dt))
    assert not tx.no_adapt_retry(neg, CFG)


def test_no_adapt_retry_needs_consecutive_calls(dt):
    # An edit between two failing test runs is the normal fix loop, not a blind retry.
    fix_loop = dt.make(
        [
            *_bash_pair("pytest foo", "c1", ok=False, exit_code=1, dt=dt),
            dt.edit("foo.py", call_id="e1"),
            *_bash_pair("pytest foo", "c2", ok=False, exit_code=1, dt=dt),
        ]
    )
    assert not tx.no_adapt_retry(fix_loop, CFG)
    # Any other tool call between them also breaks the cluster.
    other = dt.make(
        [
            *_bash_pair("pytest foo", "c1", ok=False, exit_code=1, dt=dt),
            dt.read("foo.py", call_id="r1"),
            *_bash_pair("pytest foo", "c2", ok=False, exit_code=1, dt=dt),
        ]
    )
    assert not tx.no_adapt_retry(other, CFG)


def test_no_adapt_retry_short_commands_need_proportional_similarity(dt):
    # 4 edits on a 12-char command is a different script, not a blind retry.
    different = dt.make(
        _bash_pair("bun run test", "c1", ok=False, exit_code=1, dt=dt)
        + _bash_pair("bun run check", "c2", ok=False, exit_code=1, dt=dt)
    )
    assert not tx.no_adapt_retry(different, CFG)
    long_cmd = "uv run --env-file .env modal run modal_app.py::run_job --limit 5"
    tweaked = dt.make(
        _bash_pair(long_cmd, "c1", ok=False, exit_code=1, dt=dt)
        + _bash_pair(long_cmd.replace("::run_job", "::app.run_job"), "c2", ok=False, exit_code=2, dt=dt)
    )
    assert tx.no_adapt_retry(tweaked, CFG)


def test_no_adapt_retry_ignores_killed_processes_and_cwd(dt):
    killed = dt.make(
        _bash_pair("uv run server.py", "c1", ok=False, exit_code=143, dt=dt)
        + _bash_pair("uv run server.py", "c2", ok=False, exit_code=130, dt=dt)
    )
    assert not tx.no_adapt_retry(killed, CFG)
    interrupted = dt.make(
        [
            dt.bash("uv run server.py", call_id="c1"),
            dt.result(call_id="c1", ok=False, output="starting...\n^C"),
            *_bash_pair("uv run server.py", "c2", ok=False, exit_code=1, dt=dt),
        ]
    )
    assert not tx.no_adapt_retry(interrupted, CFG)
    # Codex exec args carry cwd/cmd duplicates; only the command is compared.
    a = dt.call("cmd.exec", call_id="x1", args={"command": "pytest foo", "cmd": "pytest foo", "cwd": "file:///a/b"})
    b = dt.call("cmd.exec", call_id="x2", args={"command": "pytest foo", "cmd": "pytest foo", "cwd": "file:///a/c/d"})
    a.tool_category = b.tool_category = ToolCategory.execute
    same_cmd = dt.make(
        [a, dt.result(call_id="x1", ok=False, exit_code=1), b, dt.result(call_id="x2", ok=False, exit_code=1)]
    )
    assert tx.no_adapt_retry(same_cmd, CFG)


# 9
def test_assuming_not_observing(dt):
    pos = dt.make([dt.user("go"), dt.assistant("the function returns null here")])
    assert tx.assuming_not_observing(pos, CFG)
    neg = dt.make([dt.user("go"), dt.read("a.py", call_id="r"), dt.assistant("the function returns null here")])
    assert not tx.assuming_not_observing(neg, CFG)


# 10
def test_test_gaming(dt):
    pos = dt.make(
        [
            *_bash_pair("pytest", "p1", ok=False, exit_code=1, dt=dt),
            dt.edit("test_foo.py", old="assert x == 1", new="@pytest.mark.skip\ndef test_x(): pass", call_id="e1"),
        ]
    )
    assert tx.test_gaming(pos, CFG)
    neg = dt.make(
        [dt.edit("test_foo.py", old="assert x == 1", new="@pytest.mark.skip\ndef test_x(): pass", call_id="e1")]
    )  # no failing run before
    assert not tx.test_gaming(neg, CFG)


# 11
def test_panic_revert(dt):
    pos = dt.make(
        [*_bash_pair("pytest", "p1", ok=False, exit_code=1, dt=dt), dt.bash("git reset --hard HEAD", call_id="g")]
    )
    assert tx.panic_revert(pos, CFG)
    neg = dt.make([dt.bash("git reset --hard HEAD", call_id="g")])  # no preceding fail
    assert not tx.panic_revert(neg, CFG)


# 12
def test_acting_on_question(dt):
    pos = dt.make([dt.user("why is this failing?"), dt.edit("a.py", call_id="e1")])
    assert tx.acting_on_question(pos, CFG)
    neg = dt.make([dt.user("why is this failing?"), dt.assistant("because the config is wrong")])
    assert not tx.acting_on_question(neg, CFG)


# 13
def test_permission_friction(dt):
    pos = dt.make(
        _bash_pair("sudo rm x", "c1", ok=False, error_text="Permission denied", dt=dt)
        + _bash_pair("sudo rm x", "c2", ok=False, error_text="Permission denied", dt=dt)
    )
    assert tx.permission_friction(pos, CFG)
    neg = dt.make(_bash_pair("sudo rm x", "c1", ok=False, error_text="Permission denied", dt=dt))
    assert not tx.permission_friction(neg, CFG)


# 14
def test_sandbox_egress_fail(dt):
    pos = dt.make(_bash_pair("curl x", "c1", ok=False, error_text="network egress blocked", dt=dt))
    assert tx.sandbox_egress_fail(pos, CFG)
    neg = dt.make(_bash_pair("curl x", "c1", ok=True, dt=dt))
    assert not tx.sandbox_egress_fail(neg, CFG)


# 15
def test_tool_arg_malformed(dt):
    pos = dt.make(
        [dt.call("Foo", call_id="c1"), dt.result(call_id="c1", ok=False, error_text="InputValidationError: bad")]
    )
    assert tx.tool_arg_malformed(pos, CFG)
    neg = dt.make([dt.call("Foo", call_id="c1"), dt.result(call_id="c1", ok=True)])
    assert not tx.tool_arg_malformed(neg, CFG)


def test_tool_arg_malformed_never_from_shell_stdout(dt):
    # ruff/ty/diff output containing "validation" / "required parameter" is not a harness rejection.
    for out in (
        "src/x.py:3: docstring mentions revalidation",
        "error[missing-argument]: No argument provided for required parameter `x`",
        "--- /private/tmp/wikow-astra-fixes-validation.m6boml/a.py",
    ):
        sess = dt.make(_bash_pair("ruff check .", "c1", ok=False, exit_code=1, error_text=out, dt=dt))
        assert not tx.tool_arg_malformed(sess, CFG), out
    # A Codex MCP isError / FunctionCallOutput rejection still counts.
    mcp = dt.make(
        [
            dt.call("mcp.srv.tool", call_id="m1", args={"x": 1}),
            dt.result(call_id="m1", ok=False, error_text="-32602 invalid params: required parameter 'y'"),
        ]
    )
    assert tx.tool_arg_malformed(mcp, CFG)


# 16
def test_mcp_transport_err(dt):
    pos = dt.make(
        [dt.call("mcp__x", call_id="c1"), dt.result(call_id="c1", ok=False, error_text="-32000 connection closed")]
    )
    assert tx.mcp_transport_err(pos, CFG)
    neg = dt.make([dt.call("mcp__x", call_id="c1"), dt.result(call_id="c1", ok=True)])
    assert not tx.mcp_transport_err(neg, CFG)


# 17
def test_rate_limit_loop(dt):
    pos = dt.make(
        _bash_pair("curl x", "c1", ok=False, error_text="429 overloaded", dt=dt)
        + _bash_pair("curl x", "c2", ok=False, error_text="429 overloaded", dt=dt)
    )
    assert tx.rate_limit_loop(pos, CFG)
    neg = dt.make(_bash_pair("curl x", "c1", ok=False, error_text="429 overloaded", dt=dt))
    assert not tx.rate_limit_loop(neg, CFG)


# 18
def test_perfectionism_loop(dt):
    pos = dt.make(
        [*_bash_pair("pytest", "p1", ok=True, dt=dt), dt.edit("a.py", call_id="e1"), dt.edit("a.py", call_id="e2")]
    )
    assert tx.perfectionism_loop(pos, CFG)
    neg = dt.make(
        [
            *_bash_pair("pytest", "p1", ok=True, dt=dt),
            dt.user("one more thing"),
            dt.edit("a.py", call_id="e1"),
            dt.edit("a.py", call_id="e2"),
        ]
    )
    assert not tx.perfectionism_loop(neg, CFG)


# 19
def test_over_engineering(dt):
    pos = dt.make(
        [dt.user("please fix the bug")] + [dt.write(f"n{i}.py", content="x", call_id=f"w{i}") for i in range(3)]
    )
    assert tx.over_engineering(pos, CFG)
    neg = dt.make([dt.user("please fix the bug"), dt.write("n0.py", content="x", call_id="w0")])
    assert not tx.over_engineering(neg, CFG)


# 20
def test_goal_drift(dt):
    pos = dt.make([dt.user("update config.py please"), dt.edit("unrelated.py", call_id="e1")])
    assert tx.goal_drift(pos, CFG)
    neg = dt.make([dt.user("update config.py please"), dt.edit("config.py", call_id="e1")])
    assert not tx.goal_drift(neg, CFG)


def test_goal_drift_recall_over_source_paths_vs_edits_only(dt):
    # Hostnames / e-mails / screenshots are not asked-for source files.
    no_source = dt.make(
        [dt.user("mail stepan.csiba@yahoo.com about prodej.wikov.app and 1-Photo-1.jpg"), dt.edit("x.py", call_id="e1")]
    )
    assert not tx.goal_drift(no_source, CFG)
    # Reads / directory listings do not dilute the comparison: only edits count.
    reads = [_cmd_read([p], call_id=f"r{i}") for i, p in enumerate(["/repo/.", "/repo/..", "/repo/src", "/repo/a.py"])]
    focused = dt.make([dt.user("update config.py please"), *reads, dt.edit("config.py", call_id="e1")])
    assert not tx.goal_drift(focused, CFG)
    # Two of two asked files edited plus extras: recall is 1.0 -> no drift.
    extras = [dt.edit(f"other{i}.py", call_id=f"o{i}") for i in range(6)]
    full = dt.make(
        [dt.user("update a.py and b.py"), dt.edit("a.py", call_id="e1"), dt.edit("b.py", call_id="e2"), *extras]
    )
    assert not tx.goal_drift(full, CFG)
    # Nothing edited at all: nothing to compare.
    assert not tx.goal_drift(dt.make([dt.user("update a.py"), dt.read("a.py", call_id="r")]), CFG)
    # Docs named in the ask are to be read, not edited.
    assert not tx.goal_drift(
        dt.make([dt.user("read docs/PLAN.md and README.md, then fix it"), dt.edit("x.py", call_id="e")]), CFG
    )
    drift = tx.goal_drift(dt.make([dt.user("update a.py and b.py"), dt.edit("zzz.py", call_id="e1")]), CFG)
    assert drift and drift[0].evidence["recall"] == 0.0 and drift[0].evidence["edited"] == ["zzz.py"]


# 21
def test_looks_right_runs_wrong(dt):
    pos = dt.make([dt.edit("a.py", call_id="e1"), dt.assistant("all done, fixed")])
    assert tx.looks_right_runs_wrong(pos, CFG)
    neg = dt.make(
        [
            dt.edit("a.py", call_id="e1"),
            *_bash_pair("pytest a.py", "p1", ok=True, dt=dt),
            dt.assistant("all done, fixed"),
        ]
    )
    assert not tx.looks_right_runs_wrong(neg, CFG)


# 22
def test_unverified_completion(dt):
    pos = dt.make([dt.edit("a.py", call_id="e1"), dt.assistant("done, should work")])
    assert tx.unverified_completion(pos, CFG)
    neg = dt.make(
        [dt.edit("a.py", call_id="e1"), *_bash_pair("pytest", "p1", ok=True, dt=dt), dt.assistant("done, should work")]
    )
    assert not tx.unverified_completion(neg, CFG)


def test_completion_detectors_ignore_doc_only_edits(dt):
    # CHANGELOG/.gitignore edits need no test run; a trailing doc edit does not
    # invalidate the test run that covered the last code edit.
    docs = dt.make([dt.edit("CHANGELOG.md", call_id="e1"), dt.edit(".gitignore", call_id="e2"), dt.assistant("done")])
    assert not tx.unverified_completion(docs, CFG)
    assert not tx.looks_right_runs_wrong(docs, CFG)
    code_then_doc = dt.make(
        [
            dt.edit("a.py", call_id="e1"),
            *_bash_pair("pytest", "p1", ok=True, dt=dt),
            dt.edit("CHANGELOG.md", call_id="e2"),
            dt.assistant("done"),
        ]
    )
    assert not tx.unverified_completion(code_then_doc, CFG)


def test_verification_recognises_modern_runners(dt):
    for cmd in (
        "bunx vitest run src",
        "bun run check",
        "make check",
        "python -m pytest -q",
        "ruff check .",
        "ty check",
        "cargo test",
        "go test ./...",
        "browser-pilot eval flows/login.toml",
        "uv run python -m pytest tests",
    ):
        sess = dt.make(
            [dt.edit("src/thing.py", call_id="e1"), *_bash_pair(cmd, "p1", ok=True, dt=dt), dt.assistant("done, fixed")]
        )
        assert not tx.unverified_completion(sess, CFG), cmd
        # Project-wide runners exercise every edited file (no basename needed).
        assert not tx.looks_right_runs_wrong(sess, CFG), cmd


def test_looks_right_file_targeted_runs(dt):
    # A test file for the edited module counts (stem match), an unrelated one does not.
    related = dt.make(
        [
            dt.edit("src/parser.py", call_id="e1"),
            *_bash_pair("pytest tests/test_parser.py", "p1", ok=True, dt=dt),
            dt.assistant("done"),
        ]
    )
    assert not tx.looks_right_runs_wrong(related, CFG)
    unrelated = dt.make(
        [
            dt.edit("src/parser.py", call_id="e1"),
            *_bash_pair("pytest tests/test_other.py", "p1", ok=True, dt=dt),
            dt.assistant("done"),
        ]
    )
    assert tx.looks_right_runs_wrong(unrelated, CFG)
    assert not tx.unverified_completion(unrelated, CFG)  # a test did run


def test_completion_claim_ignores_codex_commentary(dt):
    # "done" inside mid-turn narration is not a completion claim; the final answer is.
    narrated = dt.make([dt.edit("a.py", call_id="e1"), _codex_msg("Done reading, editing next.", "commentary")])
    assert not tx.unverified_completion(narrated, CFG)
    assert not tx.looks_right_runs_wrong(narrated, CFG)
    final = dt.make([dt.edit("a.py", call_id="e1"), _codex_msg("All done and fixed.", "final_answer")])
    assert tx.unverified_completion(final, CFG)
    assert tx.looks_right_runs_wrong(final, CFG)


# 23
def test_silent_error_swallow(dt):
    pos = dt.make(
        [*_bash_pair("ls missing", "c1", ok=False, exit_code=2, dt=dt), dt.assistant("moving on to the next section")]
    )
    assert tx.silent_error_swallow(pos, CFG)
    neg = dt.make(
        [*_bash_pair("ls missing", "c1", ok=False, exit_code=2, dt=dt), dt.assistant("that failed, let me fix it")]
    )
    assert not tx.silent_error_swallow(neg, CFG)


def test_silent_error_swallow_scans_past_codex_commentary(dt):
    # Codex narrates ("checking the config") then keeps working: not swallowed.
    fail = _bash_pair("pytest", "c1", ok=False, exit_code=1, dt=dt)
    kept_working = dt.make(
        [
            *fail,
            _codex_msg("Now checking the config.", "commentary"),
            dt.edit("conf.py", call_id="e1"),
            _codex_msg("All good.", "final_answer"),
        ]
    )
    assert not tx.silent_error_swallow(kept_working, CFG)
    # Commentary that acknowledges the failure also clears it.
    acked = dt.make(
        [*fail, _codex_msg("The probe timed out, moving on.", "commentary"), _codex_msg("Done.", "final_answer")]
    )
    assert not tx.silent_error_swallow(acked, CFG)
    # Commentary, then a final answer that never mentions it: swallowed.
    swallowed = dt.make(
        [*fail, _codex_msg("Looking at the layout.", "commentary"), _codex_msg("Shipped.", "final_answer")]
    )
    found = tx.silent_error_swallow(swallowed, CFG)
    assert found and found[0].spans[0].end_idx == swallowed.events[-1].idx
    # A turn that ends (task_complete) with only commentary has no terminal answer to blame.
    aborted = dt.make([*fail, _codex_msg("Looking.", "commentary"), _task_complete()])
    assert not tx.silent_error_swallow(aborted, CFG)


def test_silent_error_swallow_ignores_benign_failures(dt):
    # rg no-match (exit 1), read command exiting 1 with tiny output, killed process.
    probe = dt.make([*_bash_pair("rg needle src", "c1", ok=False, exit_code=1, dt=dt), dt.assistant("Moving on.")])
    assert not tx.silent_error_swallow(probe, CFG)
    read = _cmd_read(["/repo/a.py"], call_id="r1")
    read.tool_args = {"command": "sed -n '1,40p' a.py && ./probe", "cwd": "/repo"}
    tiny = dt.make([read, dt.result(call_id="r1", ok=False, exit_code=1, output="nope"), dt.assistant("Moving on.")])
    assert not tx.silent_error_swallow(tiny, CFG)
    # ...but a short "No such file" on a read is real signal, not a probe.
    missing_read = _cmd_read(["/repo/gone.py"], call_id="r2")
    missing_read.tool_args = {"command": "cat gone.py", "cwd": "/repo"}
    missing = dt.make(
        [
            missing_read,
            dt.result(call_id="r2", ok=False, exit_code=1, error_text="cat: gone.py: No such file or directory"),
            dt.assistant("Moving on."),
        ]
    )
    assert tx.silent_error_swallow(missing, CFG)
    diffed = dt.make(
        [*_bash_pair("git diff --no-index --check a b", "c1", ok=False, exit_code=1, dt=dt), dt.assistant("Moving on.")]
    )
    assert not tx.silent_error_swallow(diffed, CFG)
    killed = dt.make([*_bash_pair("uv run server", "c1", ok=False, exit_code=143, dt=dt), dt.assistant("Moving on.")])
    assert not tx.silent_error_swallow(killed, CFG)
    # A real failure with a big output is still flagged.
    real = dt.make(
        [*_bash_pair("pytest", "c1", ok=False, exit_code=1, error_text="F" * 500, dt=dt), dt.assistant("Moving on.")]
    )
    assert tx.silent_error_swallow(real, CFG)


def test_ack_lexicon_extended():
    for text in ("the request timed out", "assertion mismatch", "module not found", "failed to connect"):
        assert tx._ACK_RE.search(text), text
    assert not tx._ACK_RE.search("all green, moving on")


# 24
def test_linear_scan(dt):
    pos = dt.make([dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(12)])
    assert tx.linear_scan(pos, CFG)
    neg = dt.make(
        [dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(12)]
        + [dt.call("Grep", call_id="g", args={"pattern": "x"})]
    )
    assert not tx.linear_scan(neg, CFG)


# 25
def test_compaction_amnesia(dt):
    pos = dt.make([dt.read("a.py", call_id="r1"), Event(kind=EventKind.compaction), dt.read("a.py", call_id="r2")])
    assert tx.compaction_amnesia(pos, CFG)
    neg = dt.make([dt.read("a.py", call_id="r1"), Event(kind=EventKind.compaction), dt.read("b.py", call_id="r2")])
    assert not tx.compaction_amnesia(neg, CFG)


def test_compaction_amnesia_one_finding_per_compaction_with_paths(dt):
    reads = [dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(5)]
    rereads = [dt.read(f"f{i}.py", call_id=f"rr{i}") for i in range(5)]
    sess = dt.make([*reads, dt.compaction(), *rereads])
    found = tx.compaction_amnesia(sess, CFG)
    assert len(found) == 1
    assert found[0].evidence["reread_count"] == 5
    assert found[0].evidence["reread_paths"] == [f"f{i}.py" for i in range(5)]
    assert found[0].spans[0].event_idx == sess.events[5].idx


def test_compaction_amnesia_scoped_to_turn_and_window(dt):
    # Re-read in a NEW user turn is the new task's read, not amnesia.
    new_turn = dt.make(
        [dt.read("a.py", call_id="r1"), dt.compaction(), dt.user("next task"), dt.read("a.py", call_id="r2")]
    )
    assert not tx.compaction_amnesia(new_turn, CFG)
    # Re-read far after the compaction (> window) does not count.
    filler = [dt.assistant(f"m{i}") for i in range(30)]
    late = dt.make([dt.read("a.py", call_id="r1"), dt.compaction(), *filler, dt.read("a.py", call_id="r2")])
    assert not tx.compaction_amnesia(late, CFG)
    # Different turn_id on the read vs the compaction: skipped.
    comp = dt.compaction()
    comp.turn_id = "t1"
    rr = dt.read("a.py", call_id="r2")
    rr.turn_id = "t2"
    assert not tx.compaction_amnesia(dt.make([dt.read("a.py", call_id="r1"), comp, rr]), CFG)


def test_compaction_amnesia_anchors_on_nearest_compaction(dt):
    # Two compactions: the re-read after the second anchors there, not on the first.
    sess = dt.make(
        [
            dt.read("a.py", call_id="r1"),
            dt.compaction(),
            *[dt.assistant(f"m{i}") for i in range(5)],
            dt.compaction(),
            dt.read("a.py", call_id="r2"),
        ]
    )
    found = tx.compaction_amnesia(sess, CFG)
    assert len(found) == 1
    assert found[0].spans[0].event_idx == sess.events[7].idx


# 26
def test_style_imposition(dt):
    pos = dt.make(
        [
            dt.read("a.py", call_id="r1"),
            dt.result(call_id="r1", ok=True, output="import os\nx = 1"),
            dt.edit("a.py", old="x = 1", new="import requests\nx = 1", call_id="e1"),
        ]
    )
    assert tx.style_imposition(pos, CFG)
    neg = dt.make(
        [
            dt.read("a.py", call_id="r1"),
            dt.result(call_id="r1", ok=True, output="import os\nx = 1"),
            dt.edit("a.py", old="x = 1", new="import os\nx = 2", call_id="e1"),
        ]
    )
    assert not tx.style_imposition(neg, CFG)


# 27
def test_credential_leak(dt):
    pos = dt.make([dt.assistant("token is AKIA1234567890ABCDEF here")])
    assert tx.credential_leak(pos, CFG)
    neg = dt.make([dt.assistant("no secrets in this message")])
    assert not tx.credential_leak(neg, CFG)


# 28
def test_runaway_session(dt):
    pos = dt.make([dt.assistant(f"m{i}") for i in range(401)])
    assert tx.runaway_session(pos, CFG)
    neg = dt.make([dt.user("go"), dt.assistant("ok"), dt.user("thanks that works great")])
    assert not tx.runaway_session(neg, CFG)


def test_runaway_session_positive_outcome_from_final_answer(dt):
    many = [dt.assistant(f"m{i}") for i in range(401)]
    # Codex: the only user message is the task; a final_answer closes the turn positively.
    concluded = dt.make([dt.user("do the thing"), *many, _codex_msg("Done.", "final_answer")])
    assert not tx.runaway_session(concluded, CFG)
    marked = dt.make([dt.user("do the thing"), *many, _task_complete()])
    assert not tx.runaway_session(marked, CFG)
    # Commentary only, never concluded -> runaway.
    open_turn = dt.make([dt.user("do the thing"), *many, _codex_msg("still going", "commentary")])
    assert tx.runaway_session(open_turn, CFG)
    # The last user message being the task itself is NOT a positive reply, even if it says "works".
    task_only = dt.make([dt.user("make it works great please"), *many])
    assert tx.runaway_session(task_only, CFG)


def test_runaway_session_minutes_exclude_idle_gaps(dt):
    # 30 events 5 min apart = 145 active minutes -> over the 90-minute cap.
    busy = dt.make([dt.user("go"), *[dt.assistant(f"m{i}") for i in range(29)]], with_timestamps=True, gap_minutes=5.0)
    assert (tx._active_minutes(busy, 10.0) or 0.0) > 90
    assert tx.runaway_session(busy, CFG)
    # 3 events with an overnight gap: wall clock 600 min, active 0 -> not runaway.
    idle = dt.make([dt.user("go"), dt.assistant("ok"), dt.assistant("later")], with_timestamps=True, gap_minutes=300.0)
    assert (tx._minutes(idle) or 0.0) > 90
    assert tx._active_minutes(idle, 10.0) == 0.0
    assert not tx.runaway_session(idle, CFG)


def test_time_gated_detectors_noop_without_timestamps(dt):
    # RUNAWAY_SESSION minutes sub-condition must not fire on event count alone
    # when it is the only signal — here we assert the minutes path stays inert.
    sess = dt.make([dt.user("go"), dt.assistant("ok")])
    assert tx._minutes(sess) is None
    assert tx._active_minutes(sess, 10.0) is None
