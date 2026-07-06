"""One positive + one negative synthetic Session per taxonomy detector (28)."""

from __future__ import annotations

from agent_hotwash.config import load_config
from agent_hotwash.detectors import taxonomy as tx
from agent_hotwash.events import Event, EventKind

CFG = load_config()


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
    pos = dt.make([dt.user("go"), dt.edit("a.py", call_id="e1")])
    assert tx.edit_without_read(pos, CFG)
    neg = dt.make([dt.read("a.py", call_id="r1"), dt.edit("a.py", call_id="e1")])
    assert not tx.edit_without_read(neg, CFG)
    # create-then-edit: a prior Write grounds the edit, so it must NOT fire.
    neg_created = dt.make([dt.write("a.py", call_id="w1"), dt.edit("a.py", call_id="e1")])
    assert not tx.edit_without_read(neg_created, CFG)


# 6
def test_full_file_rewrite(dt):
    big = "line\n" * 60
    pos = dt.make([dt.read("a.py", call_id="r1"), dt.write("a.py", content=big, call_id="w1")])
    assert tx.full_file_rewrite(pos, CFG)
    neg = dt.make([dt.write("new.py", content=big, call_id="w1")])  # brand new file
    assert not tx.full_file_rewrite(neg, CFG)


# 7
def test_retry_storm(dt):
    pos = dt.make([dt.bash("ls -la", call_id=f"c{i}") for i in range(4)])
    assert tx.retry_storm(pos, CFG)
    neg = dt.make([dt.bash("ls -la", call_id=f"c{i}") for i in range(3)])
    assert not tx.retry_storm(neg, CFG)


# 8
def test_no_adapt_retry(dt):
    pos = dt.make(
        _bash_pair("pytest foo", "c1", ok=False, exit_code=1, dt=dt)
        + _bash_pair("pytest foo", "c2", ok=False, exit_code=1, dt=dt)
    )
    assert tx.no_adapt_retry(pos, CFG)
    neg = dt.make(_bash_pair("pytest foo", "c1", ok=False, exit_code=1, dt=dt))
    assert not tx.no_adapt_retry(neg, CFG)


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


def test_time_gated_detectors_noop_without_timestamps(dt):
    # RUNAWAY_SESSION minutes sub-condition must not fire on event count alone
    # when it is the only signal — here we assert the minutes path stays inert.
    sess = dt.make([dt.user("go"), dt.assistant("ok")])
    assert tx._minutes(sess) is None
