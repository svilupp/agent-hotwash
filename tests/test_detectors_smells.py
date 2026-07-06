"""One positive + one negative synthetic Session per smell (12 smells)."""

from __future__ import annotations

from agent_hotwash.config import load_config
from agent_hotwash.detectors.smells import (
    bash_as_editor,
    bloated_opener,
    cold_start_reads,
    context_bloat_no_clear,
    idle_gap,
    linear_scan_search,
    no_plan_dive,
    overlong_trace,
    runaway_todo,
    slow_to_action,
    thin_prompt,
    tool_monoculture,
)
from agent_hotwash.events import Event, EventKind, Usage

CFG = load_config()


def _ids(findings):
    return {f.id for f in findings}


def test_bloated_opener(dt):
    big = "word " * 9000  # ~ len//4 tokens well over 2000
    assert bloated_opener(dt.make([dt.user(big)]), CFG)
    assert not bloated_opener(dt.make([dt.user("short prompt here")]), CFG)


def test_thin_prompt(dt):
    assert thin_prompt(dt.make([dt.user("fix it")]), CFG)
    assert not thin_prompt(dt.make([dt.user("please implement a full featured parser with tests and docs today")]), CFG)


def test_cold_start_reads(dt):
    reads = [dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(10)]
    assert cold_start_reads(dt.make([dt.user("go"), *reads, dt.edit("f0.py", call_id="e")]), CFG)
    assert not cold_start_reads(
        dt.make([dt.user("go"), dt.read("a.py", call_id="r"), dt.edit("a.py", call_id="e")]), CFG
    )


def test_slow_to_action(dt):
    chatter = [dt.assistant(f"thought {i}") for i in range(8)]
    assert slow_to_action(dt.make([dt.user("go"), *chatter, dt.read("a.py", call_id="r")]), CFG)
    assert not slow_to_action(dt.make([dt.user("go"), dt.read("a.py", call_id="r")]), CFG)


def test_overlong_trace(dt):
    many = [dt.assistant(f"m{i}") for i in range(401)]
    assert overlong_trace(dt.make(many), CFG)
    assert not overlong_trace(dt.make([dt.user("go"), dt.assistant("ok")]), CFG)


def _with_usage(ev: Event, **kw) -> Event:
    ev.usage = Usage(**kw)
    return ev


def test_context_bloat_no_clear(dt):
    hot = _with_usage(dt.assistant("big"), input=180_000)
    assert context_bloat_no_clear(dt.make([dt.user("go"), hot]), CFG)
    # with a compaction present -> suppressed
    comp = Event(kind=EventKind.compaction)
    hot2 = _with_usage(dt.assistant("big"), input=180_000)
    assert not context_bloat_no_clear(dt.make([dt.user("go"), comp, hot2]), CFG)
    # no usage -> no-op
    assert not context_bloat_no_clear(dt.make([dt.user("go"), dt.assistant("small")]), CFG)


def test_tool_monoculture(dt):
    reads = [dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(10)]
    assert tool_monoculture(dt.make([dt.user("go"), *reads]), CFG)
    mixed = [
        dt.read("a.py", call_id="r"),
        dt.edit("a.py", call_id="e"),
        dt.bash("ls", call_id="b"),
        dt.write("c.py", call_id="w"),
        dt.read("d.py", call_id="r2"),
    ]
    assert not tool_monoculture(dt.make([dt.user("go"), *mixed]), CFG)


def test_linear_scan_search(dt):
    reads = [dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(12)]
    assert linear_scan_search(dt.make([dt.user("go"), *reads]), CFG)
    with_grep = [*reads, dt.call("Grep", call_id="g", args={"pattern": "x"})]
    assert not linear_scan_search(dt.make([dt.user("go"), *with_grep]), CFG)


def test_bash_as_editor(dt):
    assert bash_as_editor(dt.make([dt.bash("echo 'x' > file.py", call_id="b")]), CFG)
    assert bash_as_editor(dt.make([dt.bash("sed -i 's/a/b/' file.py", call_id="b")]), CFG)
    assert not bash_as_editor(dt.make([dt.bash("cat file.py", call_id="b")]), CFG)
    # benign redirection must NOT fire: /dev/null and scratch logs from test runs.
    assert not bash_as_editor(dt.make([dt.bash("pytest -q > /dev/null 2>&1", call_id="b")]), CFG)
    assert not bash_as_editor(dt.make([dt.bash("pytest > out.log", call_id="b")]), CFG)
    assert not bash_as_editor(dt.make([dt.bash("npm run build 2> /dev/null", call_id="b")]), CFG)
    # tee/patch to a real source file still fire.
    assert bash_as_editor(dt.make([dt.bash("echo 'x' | tee app.ts", call_id="b")]), CFG)


def test_no_plan_dive(dt):
    assert no_plan_dive(dt.make([dt.user("go"), dt.edit("a.py", call_id="e")]), CFG)
    planned = dt.make([dt.user("go"), dt.thinking("let me plan"), dt.edit("a.py", call_id="e")])
    assert not no_plan_dive(planned, CFG)


def test_runaway_todo(dt):
    grow1 = dt.call("TodoWrite", call_id="t1", args={"todos": [{"status": "pending"}, {"status": "in_progress"}]})
    grow2 = dt.call(
        "TodoWrite", call_id="t2", args={"todos": [{"status": "pending"}, {"status": "pending"}, {"status": "pending"}]}
    )
    assert runaway_todo(dt.make([dt.user("go"), grow1, grow2]), CFG)
    done = dt.call(
        "TodoWrite",
        call_id="t3",
        args={"todos": [{"status": "completed"}, {"status": "completed"}, {"status": "completed"}]},
    )
    assert not runaway_todo(dt.make([dt.user("go"), grow1, done]), CFG)


def test_idle_gap_timestamp_gated(dt):
    # Big gap with timestamps -> fires.
    sess = dt.make([dt.user("go"), dt.assistant("later")], with_timestamps=True, gap_minutes=30.0)
    assert idle_gap(sess, CFG)
    # Same events without timestamps -> no-op (not a false positive).
    sess2 = dt.make([dt.user("go"), dt.assistant("later")])
    assert not idle_gap(sess2, CFG)
