"""One positive + one negative synthetic Session per smell (12 smells)."""

from __future__ import annotations

from agent_hotwash.config import load_config
from agent_hotwash.detectors.smells import (
    _mutates_file,
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
from agent_hotwash.events import ArtifactInteraction, ArtifactOp, Event, EventKind, ToolCategory, Usage

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


def test_cold_start_reads_noop_without_a_write(dt):
    # A research/QA thread that never edits has no "cold start".
    reads = [dt.read(f"f{i}.py", call_id=f"r{i}") for i in range(20)]
    assert not cold_start_reads(dt.make([dt.user("go"), *reads]), CFG)


def test_cold_start_reads_counts_distinct_paths(dt):
    # Ten slices of the same file are one read; a compound read of two files is two.
    same = [dt.read("a.py", call_id=f"r{i}") for i in range(10)]
    assert not cold_start_reads(dt.make([dt.user("go"), *same, dt.edit("a.py", call_id="e")]), CFG)
    compound = [
        Event(
            kind=EventKind.tool_call,
            tool_name="cmd.read",
            op_kind="cmd.read",
            tool_category=ToolCategory.read,
            call_id=f"c{i}",
            tool_args={"command": f"sed -n 1,9p f{i}.py && cat g{i}.py"},
            artifacts=[
                ArtifactInteraction(path=f"/r/f{i}.py", op=ArtifactOp.read),
                ArtifactInteraction(path=f"/r/g{i}.py", op=ArtifactOp.read),
            ],
        )
        for i in range(5)
    ]
    found = cold_start_reads(dt.make([dt.user("go"), *compound, dt.edit("a.py", call_id="e")]), CFG)
    assert found and found[0].evidence["reads_before_first_edit"] == 10


def test_slow_to_action(dt):
    chatter = [dt.assistant(f"thought {i}") for i in range(8)]
    assert slow_to_action(dt.make([dt.user("go"), *chatter, dt.read("a.py", call_id="r")]), CFG)
    assert not slow_to_action(dt.make([dt.user("go"), dt.read("a.py", call_id="r")]), CFG)


def test_slow_to_action_counts_logical_position(dt):
    # Codex prelude: task_started, turn_context, user, reasoning, message, wrapper, usage -> call at raw idx 7
    metas = [Event(kind=EventKind.meta, raw_type=t) for t in ("task_started", "turn_context")]
    prelude = [
        *metas,
        dt.user("go"),
        dt.thinking(),
        dt.assistant("ok"),
        Event(kind=EventKind.meta, raw_type="custom_tool_call"),
    ]
    prelude.append(Event(kind=EventKind.meta, raw_type="token_usage_record"))
    sess = dt.make([*prelude, dt.read("a.py", call_id="r")])
    assert sess.events[-1].idx == 7
    assert not slow_to_action(sess, CFG)


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
    # Fewer than 10 calls is too little to call a mix a monoculture.
    assert not tool_monoculture(dt.make([dt.user("go"), *reads[:9]]), CFG)
    # The legacy `exec` wrapper name hides the real tool: ignored.
    execs = [dt.call("exec", call_id=f"x{i}", args={"cmd": "ls"}) for i in range(12)]
    assert not tool_monoculture(dt.make([dt.user("go"), *execs]), CFG)
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


def test_bash_as_editor_segment_anchored():
    # `-i` must be an argument of `sed`, not of a later command on another line.
    assert not _mutates_file("sed -n '1,240p' a.py\nprintf 'x'\nrg -n -i foo src")
    assert _mutates_file("cd src && sed -i 's/a/b/' file.py")
    assert _mutates_file("sed -i.bak 's/a/b/' file.py")
    # Operators that contain `>` are not redirects.
    assert not _mutates_file("node -e 'const h = x=>x.hostname; printf(h)'")
    assert not _mutates_file('psql -c "select * from t where a >= 5"')
    assert not _mutates_file("psql -c \"select payload->>'id' from t\" && printf done")
    # Heredoc into an interpreter is not a file write; heredoc into a file is.
    assert not _mutates_file("uv run python - <<'PY'\nx = 1 > 0\nprint(x)\nPY")
    assert _mutates_file("cat <<'EOF' > config.py\nX = 1\nEOF")
    # Scratch targets never count, even with a writer verb.
    assert not _mutates_file("echo hi > /tmp/scratch.py")
    assert not _mutates_file("printf 'x' > /private/tmp/out.md")
    assert not _mutates_file("echo hi > build.log")
    assert not _mutates_file("pytest 2>&1 | tee /tmp/run.log")
    assert not _mutates_file('deploy 2>&1 | tee "$evidence_dir/deploy.log"')
    assert not _mutates_file("lft query 'SELECT a->>1 FROM t' > logs/run-1/evidence.json")
    assert _mutates_file("lft query 'SELECT a FROM t' > fixtures/evidence.json")
    assert _mutates_file("printf 'x' > notes")  # writer verb to a bare project path
    assert _mutates_file("dd if=/dev/zero of=blob.bin")
    assert not _mutates_file("dd if=/dev/zero of=/tmp/blob.bin")


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
