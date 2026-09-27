"""Delegation identity, visibility, and lifecycle evidence boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.diagnostics.handover_cache import build_cache_waits
from agent_hotwash.events import (
    AgentKind,
    ArtifactInteraction,
    ArtifactOp,
    Event,
    EventKind,
    ModelCall,
    ModelConfig,
    Session,
    SourceRef,
    ThreadLink,
    ThreadLinkKind,
    Turn,
    Usage,
)
from agent_hotwash.primitives.handovers import build_handovers
from agent_hotwash.report.html import render_html
from agent_hotwash.report.model import Report, ReportMeta, RunResult


def test_pi_background_ack_is_not_completion_and_exact_id_joins(tf) -> None:
    parent = Session(
        session_id="parent",
        agent=AgentKind.pi,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="Agent",
                call_id="a",
                ts=None,
                source=SourceRef(record_index=4),
                tool_args={"prompt": "Implement src/a.py.", "subagent_type": "worker"},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                tool_name="Agent",
                call_id="a",
                ok=True,
                output="Agent started in background",
                tool_args={"agentId": "child"},
                source=SourceRef(record_index=5),
            ),
            Event(
                kind=EventKind.tool_call,
                idx=2,
                tool_name="steer_subagent",
                call_id="s",
                tool_args={"agent_id": "child", "message": "Also run make test."},
                source=SourceRef(record_index=6),
            ),
            Event(
                kind=EventKind.tool_call,
                idx=3,
                tool_name="get_subagent_result",
                call_id="w",
                tool_args={"agent_id": "child", "wait": True},
                source=SourceRef(record_index=7),
            ),
            Event(
                kind=EventKind.tool_result,
                idx=4,
                tool_name="get_subagent_result",
                call_id="w",
                ok=True,
                output="Status: completed\nImplemented src/a.py; make test passed.",
                source=SourceRef(record_index=8),
            ),
        ],
    )
    child = Session(session_id="child", agent=AgentKind.pi, parent_session_id="parent")
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.pi)
    trace.subagents = [child]
    trace.links = [
        ThreadLink(parent_id="parent", child_id="child", kind=ThreadLinkKind.spawn, evidence=["parentSession"])
    ]
    rows = build_handovers(trace)
    assert len(rows) == 1
    row = rows[0]
    assert row.child_id == "child"
    assert row.status == "completed"
    assert row.steer_count == 1
    assert row.request_chars == len("Implement src/a.py.")
    assert row.reply_chars is not None
    assert [event.kind for event in row.events].count("final") == 1


def test_codex_ciphertext_has_no_payload_size(tf) -> None:
    parent = Session(
        session_id="parent",
        agent=AgentKind.codex,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="agent.spawn",
                op_kind="agent.spawn",
                call_id="a",
                tool_args={"message": "gAAAA" + "x" * 200},
                source=SourceRef(record_index=10),
            ),
        ],
    )
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.codex)
    row = build_handovers(trace)[0]
    assert row.request_visibility == "encrypted"
    assert row.request_chars is None
    assert row.request_bytes is None
    assert row.status == "unknown"


def test_background_ack_alone_remains_running(tf) -> None:
    parent = Session(
        session_id="p",
        agent=AgentKind.pi,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="Agent",
                call_id="a",
                tool_args={"prompt": "Inspect src/a.py."},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                call_id="a",
                ok=True,
                output="Agent started in background",
                tool_args={"agentId": "agent-full-id"},
            ),
        ],
    )
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.pi)
    row = build_handovers(trace)[0]
    assert row.status == "running-at-capture"
    assert row.reply_visibility == "missing"


def test_parent_timing_survives_unjoined_child(tf) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    parent = Session(
        session_id="p",
        agent=AgentKind.pi,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="Agent",
                call_id="a",
                ts=start,
                tool_args={"prompt": "Inspect."},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                call_id="a",
                ts=start,
                output="Agent started in background",
                tool_args={"agentId": "agent-full-id"},
            ),
            Event(
                kind=EventKind.tool_call,
                idx=2,
                tool_name="get_subagent_result",
                call_id="w",
                ts=start + timedelta(seconds=5),
                tool_args={"agent_id": "agent-full-id"},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=3,
                call_id="w",
                ts=start + timedelta(seconds=8),
                output="Status: completed\nDone.",
            ),
        ],
    )
    row = build_handovers(tf.trace(parent, trace_id="trace", agent=AgentKind.pi))[0]
    assert row.child_id is None
    assert row.spawn_to_final_seconds == 8
    assert row.final_to_consumption_seconds == 0


def test_cache_wait_pairs_each_poll_with_its_own_result(tf) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    parent = Session(
        session_id="p",
        agent=AgentKind.pi,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="Agent",
                call_id="a",
                ts=start,
                tool_args={"prompt": "Inspect."},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                call_id="a",
                ts=start,
                output="Agent started in background",
                tool_args={"agentId": "child"},
            ),
            Event(
                kind=EventKind.tool_call,
                idx=2,
                tool_name="get_subagent_result",
                call_id="w1",
                ts=start + timedelta(seconds=1),
                tool_args={"agent_id": "child"},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=3,
                call_id="w1",
                ts=start + timedelta(seconds=2),
                output="Status: running",
            ),
            Event(
                kind=EventKind.tool_call,
                idx=4,
                tool_name="get_subagent_result",
                call_id="w2",
                ts=start + timedelta(seconds=10),
                tool_args={"agent_id": "child"},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=5,
                call_id="w2",
                ts=start + timedelta(seconds=14),
                output="Status: completed\nDone.",
            ),
        ],
    )
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.pi)
    waits = build_cache_waits(trace, build_handovers(trace), min_write_tokens=10000)
    assert [wait.wait_seconds for wait in waits] == [1, 4]


def test_handover_excerpt_masks_token_but_keeps_plaintext_size(tf) -> None:
    secret = "apikey_" + "A" * 40
    message = f"Inspect src/a.py with {secret}"
    parent = Session(
        session_id="p",
        agent=AgentKind.pi,
        events=[
            Event(kind=EventKind.tool_call, idx=0, tool_name="Agent", call_id="a", tool_args={"prompt": message}),
        ],
    )
    row = build_handovers(tf.trace(parent, trace_id="trace", agent=AgentKind.pi))[0]
    assert row.request_chars == len(message)
    assert secret not in (row.request_excerpt or "")


def test_cache_transition_requires_comparable_usage(tf) -> None:
    parent = Session(
        session_id="parent",
        agent=AgentKind.pi,
        model="m",
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="Agent",
                call_id="a",
                tool_args={"prompt": "Inspect code."},
                source=SourceRef(record_index=1),
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                call_id="a",
                ok=True,
                output="Agent started in background",
                tool_args={"agentId": "child"},
            ),
            Event(
                kind=EventKind.tool_call,
                idx=2,
                tool_name="get_subagent_result",
                call_id="w",
                tool_args={"agent_id": "child"},
            ),
            Event(kind=EventKind.tool_result, idx=3, call_id="w", ok=True, output="Status: completed\nDone."),
        ],
        turns=[
            Turn(
                turn_id="t",
                session_id="parent",
                model_config_active=ModelConfig(model="m"),
                model_calls=[
                    ModelCall(
                        response_id="before",
                        event_start=0,
                        event_end=1,
                        usage=Usage(input=100, cache_read=5000, cache_write=0),
                    ),
                    ModelCall(
                        response_id="after",
                        event_start=4,
                        event_end=5,
                        usage=Usage(input=120, cache_read=0, cache_write=12000),
                    ),
                ],
            )
        ],
    )
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.pi)
    handovers = build_handovers(trace)
    rows = build_cache_waits(trace, handovers, min_write_tokens=10000)
    assert len(rows) == 1
    assert rows[0].comparable is True
    assert rows[0].cache_rewrite_after_wait is True
    analysis = analyze(trace, load_config())
    html = render_html(Report.build([RunResult(analysis=analysis)], ReportMeta(tool_version="test")))
    assert "Parent cache use around child-result waits" in html
    assert "5000" in html and "12000" in html
    assert f"href='#handover-{analysis.cache_waits[0].handover_id}'" in html
    parent.usage_reliable = False
    assert build_cache_waits(trace, handovers, min_write_tokens=10000)[0].cache_rewrite_after_wait is None


def test_cache_wait_uses_issuing_response_and_immediate_next_response(tf):
    from agent_hotwash.canonical import build_turns

    events = [
        tf.user("Implement"),
        tf.tool("Agent", call_id="spawn", args={"prompt": "Work"}),
        tf.result(call_id="spawn", output="Agent started in background"),
    ]
    events[-1].tool_args = {"agentId": "child"}
    before = tf.with_usage(tf.assistant(), tf.usage(input=1, cache_read=40000, cache_write=100))
    before.source = SourceRef(record_index=4)
    call = tf.tool("get_subagent_result", call_id="wait", args={"agent_id": "child", "wait": True})
    call.source = SourceRef(record_index=4, ordinal=1)
    events += [before, call, tf.result(call_id="wait", output="Status: completed\nDone")]
    for i, read in enumerate((41000, 45000)):
        ev = tf.with_usage(tf.assistant(), tf.usage(input=2, cache_read=read, cache_write=200 + i))
        ev.source = SourceRef(record_index=6 + i)
        events.append(ev)
    parent = tf.session(events)
    parent.turns = build_turns(parent)
    trace = tf.trace(parent)
    handovers = build_handovers(trace)
    row = build_cache_waits(trace, handovers, min_write_tokens=10000)[0]
    assert row.prior_cache_read == 40000
    assert row.next_cache_read == 41000
    assert row.prior_event_start == 3 and row.next_event_start == 6
    assert row.comparable
    # Never substitute the following priced response for an immediate unknown one.
    parent.events[6].usage = None
    assert not build_cache_waits(trace, handovers, min_write_tokens=10000)[0].comparable


def test_cache_wait_respects_observed_model_and_missing_result(tf):
    from agent_hotwash.canonical import build_turns

    before = tf.with_usage(
        tf.tool("get_subagent_result", call_id="wait", args={"agent_id": "child"}),
        tf.usage(cache_read=100, cache_write=0),
    )
    before.source = SourceRef(record_index=4)
    before.usage_model = "model-before"
    after = tf.with_usage(tf.assistant(), tf.usage(cache_read=0, cache_write=20000))
    after.source = SourceRef(record_index=6)
    after.usage_model = "model-after"
    events = [
        tf.user("Work"),
        tf.tool("Agent", call_id="spawn", args={"prompt": "Work"}),
        tf.result(call_id="spawn", output="Agent started in background"),
        before,
        tf.result(call_id="wait", output="Status: completed\nDone"),
        after,
    ]
    events[2].tool_args = {"agentId": "child"}
    parent = tf.session(events)
    parent.turns = build_turns(parent)
    trace = tf.trace(parent)
    handovers = build_handovers(trace)
    row = build_cache_waits(trace, handovers, min_write_tokens=10000)[0]
    assert row.model_switch and not row.comparable
    after.usage_model = "model-before"
    parent.events[4].call_id = "missing"
    assert not build_cache_waits(trace, handovers, min_write_tokens=10000)[0].comparable


def test_html_ledger_links_observed_spawn_and_visibility(tf) -> None:
    parent = Session(
        session_id="p",
        agent=AgentKind.codex,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="agent.spawn",
                op_kind="agent.spawn",
                call_id="a",
                tool_args={"message": "gAAAA" + "x" * 200},
                source=SourceRef(record_index=10),
            ),
        ],
    )
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.codex)
    analysis = analyze(trace, load_config())
    html = render_html(Report.build([RunResult(analysis=analysis)], ReportMeta(tool_version="test")))
    assert f'id="handover-{analysis.handovers[0].id}"' in html
    assert "0/1 plaintext requests" in html
    assert "encrypted" in html
    assert "Runtime seconds p50/p95" in html
    assert "Full handover record" in html
    assert "Reply/request" in html


def test_html_handover_full_records_are_bounded(tf) -> None:
    parent = Session(
        session_id="p",
        agent=AgentKind.codex,
        events=[Event(kind=EventKind.tool_call, idx=0, tool_name="agent.spawn", op_kind="agent.spawn", call_id="a")],
    )
    analysis = analyze(tf.trace(parent, trace_id="trace", agent=AgentKind.codex), load_config())
    original = analysis.handovers[0]
    analysis.handovers = [
        original.model_copy(update={"id": f"h{i}", "spawn_to_final_seconds": float(i)}) for i in range(101)
    ]
    html = render_html(Report.build([RunResult(analysis=analysis)], ReportMeta(tool_version="test")))
    assert html.count("Full handover record</summary>") == 100
    assert html.count('class="error-example"') >= 101
    assert "All 101 handovers have navigable summaries" in html


def test_ordered_late_return_overlap_requires_common_file(tf) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    parent = Session(
        session_id="p",
        agent=AgentKind.pi,
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="Agent",
                call_id="a",
                ts=start,
                tool_args={"prompt": "Edit src/a.py."},
            ),
            Event(
                kind=EventKind.tool_result,
                idx=1,
                call_id="a",
                ok=True,
                ts=start,
                output="Agent started in background",
                tool_args={"agentId": "child"},
            ),
            Event(
                kind=EventKind.assistant_msg, idx=2, phase="final_answer", text="Done.", ts=start + timedelta(seconds=2)
            ),
            Event(
                kind=EventKind.tool_call,
                idx=3,
                tool_name="edit",
                ts=start + timedelta(seconds=4),
                artifacts=[ArtifactInteraction(path="src/a.py", op=ArtifactOp.update)],
            ),
        ],
    )
    child = Session(
        session_id="child",
        agent=AgentKind.pi,
        parent_session_id="p",
        events=[
            Event(
                kind=EventKind.tool_call,
                idx=0,
                tool_name="edit",
                ts=start + timedelta(seconds=1),
                artifacts=[ArtifactInteraction(path="src/a.py", op=ArtifactOp.update)],
            ),
            Event(kind=EventKind.assistant_msg, idx=1, text="Updated src/a.py.", ts=start + timedelta(seconds=3)),
        ],
    )
    trace = tf.trace(parent, trace_id="trace", agent=AgentKind.pi)
    trace.subagents = [child]
    trace.links = [ThreadLink(parent_id="p", child_id="child", kind=ThreadLinkKind.spawn)]
    row = build_handovers(trace)[0]
    assert row.return_after_parent_final is True
    assert row.late_return_overlap is True
