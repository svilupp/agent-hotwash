"""Codex decoder v2 (item_completed-first) + linkage + usage identities."""

from __future__ import annotations

from pathlib import Path

from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.events import EventKind, RoleHint, TurnStatus
from agent_hotwash.sources.codex_native import load_rollout
from agent_hotwash.sources.detect import build_codex_forest, iter_traces

FIXTURES = Path(__file__).parent / "fixtures" / "codex_native"
V0153 = FIXTURES / "v0153"
V0150 = FIXTURES / "v0150"
V0142 = FIXTURES / "rollout-fixture.jsonl"


def test_v0142_legacy_event_kinds_unchanged() -> None:
    """Existing 0.142 fixture still decodes through the legacy path."""
    from agent_hotwash.sources._common import iter_jsonl
    from agent_hotwash.sources.codex_native import decode_codex_native

    events, sid, model = decode_codex_native(list(iter_jsonl(V0142)))
    kinds = [e.kind for e in events]
    assert EventKind.user_msg in kinds
    assert EventKind.assistant_msg in kinds
    assert EventKind.thinking in kinds
    assert EventKind.tool_call in kinds
    assert EventKind.compaction in kinds
    assert sid == "dddddddd-0000-0000-0000-000000000004"
    assert model == "gpt-5.5"


def test_user_multiturn_acceptance() -> None:
    tr = load_rollout(V0153 / "user_multiturn.jsonl")
    sess = tr.root
    assert len(sess.turns) == 2
    assert [t.status for t in sess.turns] == [TurnStatus.completed, TurnStatus.completed]
    n_calls = sum(len(t.model_calls) for t in sess.turns)
    assert n_calls == 83

    users = [e for e in sess.events if e.kind is EventKind.user_msg]
    assert sum(1 for e in users if e.role_hint is RoleHint.user) == 2
    assert sum(1 for e in users if e.role_hint is RoleHint.injected) == 2
    assert sum(1 for e in sess.events if e.kind is EventKind.assistant_msg) == 17

    cfg = load_config()
    a = analyze(tr, cfg)
    # one failed cat, rest of CommandExecution/FileChange succeed
    assert a.root.tool_error_count == 1
    cats = a.root.tools_by_category
    assert cats.get("read", 0) == 2  # search + failed read
    assert cats.get("write", 0) == 1
    assert a.root.user_turns == 2  # injected excluded

    # Sigma per-response usage == final thread_token_usage - baseline
    usages = {}
    for ev in sess.events:
        if ev.response_id and ev.usage and ev.raw_type == "token_usage_record":
            usages[ev.response_id] = ev.usage
    assert len(usages) == 83
    sum_in = sum((u.input or 0) + (u.cache_read or 0) for u in usages.values())
    sum_out = sum(u.output or 0 for u in usages.values())
    assert sum_in == 83 * 100
    assert sum_out == 83 * 10
    for u in usages.values():
        assert u.reasoning_output is not None and u.output is not None
        assert u.reasoning_output <= u.output
        assert u.input == 60  # 100 - 40 cached
        assert u.cache_read == 40


def test_child_spawn_replay_drops_prefix_and_foreign_usage() -> None:
    tr = load_rollout(V0153 / "tree" / "rollout-child-spawn-replay.jsonl")
    sess = tr.root
    # no turns/ops/usage from the replay prefix (parent session_meta + compacted + parent turn)
    assert sess.replay_prefix is not None
    users = [e for e in sess.events if e.kind is EventKind.user_msg]
    assert len(users) == 1
    assert users[0].text and "delegated" in users[0].text
    # foreign parent usage dropped
    rids = {e.response_id for e in sess.events if e.response_id}
    assert "resp-parent" not in rids
    assert "resp-child" in rids
    assert "usage_identity" not in sess.degraded


def test_multi_op_exec_shares_group_id() -> None:
    tr = load_rollout(V0153 / "multi_op_exec.jsonl")
    children = [e for e in tr.root.events if e.group_id == "call_wrap_1" and e.kind is not EventKind.meta]
    assert len(children) >= 4  # 2 exec pairs + filechange pair, minus maybe
    gids = {e.group_id for e in children}
    assert gids == {"call_wrap_1"}
    notes = " ".join(tr.provenance.notes)
    assert "malformed" in notes


def test_ops_all_types_mapped() -> None:
    tr = load_rollout(V0153 / "ops_all_types.jsonl")
    kinds = {e.raw_type for e in tr.root.events}
    for expected in (
        "UserMessage",
        "Reasoning",
        "AgentMessage",
        "CommandExecution",
        "FileChange",
        "McpToolCall",
        "SubAgentActivity",
        "Extension",
        "ContextCompaction",
        "CollabAgentToolCall",
        "ImageView",
    ):
        assert expected in kinds
    # A FileChange that only *adds* a file is a write, not an edit of unread content.
    file_ops = [e for e in tr.root.events if e.raw_type == "FileChange" and e.kind is EventKind.tool_call]
    assert file_ops and file_ops[0].artifacts
    assert file_ops[0].artifacts[0].op.value == "add"
    assert file_ops[0].op_kind == "file.write"
    assert file_ops[0].lines_added == 2
    mcp = [e for e in tr.root.events if e.op_kind and e.op_kind.startswith("mcp.")]
    assert mcp


def test_exec_output_shapes() -> None:
    tr = load_rollout(V0153 / "exec_outputs.jsonl")
    results = [e for e in tr.root.events if e.kind is EventKind.tool_result]
    by_id = {e.call_id: e for e in results}
    assert by_id["exec-ok"].ok is True and by_id["exec-ok"].exit_code == 0
    assert by_id["exec-fail"].ok is False
    assert by_id["exec-json_exit"].exit_code == 3
    assert by_id["exec-trunc_warn"].output_tokens_original == 9999
    assert by_id["exec-trunc_json"].output_tokens_original == 4321


def test_aborted_open_and_unterminated() -> None:
    tr = load_rollout(V0153 / "aborted_open_turns.jsonl")
    by_id = {t.turn_id: t for t in tr.root.turns}
    assert by_id["ab1"].status is TurnStatus.aborted
    assert by_id["op1"].status is TurnStatus.open
    assert by_id["z1"].status is TurnStatus.completed
    # unterminated call on the open turn
    open_calls = by_id["op1"].model_calls
    assert open_calls
    assert open_calls[-1].response_id is None


def test_v0150_agent_message_surface() -> None:
    tr = load_rollout(V0150 / "agent_message.jsonl")
    assert tr.root.harness_version and tr.root.harness_version.startswith("0.150")
    assert tr.root.thread_source is None
    texts = [e.text for e in tr.root.events if e.kind is EventKind.assistant_msg]
    assert any(t and "legacy agent surface" in t for t in texts)


def test_role_delegation_before_injected() -> None:
    tr = load_rollout(V0153 / "tree" / "rollout-child-created.jsonl")
    users = [e for e in tr.root.events if e.kind is EventKind.user_msg]
    assert users[0].role_hint is RoleHint.delegation
    assert users[0].text and "<codex_delegation>" in users[0].text


def test_orchestrator_tree_links() -> None:
    tree_dir = V0153 / "tree"
    traces = list(iter_traces(tree_dir))
    # parent + missing-parent? created/spawn/fork should attach when parent is present
    roots = {t.trace_id: t for t in traces}
    parent_id = "bbbbbbbb-0000-0000-0000-000000000002"
    assert parent_id in roots
    parent = roots[parent_id]
    child_ids = {s.session_id for s in parent.subagents}
    # spawn child and fork (forked_from_id) and maybe created (no parent field)
    assert "cccccccc-0000-0000-0000-000000000003" in child_ids
    assert "dddddddd-0000-0000-0000-000000000004" in child_ids
    kinds = {(lk.child_id, lk.kind.value) for lk in parent.links}
    assert ("cccccccc-0000-0000-0000-000000000003", "spawn") in kinds
    assert ("dddddddd-0000-0000-0000-000000000004", "fork") in kinds


def test_missing_parent_becomes_root_with_note(tmp_path: Path) -> None:
    src = V0153 / "tree" / "rollout-child-spawn-replay.jsonl"
    dest = tmp_path / "rollout-orphan.jsonl"
    dest.write_text(src.read_text(), encoding="utf-8")
    traces = list(build_codex_forest([dest]))
    assert len(traces) == 1
    assert any(n.startswith("parent not in input") for n in traces[0].provenance.notes)
    assert traces[0].provenance.thread_linkage == "partial"
    assert "thread_linkage" in traces[0].root.degraded


def test_command_classification_precedence() -> None:
    from agent_hotwash.events import ParsedCommand
    from agent_hotwash.sources.codex_items import _primary_op_kind

    mixed = [
        ParsedCommand(type="read", cmd="cat a", path="a"),
        ParsedCommand(type="search", cmd="rg x", path=None),
        ParsedCommand(type="unknown", cmd="pytest -q", path=None),
    ]
    # unknown → cmd.exec (pytest), and exec outranks search/read
    assert _primary_op_kind(mixed, "cat a && rg x && pytest -q") == "cmd.exec"
    only_unknown = [ParsedCommand(type="unknown", cmd="rg -n foo src")]
    assert _primary_op_kind(only_unknown, "rg -n foo src") == "cmd.search"
