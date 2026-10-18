"""Observable command collection across normalized harness inputs."""

import pytest
from pydantic import ValidationError

from agent_hotwash.config import CommandHookConfig, MemoryConfig, load_config
from agent_hotwash.memory.command_hooks import collect_command_hooks


def record(command, tool="Bash", key="command", identity="e1"):
    return {
        "id": identity,
        "session_id": "session1",
        "kind": "tool_call",
        "tool_name": tool,
        "arguments": {key: command},
        "source_path": "/tmp/source.jsonl",
        "line_number": 12,
        "source_refs": [{"line_number": 12}],
    }


def summaries(result):
    return {row["name"]: row for row in result["summaries"]}


@pytest.mark.parametrize(
    ("tool", "key"),
    [
        ("Bash", "command"),
        ("bash", "command"),
        ("exec_command", "cmd"),
        ("functions.exec_command", "cmd"),
        ("mcp__shell__exec_command", "cmd"),
        ("shell", "command"),
    ],
)
def test_shell_harness_shapes(tool, key):
    result = collect_command_hooks([record('mem index; mem search "deployment errors"', tool, key)])
    found = summaries(result)
    assert len(result["calls"]) == 2
    assert found["mem_index"]["call_count"] == 1
    assert found["mem_search"]["values"] == [{"value": "deployment errors", "count": 1, "session_count": 1}]
    assert result["calls"][0]["source_refs"] == [{"line_number": 12}]


def test_playbook_names_options_and_commas():
    result = collect_command_hooks(
        [record('env X=1 /bin/agent-playbooks --harness codex read --exact "poteto-mode, supervisor-mode" how --json')]
    )
    assert summaries(result)["playbooks"]["values"] == [
        {"value": name, "count": 1, "session_count": 1} for name in ("how", "poteto-mode", "supervisor-mode")
    ]


@pytest.mark.parametrize("alias", ["read", "load", "show"])
def test_playbook_aliases(alias):
    assert summaries(collect_command_hooks([record(f"agent-playbooks {alias} foo")]))["playbooks"]["call_count"] == 1


def test_quoted_queries_shell_wrappers_and_full_dataset():
    result = collect_command_hooks([record("bash -lc 'mem search \"a,b; c && d\"; ls'")])
    assert [r["command"] for r in result["calls"]] == ['mem search "a,b; c && d"', "ls"]
    assert summaries(result)["mem_search"]["values"][0]["value"] == "a,b; c && d"


def test_echo_and_nonexecuted_text_do_not_match_presets():
    result = collect_command_hooks([record('echo "mem index; agent-playbooks read foo"; cat README.md')])
    assert len(result["calls"]) == 2
    assert all(row["call_count"] == 0 for row in result["summaries"])


def test_nested_parallel_literal_commands_ignore_strings_comments():
    code = """
const example = "tools.exec_command({cmd:'mem index'})";
// tools.exec_command({cmd:"mem index"})
await Promise.all([tools.exec_command({cmd: "mem index"}),
 tools.exec_command({"cmd": 'agent-playbooks read foo,bar'})]);
await tools.exec_command({cmd: `mem search "theme"`});
"""
    result = collect_command_hooks([record(code, "functions.exec", "code")])
    assert len(result["calls"]) == 3
    assert summaries(result)["mem_index"]["call_count"] == 1
    assert [v["value"] for v in summaries(result)["playbooks"]["values"]] == ["bar", "foo"]


def test_dynamic_nested_commands_are_diagnosed_without_inventing_calls():
    result = collect_command_hooks(
        [
            record(
                'tools.exec_command({cmd: "mem " + operation}); tools.exec_command({cmd: `mem ${operation}`})',
                "functions.exec",
                "code",
            )
        ]
    )
    assert result["calls"] == []
    assert result["diagnostics"] == [{"evidence_id": "e1", "reason": "nested_exec_command_is_dynamic"}]


def test_same_text_distinct_events_preserved_duplicate_identity_not_recounted():
    first, second = record("mem index"), record("mem index", identity="e2")
    result = collect_command_hooks([first, second, first])
    assert len(result["calls"]) == 2
    assert len({r["id"] for r in result["calls"]}) == 2
    assert summaries(result)["mem_index"]["call_count"] == 2


def test_custom_regex_patterns_and_safe_value_splitting():
    hook = CommandHookConfig(
        name="docs",
        patterns=(r"^custom read", r"^custom show"),
        extract_patterns=(r"read (?P<names>[a-z,]+)",),
        split_values=",",
    )
    result = collect_command_hooks([record("custom read foo,bar"), record("custom show baz", identity="e2")], [hook])
    assert summaries(result)["docs"]["call_count"] == 2
    assert result["matches"][0]["captures"] == [{"names": "foo,bar"}]
    assert [v["value"] for v in summaries(result)["docs"]["values"]] == ["bar", "foo"]


def test_regex_commas_are_not_split_and_positional_captures_supported():
    hook = CommandHookConfig(name="repeat", patterns=(r"x{2,3}",), extract_patterns=(r"(x{2,3})",))
    result = collect_command_hooks([record("xx xxx")], [hook])
    assert result["matches"][0]["values"] == ["xx", "xxx"]


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "broken", "patterns": ["["]},
        {"name": "broken", "patterns": ["x"], "extract_patterns": ["("]},
        {"name": "empty", "patterns": []},
    ],
)
def test_invalid_regex_config_fails_at_boundary(payload):
    with pytest.raises(ValidationError):
        CommandHookConfig.model_validate(payload)


def test_duplicate_hook_names_rejected():
    hook = {"name": "x", "patterns": ["x"]}
    with pytest.raises(ValidationError, match="unique"):
        MemoryConfig.model_validate({"command_hooks": [hook, hook]})


def test_toml_user_hooks_replace_defaults(tmp_path):
    path = tmp_path / "hooks.toml"
    path.write_text(
        "[[memory.command_hooks]]\nname=\"custom\"\npatterns=['^custom\\s']\nextract_patterns=['(?P<theme>foo{1,3})']\n"
    )
    assert [h.name for h in load_config(path).memory.command_hooks] == ["custom"]


def test_heredoc_is_partial_and_body_not_separately_counted():
    result = collect_command_hooks([record("python3 - <<'PY'\nprint('mem index')\nPY")])
    assert len(result["calls"]) == 1
    assert result["calls"][0]["parse_status"] == "partial"
    assert all(row["call_count"] == 0 for row in result["summaries"])


def test_unrelated_tool_command_field_does_not_become_executed_shell():
    assert collect_command_hooks([record("mem index", "write")])["calls"] == []


def test_adapted_cmd_exec_is_shell_not_javascript():
    result = collect_command_hooks([record("mem index", "cmd.exec", "cmd")])
    assert summaries(result)["mem_index"]["call_count"] == 1
    assert result["diagnostics"] == []


def test_redirections_excluded_from_playbook_names_but_quoted_search_preserved():
    result = collect_command_hooks(
        [
            record("agent-playbooks read foo 2>&1 > /tmp/out"),
            record('mem search "2>&1"', identity="e2"),
        ]
    )
    assert summaries(result)["playbooks"]["values"][0]["value"] == "foo"
    assert len(summaries(result)["playbooks"]["values"]) == 1
    assert summaries(result)["mem_search"]["values"][0]["value"] == "2>&1"
    assert result["calls"][0]["shell_command"] == "agent-playbooks read foo 2>&1 > /tmp/out"


def test_javascript_regex_literal_is_data_and_division_keeps_real_calls():
    code = r"""const r = /tools.exec_command({cmd:"mem index"})/;
const escaped = /a[\/]tools.exec_command({cmd:"mem index"})/g;
const rate = 12 / 3;
await tools.exec_command({cmd:"mem search actual"});"""
    result = collect_command_hooks([record(code, "functions.exec", "code")])
    assert len(result["calls"]) == 1
    assert summaries(result)["mem_index"]["call_count"] == 0
    assert summaries(result)["mem_search"]["values"][0]["value"] == "actual"


def test_heredoc_prefix_keeps_playbook_call_and_excludes_body_names():
    command = "agent-playbooks read supervisor-mode-cloop --outline; python3 - <<'PY'\nfor k in ['x']:\n print(k)\nPY"
    result = collect_command_hooks([record(command)])
    assert len(result["calls"]) == 2
    assert all(row["shell_command"] == command for row in result["calls"])
    assert summaries(result)["playbooks"]["values"] == [
        {"value": "supervisor-mode-cloop", "count": 1, "session_count": 1}
    ]
    assert result["diagnostics"] == [{"evidence_id": "e1", "reason": "heredoc_body_not_parsed"}]


def test_quoted_heredoc_marker_is_ordinary_search_query():
    result = collect_command_hooks([record('mem search "<< example"')])
    assert summaries(result)["mem_search"]["values"][0]["value"] == "<< example"
    assert result["diagnostics"] == []
