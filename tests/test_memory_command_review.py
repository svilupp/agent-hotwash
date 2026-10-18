"""Independent regressions for misleading literal command attribution."""

import pytest

from agent_hotwash.memory.command_hooks import collect_command_hooks


def collect(command):
    return collect_command_hooks(
        [
            {
                "id": "review-evidence",
                "session_id": "review-session",
                "kind": "tool_call",
                "tool_name": "Bash",
                "arguments": {"command": command},
            }
        ]
    )


def summary(result, hook):
    return next(row for row in result["summaries"] if row["name"] == hook)


def test_shell_comments_are_data_neither_calls_nor_playbooks():
    result = collect("agent-playbooks read foo # explanation; mem index")
    assert summary(result, "playbooks")["values"] == [{"value": "foo", "count": 1, "session_count": 1}]
    assert summary(result, "mem_index")["call_count"] == 0
    assert len(result["calls"]) == 1


def test_quoted_comment_character_remains_a_search_theme():
    result = collect('mem search "C# patterns" # ignored; mem index')
    assert summary(result, "mem_search")["values"] == [{"value": "C# patterns", "count": 1, "session_count": 1}]
    assert summary(result, "mem_index")["call_count"] == 0


def test_background_separator_does_not_invent_playbook_names():
    result = collect("agent-playbooks read foo & mem index")
    assert summary(result, "playbooks")["values"] == [{"value": "foo", "count": 1, "session_count": 1}]
    assert summary(result, "mem_index")["call_count"] == 1


@pytest.mark.parametrize("command", ['agent-playbooks read "$BOOK"', "mem search $(echo theme)"])
def test_dynamic_shell_arguments_are_not_claimed_as_literal_values(command):
    result = collect(command)
    assert result["diagnostics"]
    assert all(row["parse_status"] == "partial" for row in result["calls"])
    assert all(not row["values"] for row in result["summaries"])


@pytest.mark.parametrize("command", ['"mem" index', "sudo -u jan mem index"])
def test_literal_executable_normalization(command):
    assert summary(collect(command), "mem_index")["call_count"] == 1


def test_one_command_can_match_independent_hooks():
    from agent_hotwash.config import CommandHookConfig

    records = [
        {
            "id": "e",
            "session_id": "s",
            "kind": "tool_call",
            "tool_name": "Bash",
            "arguments": {"command": "mem search deploy"},
        }
    ]
    hooks = [
        CommandHookConfig(name="count", patterns=("mem",)),
        CommandHookConfig(name="theme", patterns=("mem",), extract_patterns=(r"search (\w+)",)),
    ]
    result = collect_command_hooks(records, hooks)
    assert len(result["calls"]) == 1
    assert len(result["matches"]) == 2
    assert summary(result, "count")["call_count"] == 1
    assert summary(result, "theme")["values"][0]["value"] == "deploy"


def test_export_redacts_sensitive_command_arguments_before_all_datasets(tmp_path):
    import json

    from agent_hotwash.config import Config, LexiconConfig
    from agent_hotwash.memory.pipeline import load_saved, run_audit

    secret = "review-private-token-7162"
    source = tmp_path / "rollout-review.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "redaction-review"}},
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "review",
                "arguments": json.dumps({"cmd": f'mem search "{secret}"'}),
            },
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "out"
    run_audit([source], output, config=Config(lexicons=LexiconConfig(secret=[secret])))
    load_saved(output)
    dataset = json.loads((output / "command-hooks.json").read_text())
    assert dataset["calls"]
    assert summary(dataset, "mem_search")["values"][0]["value"] == "[REDACTED]"
    for path in output.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(), path.name


def test_javascript_regex_literals_do_not_invent_nested_shell_calls():
    result = collect_command_hooks(
        [
            {
                "id": "regex-example",
                "session_id": "review-session",
                "kind": "tool_call",
                "tool_name": "functions.exec",
                "arguments": {"code": 'const r = /tools.exec_command({cmd:"mem index"})/;'},
            }
        ]
    )
    assert result["calls"] == []


def test_heredoc_body_cannot_become_playbook_names():
    result = collect("agent-playbooks read alpha; python3 - <<'PY'\nfor k in ['fake']:\n print(k)\nPY")
    assert result["diagnostics"]
    assert {row["value"] for row in summary(result, "playbooks")["values"]} <= {"alpha"}
