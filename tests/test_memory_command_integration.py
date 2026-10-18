"""Command datasets remain offline, attributable, and replayable across sources."""

import csv
import hashlib
import io
import json
import shutil
import stat
from pathlib import Path

import pytest

from agent_hotwash.config import Config
from agent_hotwash.memory.pipeline import load_saved, run_audit

FIXTURES = Path(__file__).parent / "fixtures"
COMMAND = 'agent-playbooks read poteto-mode,supervisor-mode; mem index; mem search "trace parsing"'


def rewrite_commands(value):
    if isinstance(value, list):
        return [rewrite_commands(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = {key: rewrite_commands(item) for key, item in value.items()}
    if value.get("type") == "function_call":
        value.update(name="exec_command", arguments=json.dumps({"cmd": COMMAND}))
    elif value.get("type") == "tool_use":
        value.update(name="Bash", input={"command": COMMAND})
    elif value.get("type") == "toolCall":
        value.update(name="bash", arguments={"command": COMMAND})
    elif value.get("type") == "command_execution":
        value["command"] = COMMAND
    elif value.get("type") == "tool_execution_start":
        value.update(toolName="Bash", args={"command": COMMAND})
    return value


@pytest.mark.parametrize(
    "fixture",
    [
        "claude_native/proj/sess-fixture.jsonl",
        "codex_native/rollout-fixture.jsonl",
        "pi_native/session-fixture.jsonl",
        "codebench/claude_run",
        "codebench/codex_run",
        "codebench/pi_run_zerousage",
    ],
)
def test_offline_exports_across_harnesses(tmp_path, fixture):
    original = FIXTURES / fixture
    source = tmp_path / original.name
    if original.is_dir():
        shutil.copytree(original, source)
        stream = source / "stdout.jsonl"
    else:
        shutil.copyfile(original, source)
        stream = source
    rewritten = []
    for line in stream.read_text().splitlines():
        try:
            rewritten.append(json.dumps(rewrite_commands(json.loads(line))))
        except json.JSONDecodeError:
            rewritten.append(line)
    stream.write_text("\n".join(rewritten) + "\n")
    output = tmp_path / "out"
    summary = run_audit([source], output, config=Config())
    assert summary["status"] == "preprocessed"
    evidence, result = load_saved(output)
    dataset = json.loads((output / "command-hooks.json").read_text())
    assert dataset["calls"]
    summaries = {item["name"]: item for item in dataset["summaries"]}
    assert summaries["mem_index"]["call_count"] > 0
    assert summaries["mem_search"]["values"][0]["value"] == "trace parsing"
    assert {row["value"] for row in summaries["playbooks"]["values"]} == {"poteto-mode", "supervisor-mode"}
    records = {row["id"]: row for row in evidence["records"]}
    for call in dataset["calls"]:
        assert records[call["evidence_id"]]["source_refs"] == call["source_refs"]
        assert call["source_path"] and call["line_number"]
    assert result["calls"] == []
    assert result["provider_usage"]["openai"]["requests"] == 0
    assert "command_hooks" not in evidence
    for report in ("report.md", "report.html"):
        rendered = (output / report).read_text()
        assert "trace parsing" in rendered and "poteto" in rendered and "Command hooks" in rendered
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["dataset_artifacts"].items():
        data = (output / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    for name in ("calls", "matches"):
        exported = [json.loads(line) for line in (output / f"command-{name}.jsonl").read_text().splitlines()]
        assert exported == dataset[name]
        csv_rows = list(csv.DictReader(io.StringIO((output / f"command-{name}.csv").read_text())))
        assert len(csv_rows) == len(exported)
        if name == "calls":
            assert [row["shell_command"] for row in csv_rows] == [row["shell_command"] for row in exported]
    (output / "command-calls.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        load_saved(output)


def test_custom_hooks_reconfigure_exports_and_legacy_replay(tmp_path):
    from agent_hotwash.config import CommandHookConfig, MemoryConfig
    from agent_hotwash.memory.extraction import digest

    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"type": "session_meta", "payload": {"id": "custom"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "exec_command",
                        "call_id": "one",
                        "arguments": json.dumps({"cmd": "my-reader read alpha,beta"}),
                    },
                },
            ]
        )
        + "\n"
    )
    output = tmp_path / "out"
    config = Config(
        memory=MemoryConfig(
            command_hooks=(
                CommandHookConfig(
                    name="custom",
                    patterns=(r"my-reader read",),
                    extract_patterns=(r"read (?P<files>\S+)",),
                    split_values=",",
                ),
            )
        )
    )
    run_audit([source], output, config=config)
    _, result = load_saved(output)
    assert {row["value"] for row in result["command_hooks"]["summaries"][0]["values"]} == {"alpha", "beta"}
    manifest = json.loads((output / "manifest.json").read_text())
    (output / "command-calls.jsonl").unlink()
    with pytest.raises(ValueError, match="dataset is unavailable"):
        load_saved(output)
    result.pop("command_hooks")
    (output / "memory.json").write_text(json.dumps(result))
    manifest["artifacts"]["memory.json"] = digest(result)
    manifest.pop("dataset_artifacts")
    (output / "manifest.json").write_text(json.dumps(manifest))
    assert load_saved(output)[1] == result
