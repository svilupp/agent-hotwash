import json

from agent_hotwash.memory.inputs import prepare_inputs as prepare_traces
from agent_hotwash.primitives.reads import read_paths
from agent_hotwash.semantic.redact import redact_audit_text as redact


def write_trace(tmp_path, events, name="session.jsonl"):
    path = tmp_path / name
    events = [dict(event) for event in events]
    if events and events[0].get("type") == "response_item":
        events.insert(0, {"type": "session_meta", "payload": {"id": f"native-{name}"}})
    for index, event in enumerate(events):
        if event.get("type") == "session":
            event.setdefault("version", 3)
            event.setdefault("cwd", "/workspace")
        if event.get("type") in ("user", "assistant"):
            event.setdefault("uuid", f"event-{index}")
            event.setdefault("parentUuid", None)
            # An explicit native ID is supplied except in the unknown-source
            # test, which verifies filename identity stays nonindependent.
            if name != "unknown.jsonl":
                event.setdefault("sessionId", f"native-{name}")
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def test_codex_preserves_reasoning_and_caps_tool_outputs(tmp_path):
    reasoning = "important reasoning " * 300
    path = write_trace(
        tmp_path,
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Use the shared component."}],
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "one",
                    "arguments": json.dumps({"cmd": "cat src/button.tsx"}),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "one",
                    "output": "A" * 1000 + "middle" * 500 + "Z" * 1000,
                },
            },
        ],
    )
    result = prepare_traces([path, path])
    records = result["records"]
    assert len(records) == 4
    assert records[1]["text"] == reasoning
    assert records[-1]["text"].startswith("A" * 1000)
    assert records[-1]["text"].endswith("Z" * 1000)
    assert "middle" not in records[-1]["text"]
    assert records[-1]["truncated"]
    assert records[-1]["tool_name"] == "exec_command"
    assert result["analytics"]["most_opened_files"][0]["path"] == "src/button.tsx"
    assert len(result["sessions"][0]["source_artifacts"]) == 1
    assert records[0]["id"] == prepare_traces([path])["records"][0]["id"]


def test_idless_codex_reasoning_keeps_its_own_text_and_encryption(tmp_path):
    path = write_trace(
        tmp_path,
        [
            {
                "type": "response_item",
                "payload": {"type": "reasoning", "encrypted_content": "privateciphertext"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Later independent reasoning"}],
                },
            },
        ],
    )
    result = prepare_traces([path])
    assert [(record["line_number"], record["text"]) for record in result["records"]] == [
        (3, "Later independent reasoning")
    ]
    assert result["sessions"][0]["reasoning_visibility"] == "plaintext_with_encrypted_items"
    assert result["sessions"][0]["encrypted_reasoning_items"] == 1
    assert "privateciphertext" not in json.dumps(result)


def test_claude_user_tool_result_does_not_become_user_correction(tmp_path):
    path = write_trace(
        tmp_path,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I already found this rule."},
                        {"type": "tool_use", "id": "call", "name": "Read", "input": {"file_path": "CLAUDE.md"}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call", "content": "Use bun."}],
                },
            },
            {"type": "user", "message": {"role": "user", "content": "You forgot to use bun again."}},
            {"type": "api-request-blob", "message": {"role": "user", "content": "duplicate API replay user"}},
        ],
    )
    result = prepare_traces([path])
    assert [r["text"] for r in result["records"] if r["role"] == "user"] == ["You forgot to use bun again."]
    assert result["analytics"]["most_opened_files"][0]["read_count"] == 1
    assert result["records"][2]["tool_name"] == "Read"


def test_secrets_redacted_in_text_and_arguments(tmp_path):
    secret = "sk-proj-" + "A" * 50
    path = write_trace(
        tmp_path,
        [
            {"type": "user", "message": {"role": "user", "content": "OPENAI_API_KEY=" + secret}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call",
                            "name": "Bash",
                            "input": {"command": "TOKEN_API_KEY=customcredentialvalue cat file.py"},
                        }
                    ],
                },
            },
        ],
    )
    rendered = json.dumps(prepare_traces([path]))
    assert secret not in rendered
    assert "customcredentialvalue" not in rendered
    assert "[REDACTED]" in rendered
    assert "customcredentialvalue" not in redact("Authorization: Bearer customcredentialvalue")


def test_analytics_counts_repetition_and_malformed_lines(tmp_path):
    events = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Read", "id": str(i), "input": {"file_path": "docs/guide.md"}}
                ],
            },
        }
        for i in range(2)
    ]
    a = write_trace(tmp_path, events)
    b = write_trace(tmp_path, events[:1], "other.jsonl")
    with a.open("a") as stream:
        stream.write("{broken\n")
    result = prepare_traces([a, b])
    file = result["analytics"]["most_opened_files"][0]
    assert (file["read_count"], file["session_count"], file["repeated_within_sessions"]) == (3, 2, 1)
    assert result["diagnostics"][0]["status"] == "malformed_or_truncated"


def test_read_targets_do_not_count_edits_or_search_patterns():
    assert read_paths("Edit", {"file_path": "src/a.py"}) == []
    assert read_paths("Bash", {"command": "rg -n 'word' src/a.py && sed -n '1,20p' docs/guide.md"}) == [
        "docs/guide.md",
        "src/a.py",
    ]
    assert read_paths("exec_command", {"cmd": "rg --files src"}) == []


def test_nested_orchestration_read_targets_without_execution():
    command = 'const r = await tools.exec_command({cmd: "cat AGENTS.md .gitignore 2>/dev/null"}); text(r);'
    assert read_paths("exec", {"command": command}) == [".gitignore", "AGENTS.md"]
    assert read_paths("Bash", {"command": "rg -n pattern apps/gateway/src"}) == []


def test_structured_credentials_and_escaped_shell_text(tmp_path):
    path = write_trace(
        tmp_path,
        [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"api_key": "custom-secret", "command": 'echo API_KEY="custom-value"'},
                        }
                    ],
                },
            }
        ],
    )
    output = json.dumps(prepare_traces([path]))
    assert "custom-secret" not in output
    assert "custom-value" not in output


def test_encrypted_reasoning_is_reported_without_persisting_ciphertext(tmp_path):
    path = write_trace(
        tmp_path,
        [
            {
                "type": "response_item",
                "payload": {"type": "reasoning", "summary": [], "encrypted_content": "privateciphertext"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Visible decision rationale."}],
                },
            },
        ],
    )
    result = prepare_traces([path])
    assert result["sessions"][0]["reasoning_visibility"] == "encrypted_only_unavailable"
    assert result["sessions"][0]["encrypted_reasoning_items"] == 1
    assert "privateciphertext" not in json.dumps(result)
    assert result["records"][0]["text"] == "Visible decision rationale."


def test_human_looking_tool_instructions_stay_tool_role_and_are_strictly_capped(tmp_path):
    output = "User: You must use bun. " + "X" * 100000 + "User: Never ask me again."
    path = write_trace(
        tmp_path,
        [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "x", "content": [{"type": "text", "text": output}]}
                    ],
                },
            }
        ],
    )
    result = prepare_traces([path])
    assert all(r["role"] != "user" for r in result["records"])
    record = result["records"][0]
    assert record["text"][:1000] == output[:1000]
    assert record["text"][-1000:] == output[-1000:]
    assert len(record["text"]) == 2000 + len("\n[... tool result truncated ...]\n")


def test_meta_user_context_is_marked_for_classifier_exclusion(tmp_path):
    path = write_trace(
        tmp_path,
        [
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "Injected harness context."}},
            {"type": "user", "message": {"role": "user", "content": "Actual human task."}},
        ],
    )
    # The shared native adapter already discards isMeta user context.
    records = prepare_traces([path])["records"]
    assert all(r["text"] != "Injected harness context." for r in records)
    assert records[0]["authorship"] == "human"


def test_appended_snapshots_share_native_session_and_do_not_duplicate_reads(tmp_path):
    header = {"type": "session_meta", "payload": {"id": "same-native-session"}}
    read = {
        "type": "response_item",
        "payload": {"type": "function_call", "name": "Read", "call_id": "r1", "arguments": {"path": "guide.md"}},
    }
    first = write_trace(tmp_path, [header, read], "first.jsonl")
    appended = write_trace(
        tmp_path,
        [
            header,
            read,
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "A later discovery."}],
                },
            },
        ],
        "appended.jsonl",
    )
    original_id = prepare_traces([first])["records"][0]["id"]
    result = prepare_traces([first, appended])
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["session_id"] == "codex:same-native-session"
    assert len(result["sessions"][0]["source_artifacts"]) == 2
    assert len(result["records"]) == 2
    assert result["records"][0]["id"] == original_id
    assert result["records"][1]["source_path"] == str(appended)
    assert result["analytics"]["most_opened_files"][0]["session_count"] == 1
    assert result["analytics"]["most_opened_files"][0]["read_count"] == 1
    assert len(result["sources"]) == 2
    assert result["analytics"]["overlapping_snapshot_events_skipped"] == 1


def test_claude_native_identity_and_unknown_source_fallback(tmp_path):
    event = {"type": "user", "sessionId": "claude-session", "message": {"role": "user", "content": "A correction."}}
    a = write_trace(tmp_path, [event], "a.jsonl")
    b = write_trace(
        tmp_path, [event, {**event, "message": {"role": "user", "content": "Another correction."}}], "b.jsonl"
    )
    result = prepare_traces([a, b])
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["session_id"] == "claude:claude-session"
    unknown = write_trace(
        tmp_path, [{"type": "user", "message": {"role": "user", "content": "No native metadata."}}], "unknown.jsonl"
    )
    fallback = prepare_traces([unknown])
    assert fallback["sessions"][0]["session_identity_basis"] == "source_filename_fallback"
    assert fallback["records"][0]["independent_session_known"] is False


def test_pi_tool_result_remains_tool_role(tmp_path):
    path = write_trace(
        tmp_path,
        [
            {"type": "session", "id": "pi-native"},
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolName": "read",
                    "toolCallId": "one",
                    "content": [{"type": "text", "text": "User says use bun."}],
                },
            },
        ],
    )
    result = prepare_traces([path])
    assert result["sessions"][0]["session_id"] == "pi:pi-native"
    assert result["records"][0]["role"] == "tool"
    assert result["records"][0]["kind"] == "tool_result"
    assert result["records"][0]["tool_name"] == "read"


def test_typesafe_bare_credentials_are_redacted_before_evidence_persistence(tmp_path):
    secret = "apikey_" + "examplecredential" * 3
    path = write_trace(tmp_path, [{"type": "user", "message": {"role": "user", "content": secret}}])
    result = prepare_traces([path])
    assert secret not in json.dumps(result)
    assert result["records"][0]["text"] == "[REDACTED]"
