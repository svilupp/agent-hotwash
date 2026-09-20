"""Error-classifier table tests.

Seeded with the REAL per-agent error strings verified in
``docs/research/format_findings.md`` (checked against on-disk samples). The
extended categories DESIGN.md adds beyond the reference vocabulary
(rate_limit/mcp_transport/sandbox_egress/network) use representative strings.

Note the exit code is often embedded in the text (``Exit code N`` for claude,
``Command exited with code N`` for pi, ``Process exited with code N`` for native
codex); ``classify_error`` recovers it when the caller passes ``exit_code=None``.
"""

from __future__ import annotations

import pytest

from agent_hotwash.primitives.errors import classify_error

# (tool, exit_code, msg, command) -> expected category
CASES: list[tuple[str, int | None, str, str, str]] = [
    # --- claude / native claude bash: "Exit code N\n..." (exit code in text) ---
    (
        "Bash",
        None,
        "Exit code 127\n/bin/bash: line 1: python3: command not found",
        "python3 x.py",
        "command_not_found",
    ),
    ("Bash", None, "Exit code 143\nCommand timed out after 30s", "sleep 90", "timeout"),
    (
        "Bash",
        None,
        "Exit code 2\nsrc/app.module.ts(3,30): error TS2307: Cannot find module '@window-shop/shared'",
        "tsc --noEmit",
        "build_test_fail",
    ),
    # --- claude framework errors wrapped in <tool_use_error> ---
    (
        "Edit",
        None,
        "<tool_use_error>File has not been read yet. Read it first before writing to it.</tool_use_error>",
        "",
        "edit_mismatch",
    ),
    (
        "Write",
        None,
        "<tool_use_error>InputValidationError: The required parameter `description` is missing</tool_use_error>",
        "",
        "agent_syntax_error",
    ),
    # a read-first violation fires on Write too, not just the Edit family.
    (
        "Write",
        None,
        "<tool_use_error>File has not been read yet. Read it first before writing to it.</tool_use_error>",
        "",
        "edit_mismatch",
    ),
    (
        "Bash",
        None,
        "<tool_use_error>Blocked: sleep 90 followed by: echo done</tool_use_error>",
        "sleep 90",
        "harness_blocked",
    ),
    # --- code-bench codex: text in aggregated_output, exit_code is a field ---
    (
        "command_execution",
        2,
        "rg: node_modules/ai-fallback: No such file or directory (os error 2)",
        "rg foo node_modules/ai-fallback",
        "file_not_found",
    ),
    (
        "command_execution",
        1,
        "jest\nFAIL src/modules/agents/services/agent.service.spec.ts",
        "jest",
        "build_test_fail",
    ),
    # --- pi: text in result.content[].text, "Command exited with code N" ---
    (
        "Bash",
        None,
        "cat: node_modules/ai-fallback/package.json: No such file or directory\nCommand exited with code 2",
        "cat node_modules/ai-fallback/package.json",
        "file_not_found",
    ),
    ("read", None, "Offset 72 is beyond end of file (71 lines total)", "", "other"),
    # --- native codex: "Process exited with code N", no field ---
    (
        "command_execution",
        None,
        "Process exited with code 1\nError: No section matching found",
        "./run.sh",
        "other",
    ),
    # --- DESIGN.md extended categories (representative strings) ---
    ("mcp__srv", None, "MCP error -32000: Client Closed", "", "mcp_transport"),
    ("Bash", None, "Error 429 Too Many Requests", "curl x", "rate_limit"),
    ("Bash", None, "overloaded_error: server overloaded", "curl x", "rate_limit"),
    ("Bash", 1, "network egress blocked by sandbox", "curl https://x", "sandbox_egress"),
    ("Bash", 1, "path is outside the sandbox root", "cat /etc/x", "sandbox_egress"),
    ("Bash", 1, "Permission denied", "cat /root/x", "permission"),
    ("Bash", 1, "getaddrinfo ENOTFOUND api.example.com", "curl api", "network"),
    # --- no-match probe: grep-family exit 1 with tiny output ---
    ("Bash", 1, "", "grep needle haystack.txt", "no_match_probe"),
    # diff / cmp / test / git diff --check exit 1 = "differs"/"false", not an error
    ("cmd.read", 1, "--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y", "diff -u a b", "no_match_probe"),
    ("cmd.read", 1, "a.py:3: trailing whitespace.", "git diff --no-index --check a b", "no_match_probe"),
    ("Bash", 1, "", "test -f missing.txt", "no_match_probe"),
    # --- fallthrough ---
    ("Bash", 1, "something weird happened", "./run.sh", "other"),
    # --- shell stdout is never an argument-schema error ---
    ("cmd.exec", 1, "src/x.py:3: docstring mentions revalidation", "ruff check .", "build_test_fail"),
    ("cmd.exec", 1, "src/x.py:3: docstring mentions revalidation", "./lint.sh", "other"),
    ("cmd.exec", 1, "error: No argument provided for required parameter `x`", "ty check src", "build_test_fail"),
    ("cmd.read", 1, "--- /private/tmp/wikow-astra-fixes-validation.m6boml/a.py", "diff -u a b", "no_match_probe"),
    ("cmd.exec", 1, "InputValidationError: field x", "bunx vitest run", "build_test_fail"),
    # --- harness rejections (no shell command) still classify ---
    ("mcp.srv.tool", None, "-32602 invalid params", "", "agent_syntax_error"),
    ("agent.spawn", None, "validation error: missing field `prompt`", "", "agent_syntax_error"),
    # --- plain test failure text mentioning network/root is not a sandbox denial ---
    ("cmd.exec", 1, "FAILED tests/test_net.py::test_root - assert network_root == 1", "pytest", "build_test_fail"),
    ("cmd.exec", 1, "curl: (7) Failed to connect; network is disabled in this sandbox", "curl x", "sandbox_egress"),
    ("cmd.exec", 1, "mkdir: /etc/x: Operation not permitted", "mkdir /etc/x", "sandbox_egress"),
]


@pytest.mark.parametrize(("tool", "code", "msg", "cmd", "expected"), CASES)
def test_error_categories(tool: str, code: int | None, msg: str, cmd: str, expected: str) -> None:
    category, _severity, _conf = classify_error(tool, code, msg, cmd)
    assert category == expected


def test_exit_code_recovered_from_text() -> None:
    # claude style, no exit_code field passed
    assert classify_error("Bash", None, "Exit code 127\nfoo: command not found", "foo")[0] == "command_not_found"
    # pi style
    assert classify_error("Bash", None, "boom\nCommand exited with code 143", "sleep 9")[0] == "timeout"
    # native codex style
    cat, _sev, _c = classify_error("Bash", None, "Process exited with code 127", "frob")
    assert cat == "command_not_found"


def test_severity_map() -> None:
    assert classify_error("Bash", 127, "x: command not found", "x")[1] == "agent_error"
    assert classify_error("Bash", 1, "1 failed", "pytest")[1] == "workflow"
    assert classify_error("Bash", 1, "", "grep needle f.txt")[1] == "benign"


def test_compound_tail_downgrade() -> None:
    # a failing grep tail on a compound line already yielded its data -> benign/low
    _category, severity, conf = classify_error("Bash", 1, "no matches", "cat f.txt && grep needle f.txt")
    assert severity == "benign"
    assert conf == "low"


def test_compound_downgrade_not_applied_to_hard_failures() -> None:
    # command-not-found (127) is never downgraded even on a compound line
    _cat, severity, conf = classify_error("Bash", 127, "not found", "cd x && frob")
    assert severity == "agent_error"
    assert conf == "high"
