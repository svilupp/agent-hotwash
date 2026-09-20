"""The single error-classifier table.

``classify_error`` maps one raw tool failure to
``(category, severity, confidence)``. The string vocabulary is ported from the
code-bench reference ``_classify_error_category`` - it is the ground truth for
how each agent (claude/codex/pi) serializes errors - and extended per DESIGN.md
with ``rate_limit``, ``mcp_transport``, ``sandbox_egress`` and ``network``.
The real per-format error-string shapes are documented in
``docs/research/format_findings.md`` (verified against on-disk samples).

Because each agent embeds the exit code differently and some formats carry no
exit-code field at all (native codex, pi bash), the classifier also recovers the
exit code from the message text (``Exit code N`` / ``exited with code N`` /
``Process exited with code N``) when the caller does not pass one.

Category vocabulary (fixed; feeds taxonomies #13-17):
``harness_blocked, agent_syntax_error, edit_mismatch, file_not_found,
command_not_found, timeout, cancelled, permission, no_match_probe, rate_limit,
mcp_transport, sandbox_egress, network, build_test_fail, other``.

Severity ∈ ``benign | workflow | agent_error``.
"""

from __future__ import annotations

import re

from agent_hotwash.primitives.commands import (
    classify_command,
    exit1_is_signal_free,
    failing_tail_head,
    is_compound,
)

# Default severity per category, before the compound-command downgrade.
_CATEGORY_SEVERITY: dict[str, str] = {
    "harness_blocked": "agent_error",
    "agent_syntax_error": "agent_error",
    "edit_mismatch": "agent_error",
    "file_not_found": "agent_error",
    "command_not_found": "agent_error",
    "permission": "agent_error",
    "timeout": "workflow",
    "cancelled": "workflow",
    "no_match_probe": "benign",
    "rate_limit": "workflow",
    "mcp_transport": "agent_error",
    "sandbox_egress": "workflow",
    "network": "workflow",
    "build_test_fail": "workflow",
    "other": "workflow",
}

# Write is included: a read-first violation ("File has not been read yet") fires on
# Write too, so a Write-triggered read-first error is an edit_mismatch, not `other`.
_EDIT_TOOLS = {"Edit", "MultiEdit", "edit", "Write", "write"}
# exit 2 from these read-only tools means "path/target not found", not a build fail.
_EXIT2_NOT_FOUND = {"grep", "ugrep", "egrep", "rg", "find", "sed", "ls", "cat"}
# A failing read-only tail on a compound line usually already yielded its data.
_TAIL_READONLY = {"grep", "egrep", "rg", "ugrep", "ls", "find", "sed", "cat", "head", "tail"}

_RATE_LIMIT_RE = re.compile(r"\b(429|529|overloaded|rate[_ ]?limit)\b", re.IGNORECASE)
_MCP_RE = re.compile(r"-32000|Client Closed|connection closed", re.IGNORECASE)
# Sandbox / egress denials use explicit wording; a plain test failure that
# happens to mention "network" or "root" must not land here.
_EGRESS_RE = re.compile(
    r"network\s+(?:is\s+|access\s+(?:is\s+)?)?(?:disabled|blocked|not\s+allowed|unavailable|denied)"
    r"|(?:network\s+)?egress\s+(?:is\s+)?(?:blocked|denied|disabled|restricted)"
    r"|\bnetwork\s+egress\b"
    r"|outside\s+(?:of\s+)?(?:the\s+)?(?:sandbox|workspace|writable|project)\s+root"
    r"|blocked\s+by\s+(?:the\s+)?sandbox"
    r"|sandbox(?:ed)?\s+(?:denied|blocked|violation|restriction|policy)"
    r"|\bseatbelt\b|\bEPERM\b|Operation\s+not\s+permitted",
    re.IGNORECASE,
)
# Harness-side argument rejections. Only trusted when the failure did NOT come
# from a shell command's stdout (see ``_is_harness_rejection``).
_SYNTAX_MARKERS = (
    "InputValidationError",
    "required parameter",
    "unexpected parameter",
    "unexpected argument",
    "invalid_type",
    "-32602",
    "validation error",
    "invalid arguments",
    "invalid params",
)
_NETWORK_RE = re.compile(r"ETIMEDOUT|ECONNREFUSED|ECONNRESET|ENOTFOUND|getaddrinfo|\bnetwork\b", re.IGNORECASE)
# Exit code embedded in the message text (claude "Exit code N", pi "Command
# exited with code N", native codex "Process exited with code N").
_EXIT_IN_TEXT_RE = re.compile(r"(?:Exit code|(?:Process |Command )?exited with code)\s+(\d+)")

# Environment-impediment kinds (§6.4). Deterministic first; JeV only for leftovers.
IMPEDIMENT_KINDS = (
    "sandbox_denied",
    "permission_prompt",
    "auth_failure",
    "rate_limit",
    "network",
    "missing_dependency",
    "missing_resource",
    "tool_crash",
    "timeout",
    "user_interrupt",
)

_IMPEDIMENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sandbox_denied", re.compile(r"sandbox|outside.*(root|workspace)|blocked by policy", re.I)),
    ("permission_prompt", re.compile(r"permission denied|EACCES|askuser|approval required", re.I)),
    ("auth_failure", re.compile(r"\b401\b|unauthorized|invalid.?api.?key|auth(entication|orization) failed", re.I)),
    ("rate_limit", re.compile(r"\b(429|529|overloaded|rate[_ ]?limit)\b", re.I)),
    ("network", re.compile(r"ETIMEDOUT|ECONNREFUSED|ECONNRESET|ENOTFOUND|getaddrinfo|\bdns\b", re.I)),
    ("missing_dependency", re.compile(r"command not found|No module named|Cannot find module|not installed", re.I)),
    ("missing_resource", re.compile(r"No such file or directory|ENOENT|not found", re.I)),
    ("tool_crash", re.compile(r"segfault|panic:|fatal error|Aborted \(core dumped\)", re.I)),
    ("timeout", re.compile(r"timed? ?out|deadline exceeded", re.I)),
    ("user_interrupt", re.compile(r"interrupted|aborted by user|KeyboardInterrupt", re.I)),
]


def classify_impediment(text: str | None) -> str | None:
    """Deterministic impediment kind from an error tail, or ``None`` if ambiguous."""
    if not text:
        return None
    hits = [kind for kind, pat in _IMPEDIMENT_PATTERNS if pat.search(text)]
    return hits[0] if len(hits) == 1 else None


def _exit_code_from_text(m: str) -> int | None:
    match = _EXIT_IN_TEXT_RE.search(m)
    return int(match.group(1)) if match else None


def _is_harness_rejection(m: str, command: str) -> bool:
    """A failure text produced by the harness (MCP ``isError``, function-call
    output, Claude ``<tool_use_error>``) rather than by a shell command. Shell
    stdout is never trusted for argument-schema errors: ``validation`` and
    ``required parameter`` show up in ordinary lint/test output."""
    return not command or "<tool_use_error>" in m


def _classify_category(tool: str, exit_code: int | None, m: str, command: str, head: str | None) -> str:
    if "Blocked:" in m:
        return "harness_blocked"
    low = m.lower()
    if _is_harness_rejection(m, command) and any(marker.lower() in low for marker in _SYNTAX_MARKERS):
        return "agent_syntax_error"
    if tool in _EDIT_TOOLS and (
        "File has not been read yet" in m or "String to replace not found" in m or "overlap" in m
    ):
        return "edit_mismatch"
    if _MCP_RE.search(m):
        return "mcp_transport"
    if _RATE_LIMIT_RE.search(m):
        return "rate_limit"
    if _EGRESS_RE.search(m):
        return "sandbox_egress"
    if "Permission denied" in m or "EACCES" in m:
        return "permission"
    if (
        "EISDIR" in m
        or "ENOENT" in m
        or "No such file or directory" in m
        or "can't read" in m
        or "os error 2" in m
        or (exit_code == 2 and head in _EXIT2_NOT_FOUND)
    ):
        return "file_not_found"
    if exit_code == 127 or ": command not found" in m or ": not found" in m:
        return "command_not_found"
    if exit_code == 143 or "timed out" in m or "timeout" in m.lower():
        return "timeout"
    if exit_code == 130 or "interrupted" in m:
        return "cancelled"
    # no-match probe: grep-family exit 1 with little/no output; `diff`/`cmp`/
    # `test`/`git diff --check` exit 1 just means "differs"/"false".
    if head in ("grep", "rg", "egrep", "ugrep") and exit_code == 1:
        stripped = m.split("\n", 1)[-1].strip() if m.startswith("Exit code") else m.strip()
        if len(stripped) <= 40:
            return "no_match_probe"
    elif exit_code == 1 and command and exit1_is_signal_free(command):
        return "no_match_probe"
    if _NETWORK_RE.search(m):
        return "network"
    if command and classify_command(command) == "build_test" and exit_code not in (0, None):
        return "build_test_fail"
    return "other"


def classify_error(tool: str, exit_code: int | None, msg: str, command: str) -> tuple[str, str, str]:
    """Map one raw error to ``(category, severity, confidence)``.

    First-match-wins over categories; severity is the category default, then a
    compound-command downgrade may soften a failing read-only tail to benign.
    ``confidence`` is ``"high"`` unless that downgrade fired (then ``"low"``).
    """
    m = msg or ""
    head = failing_tail_head(command) if command else None
    if exit_code is None:
        exit_code = _exit_code_from_text(m)

    category = _classify_category(tool, exit_code, m, command, head)
    severity = _CATEGORY_SEVERITY.get(category, "workflow")
    confidence = "high"

    if (
        command
        and is_compound(command)
        and head in _TAIL_READONLY
        and exit_code not in (126, 127, 143)
        and category not in ("harness_blocked", "agent_syntax_error", "edit_mismatch")
    ):
        severity = "benign"
        confidence = "low"
    return category, severity, confidence
