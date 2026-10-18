"""Local command datasets and declarative extraction over canonical evidence."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter, defaultdict
from typing import Any

from agent_hotwash.config import CommandHookConfig, MemoryConfig

_SHELL_TOOLS = {
    "bash",
    "shell",
    "exec_command",
    "command_execution",
    "shell_command",
    "run_shell_command",
    "execute",
    "terminal",
}
_JS_STRING = r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`"""
_JS_TOKEN = re.compile(rf"//[^\n]*|/\*[\s\S]*?\*/|{_JS_STRING}|[A-Za-z_$][\w$]*|[^\s]", re.MULTILINE)
_JS_REGEX = re.compile(r"/(?:\\.|\[(?:\\.|[^\]\\])*\]|[^/\\\n])+/[A-Za-z]*")
_OPTION_VALUES = {
    "--harness",
    "--project",
    "--config-dir",
    "--scope",
    "--limit",
    "--max-bytes",
    "--section",
    "--if-hash",
}


def _js_literal(token: str) -> str | None:
    if not token or token[0] not in "\"'`" or token[-1] != token[0]:
        return None
    if token[0] == "`" and "${" in token:
        return None
    if token[0] == '"':
        try:
            return json.loads(token)
        except ValueError:
            pass
    body = token[1:-1]
    return re.sub(
        r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|\r?\n|.)",
        lambda match: (
            chr(int(match[1][1:], 16))
            if match[1].startswith(("u", "x"))
            else {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "\n": "", "\r\n": ""}.get(match[1], match[1])
        ),
        body,
    )


def _nested_commands(code: str) -> tuple[list[str], list[str]]:
    tokens = []
    skip_until = 0
    for token_match in _JS_TOKEN.finditer(code):
        if token_match.start() < skip_until:
            continue
        token = token_match.group()
        if token.startswith(("//", "/*")):
            continue
        expression_start = not tokens or tokens[-1] in {"=", "(", "[", "{", ":", ",", ";", "!", "?", "return", "=>"}
        if token == "/" and expression_start:
            regex_literal = _JS_REGEX.match(code, token_match.start())
            if regex_literal is None:
                return [], ["javascript_regex_literal_not_parsed"]
            skip_until = regex_literal.end()
            tokens.append("<regex literal>")
            continue
        tokens.append(token)
    commands, diagnostics = [], []
    for index in range(len(tokens) - 3):
        if tokens[index : index + 4] != ["tools", ".", "exec_command", "("]:
            continue
        start = index + 4
        if start >= len(tokens) or tokens[start] != "{":
            diagnostics.append("nested_exec_argument_is_dynamic")
            continue
        depth, found = 0, False
        for position in range(start, len(tokens)):
            token = tokens[position]
            if token in ("{", "[", "("):
                depth += 1
            elif token in ("}", "]", ")"):
                depth -= 1
                if depth == 0:
                    break
            if depth != 1 or token.strip("\"'") not in ("cmd", "command"):
                continue
            if position + 3 >= len(tokens) or tokens[position + 1] != ":":
                continue
            literal = _js_literal(tokens[position + 2])
            terminator = tokens[position + 3]
            if literal is not None and terminator in (",", "}"):
                commands.append(literal)
                found = True
            else:
                diagnostics.append("nested_exec_command_is_dynamic")
                found = True
        if not found:
            diagnostics.append("nested_exec_command_not_found")
    if not commands and not diagnostics:
        diagnostics.append("no_literal_nested_exec_command")
    return commands, diagnostics


def _commands(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    arguments = record.get("arguments", {})
    if not isinstance(arguments, dict):
        return [], ["tool_arguments_are_not_an_object"]
    name = (record.get("tool_name") or "").lower()
    short = re.split(r"[.:/]+|__", name)[-1]
    if short in _SHELL_TOOLS or name == "cmd.exec":
        value = next((arguments[k] for k in ("cmd", "command", "script") if k in arguments), None)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [shlex.join(value)], []
        return ([value], []) if isinstance(value, str) else ([], ["shell_command_is_not_literal"])
    if short == "exec" or name.endswith("functions.exec"):
        value = next((arguments[k] for k in ("code", "input", "source", "script") if k in arguments), None)
        if isinstance(value, str):
            return _nested_commands(value)
        return [], ["javascript_source_is_not_available"]
    return [], []


def _argv(command: str) -> list[str]:
    raw_tokens = re.findall(r"(?:[^\s\"'\\]|\\.|\"(?:\\.|[^\"\\])*\"|'[^']*')+", command)
    kept = []
    skip_target = False
    for raw in raw_tokens:
        if skip_target:
            skip_target = False
            continue
        if re.match(r"^\d*(?:>>?|<|>&|<&)", raw):
            skip_target = bool(re.fullmatch(r"\d*(?:>>?|<|>&|<&)", raw))
            continue
        kept.append(raw)
    tokens = shlex.split(" ".join(kept))
    while tokens:
        head = tokens[0]
        base = head.rsplit("/", 1)[-1]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", head):
            tokens.pop(0)
        elif base in ("env", "sudo", "time", "command"):
            tokens.pop(0)
            value_options = {
                "sudo": {
                    "-u",
                    "-g",
                    "-h",
                    "-p",
                    "-C",
                    "-T",
                    "-R",
                    "-D",
                    "--user",
                    "--group",
                    "--host",
                    "--prompt",
                    "--chdir",
                    "--chroot",
                },
                "env": {"-u", "--unset", "-C", "--chdir"},
                "time": {"-f", "--format", "-o", "--output"},
            }.get(base, set())
            while tokens and tokens[0].startswith("-"):
                option = tokens.pop(0)
                if option in value_options and tokens:
                    tokens.pop(0)
        elif base == "uv" and len(tokens) > 1 and tokens[1] == "run":
            tokens = tokens[2:]
        else:
            break
    return tokens


def _shell_segments(command: str) -> tuple[list[str], bool]:
    segments, buffer = [], []
    quote = None
    escaped = False
    dynamic = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            buffer.append(char)
            escaped = False
        elif char == "\\" and quote != "'":
            buffer.append(char)
            escaped = True
        elif quote:
            buffer.append(char)
            if char == quote:
                quote = None
            elif quote != "'" and char in "$`":
                dynamic = True
        elif char in "\"'":
            quote = char
            buffer.append(char)
        elif char == "#" and (index == 0 or command[index - 1].isspace()):
            newline = command.find("\n", index)
            index = len(command) if newline == -1 else newline - 1
        elif char in ";|\n" or (char == "&" and (index == 0 or command[index - 1] not in "<>")):
            segment = "".join(buffer).strip()
            if segment:
                segments.append(segment)
            buffer = []
            if command[index : index + 2] in ("&&", "||"):
                index += 1
        else:
            buffer.append(char)
            if char in "$`":
                dynamic = True
        index += 1
    if "".join(buffer).strip():
        segments.append("".join(buffer).strip())
    return segments, dynamic


def _segments(command: str) -> tuple[list[str], list[str]]:
    heredoc = next(
        (
            match
            for match in re.finditer(r"'[^']*'|\"(?:\\.|[^\"\\])*\"|\\.|(?<!\S)\#[^\n]*|<<", command)
            if match.group() == "<<"
        ),
        None,
    )
    if heredoc is not None:
        newline = command.find("\n", heredoc.end())
        header = command if newline == -1 else command[:newline]
        segments, dynamic = _shell_segments(header)
        issues = ["heredoc_body_not_parsed"]
        if dynamic:
            issues.append("shell_dynamic_expansion_not_resolved")
        return segments, issues
    try:
        tokens = _argv(command)
    except ValueError:
        return [command], ["shell_quoting_invalid"]
    if tokens and tokens[0].rsplit("/", 1)[-1] in ("bash", "sh", "zsh", "dash"):
        for index, token in enumerate(tokens[1:], 1):
            if token.startswith("-") and "c" in token and index + 1 < len(tokens):
                return _segments(tokens[index + 1])
    segments, dynamic = _shell_segments(command)
    issues = ["shell_dynamic_expansion_not_resolved"] if dynamic else []
    if re.search(r"(?:^|[;\n])\s*(?:if|for|while|case|function)\b", command):
        issues.append("shell_control_flow_not_evaluated")
    return segments, issues


def _positionals(tokens: list[str]) -> list[str]:
    values = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            values.extend(tokens[index + 1 :])
            break
        if token in _OPTION_VALUES:
            index += 2
        elif token.startswith("-"):
            index += 1
        else:
            values.append(token)
            index += 1
    return values


def _preset_values(preset: str, command: str) -> list[str] | None:
    try:
        tokens = _argv(command)
    except ValueError:
        return None
    if not tokens:
        return None
    executable, positional = tokens[0].rsplit("/", 1)[-1], _positionals(tokens[1:])
    if preset == "playbooks":
        if executable != "agent-playbooks" or not positional or positional[0] not in ("read", "load", "show"):
            return None
        return [name.strip() for value in positional[1:] for name in value.split(",") if name.strip()]
    if executable != "mem" or not positional:
        return None
    if preset == "mem_index":
        return [] if positional[0] == "index" else None
    if positional[0] not in ("search", "deepsearch"):
        return None
    return [" ".join(positional[1:])] if positional[1:] else []


def collect_command_hooks(records, hooks=None) -> dict[str, Any]:
    """Extract literal shell calls once, retaining provenance and parse limitations."""
    hooks = MemoryConfig().command_hooks if hooks is None else tuple(CommandHookConfig.model_validate(h) for h in hooks)
    calls, matches, diagnostics = [], [], []
    seen = set()
    for record in records:
        if record.get("kind") != "tool_call" or record["id"] in seen:
            continue
        seen.add(record["id"])
        commands, issues = _commands(record)
        index = 0
        for command in commands:
            segments, segment_issues = _segments(command)
            issues.extend(segment_issues)
            for segment in segments:
                call_id = "command-" + hashlib.sha256(f"{record['id']}:{index}".encode()).hexdigest()[:24]
                row = {
                    "id": call_id,
                    "evidence_id": record["id"],
                    "session_id": record["session_id"],
                    "tool_name": record.get("tool_name"),
                    "command": segment,
                    "shell_command": command,
                    "command_index": index,
                    "parse_status": "partial" if issues else "literal",
                    **{key: record.get(key) for key in ("source_path", "line_number", "source_refs")},
                }
                calls.append(row)
                index += 1
                _, segment_dynamic = _shell_segments(segment)
                for hook in hooks:
                    if segment_dynamic or ("heredoc_body_not_parsed" in segment_issues and "<<" in segment):
                        continue
                    try:
                        match_text = shlex.join(_argv(segment)) if hook.preset else segment
                    except ValueError:
                        continue
                    if not any(re.search(pattern, match_text) for pattern in hook.patterns):
                        continue
                    values = _preset_values(hook.preset, segment) if hook.preset else []
                    if values is None:
                        continue
                    captures = []
                    for pattern in hook.extract_patterns:
                        for match in re.finditer(pattern, segment):
                            captured = {k: v for k, v in match.groupdict().items() if v is not None}
                            captures.append(captured)
                            extracted = list(captured.values()) or list(match.groups()) or [match.group()]
                            for value in extracted:
                                if value is not None:
                                    values.extend(value.split(hook.split_values) if hook.split_values else [value])
                    values = [value.strip() for value in values if value.strip()]
                    matches.append({"call_id": call_id, "hook": hook.name, "values": values, "captures": captures})
        diagnostics.extend({"evidence_id": record["id"], "reason": reason} for reason in sorted(set(issues)))
    sessions = {row["id"]: row["session_id"] for row in calls}
    summaries = []
    for hook in hooks:
        selected = [row for row in matches if row["hook"] == hook.name]
        counts: Counter[str] = Counter()
        value_sessions: dict[str, set[str]] = defaultdict(set)
        for row in selected:
            for value in row["values"]:
                counts[value] += 1
                value_sessions[value].add(sessions[row["call_id"]])
        summaries.append(
            {
                "name": hook.name,
                "call_count": len(selected),
                "session_count": len({sessions[row["call_id"]] for row in selected}),
                "values": [
                    {"value": value, "count": count, "session_count": len(value_sessions[value])}
                    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                ],
            }
        )
    return {"calls": calls, "matches": matches, "summaries": summaries, "diagnostics": diagnostics}
