"""Conservative shared read-target classification for tool and shell calls."""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

from agent_hotwash.primitives.commands import split_segments


def shell_read_tokens(command: str) -> list[str]:
    """Tokenize a read/search segment while excluding redirect destinations."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars="<>")
    lexer.whitespace_split = True
    output, skip = [], False
    for token in lexer:
        if skip:
            skip = False
            continue
        if token == "<":
            if output and output[-1].isdigit():
                output.pop()
            continue
        if token in (">", ">>", "<<", "<<<", ">&", "<&"):
            if output and output[-1].isdigit():
                output.pop()
            skip = True
            continue
        output.append(token)
    return output


def _clean_path(value):
    if not isinstance(value, str) or not value or value.startswith("-"):
        return None
    if any(c in value for c in "*?{}\n<>") or value in (".", "..", "/", "--"):
        return None
    return value.rstrip(":,")


def shell_path_arguments(tokens: list[str]) -> list[str]:
    """Extract file operands, keeping option-supplied patterns out of paths."""
    executable = Path(tokens[0]).name
    args = tokens[1:]
    pattern_paths: list[str] = []
    positional = []
    pattern_supplied = False
    pattern_file = False
    skip = False
    value_flags = {"-n", "-c", "-m", "-A", "-B", "-C", "--max-count", "--glob", "-g", "-e", "-f"}
    if executable in ("rg", "grep", "egrep", "fgrep", "ugrep", "sed"):
        value_flags.discard("-n")
    for token in args:
        if token in (">", ">>", "<", "<<", "<<<", ">&", "<&"):
            if positional and positional[-1].isdigit():
                positional.pop()
            skip = True
            continue
        if skip:
            skip = False
            if pattern_file:
                path = _clean_path(token)
                if path:
                    pattern_paths.append(path)
                pattern_file = False
            continue
        if executable in ("rg", "grep", "egrep", "fgrep", "ugrep", "sed"):
            if token in ("-e", "--regexp", "--expression", "-f", "--file"):
                pattern_supplied = True
                pattern_file = token in ("-f", "--file")
                skip = True
                continue
            if token.startswith(("--regexp=", "--expression=", "--file=")) or (
                token.startswith(("-e", "-f")) and len(token) > 2
            ):
                pattern_supplied = True
                if token.startswith(("-f", "--file=")):
                    path = _clean_path(token.split("=", 1)[1] if "=" in token else token[2:])
                    if path:
                        pattern_paths.append(path)
                continue
        if token in value_flags:
            skip = True
            continue
        if token.startswith("-"):
            continue
        positional.append(token)
    if executable in ("sed", "rg", "grep", "egrep", "fgrep", "ugrep") and not pattern_supplied:
        positional = positional[1:]
    return pattern_paths + positional


def read_paths(name, arguments):
    """Count observed read targets; shell inference is deliberately conservative."""
    lower = name.lower() if isinstance(name, str) else ""
    if any(part in lower for part in ("write", "edit", "patch", "mkdir", "delete")):
        return []
    if any(part in lower for part in ("read", "open", "view", "file_search")):
        return sorted(
            {p for key in ("file_path", "path", "filename", "file") if (p := _clean_path(arguments.get(key)))}
        )
    command = arguments.get("cmd") or arguments.get("command")
    if not isinstance(command, str):
        return []
    paths = set()
    if "exec_command" in command:
        literals = re.finditer(r"(?:cmd|command)\s*:\s*(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`)", command)
        for match in literals:
            value = match[1]
            try:
                literal = value[1:-1] if value.startswith("`") and "${" not in value else ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            paths.update(read_paths("shell", {"cmd": literal}))
        return sorted(paths)
    for segment in split_segments(command):
        try:
            tokens = shell_read_tokens(segment)
        except ValueError:
            continue
        if not tokens:
            continue
        executable = Path(tokens[0]).name
        if executable not in ("cat", "head", "tail", "sed", "rg", "grep", "less", "more", "bat"):
            continue
        args = tokens[1:]
        if executable == "rg" and any(x in args for x in ("--files", "--files-with-matches", "-l")):
            continue
        positional = shell_path_arguments(tokens)
        for token in positional:
            path = _clean_path(token)
            if executable in ("rg", "grep") and path and "." not in Path(path).name:
                continue
            if path and (("/" in path or "." in Path(path).name) or executable in ("cat", "less", "more", "bat")):
                paths.add(path)
    return sorted(paths)
