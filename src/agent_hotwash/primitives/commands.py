"""Shell command classifier + sub-intent extraction.

Ported from the code-bench reference (``classify_command`` / ``_segment_intent``)
with the design's coarse label set: ``inspect | mutate | build_test | other``.
Powers exec-intent analytics and the Bash-as-editor smell.
"""

from __future__ import annotations

# Command sub-intent keyword tables (small + documented, easy to extend).
_INSPECT_TOKENS = {"ls", "cat", "head", "tail", "grep", "rg", "find", "which", "wc", "pwd", "tree"}
_INSPECT_GIT = {"status", "diff", "log", "show", "branch"}
_MUTATE_TOKENS = {"rm", "mv", "cp", "mkdir", "touch", "chmod", "chown", "ln", "tee"}
_MUTATE_GIT = {"add", "commit", "checkout", "reset", "restore", "stash", "rebase", "merge", "revert"}
# build_test: these run standalone.
_BUILD_TEST_TOKENS = {"vitest", "jest", "tsc", "make", "pytest", "ruff", "ty", "mypy", "pyright", "eslint"}
# these need a test|build|lint|type-check-ish subcommand to count as build_test.
_PKG_RUNNERS = {"pnpm", "npm", "yarn", "npx", "bun", "bunx", "deno", "cargo", "go"}
_BUILD_TEST_SUBWORDS = {
    "test",
    "build",
    "lint",
    "type-check",
    "typecheck",
    "check",
    "clippy",
    "vet",
    "tsc",
    "vitest",
    "jest",
    "pytest",
    "ruff",
    "mypy",
}
# `python -m <module>` modules that are test/lint runners.
_PY_MODULE_RUNNERS = {"pytest", "unittest", "mypy", "ruff", "pyright", "build"}
# Tools whose *subcommand* decides: `browser-pilot eval` is a test run.
_SUBCOMMAND_RUNNERS = {"browser-pilot": {"eval", "test", "check"}}

# When a compound command mixes intents, the most "load-bearing" one wins.
_INTENT_PRIORITY = ("build_test", "mutate", "inspect", "other")


def _strip_shell_wrapper(cmd: str) -> str:
    """Drop a leading ``bash -lc "..."`` / ``/bin/sh -c '...'`` wrapper (codex)
    so the inner command line is what we classify."""
    cmd = cmd.strip()
    lower = cmd.lower()
    for shell in ("/bin/bash", "bash", "/bin/sh", "sh"):
        if lower.startswith(shell):
            rest = cmd[len(shell) :].lstrip()
            while rest[:1] == "-":  # skip flags like -lc / -c
                sp = rest.find(" ")
                if sp == -1:
                    rest = ""
                    break
                rest = rest[sp + 1 :].lstrip()
            rest = rest.strip()
            if rest[:1] in ("'", '"') and rest[-1:] == rest[:1]:
                rest = rest[1:-1]
            return rest.strip()
    return cmd


def _segment_head(seg: str) -> tuple[str, list[str]] | None:
    """Base name + remaining args for one shell segment, skipping ``cd <dir>``,
    ``sudo``/``env``/``time`` prefixes, and ``VAR=val`` assignments. ``None`` for
    an empty/pure-prefix segment."""
    tokens = seg.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("cd", "sudo", "env", "time", "then", "do"):
            i += 2 if tok == "cd" else 1
            continue
        if "=" in tok and not tok.startswith("-") and "/" not in tok.split("=", 1)[0]:
            i += 1  # VAR=val assignment prefix
            continue
        break
    if i >= len(tokens):
        return None
    return tokens[i].rsplit("/", 1)[-1], tokens[i + 1 :]  # strip any path prefix


def segment_intent(seg: str) -> str | None:
    """Classify a single shell segment (no ``&&``/``;``/``|`` operators) by its
    first bare word. Returns one of ``inspect|mutate|build_test|other`` or
    ``None`` for an empty segment."""
    head_rest = _segment_head(seg)
    if head_rest is None:
        return None
    base, rest = head_rest
    nxt = rest[0] if rest else ""

    if base == "git":
        if nxt in _MUTATE_GIT:
            return "mutate"
        if nxt in _INSPECT_GIT:
            return "inspect"
        return "other"
    if base in _PKG_RUNNERS:
        return "build_test" if any(w in _BUILD_TEST_SUBWORDS for w in rest) else "other"
    if base == "uv":
        if nxt == "run":
            # `uv run python -m pytest` / `uv run ruff check` — classify the inner command.
            inner = segment_intent(" ".join(rest[1:])) if len(rest) > 1 else None
            return inner if inner in ("build_test", "mutate", "inspect") else "build_test"
        return "other"
    if base in ("python", "python3") and "-m" in rest:
        mod = rest[rest.index("-m") + 1] if rest.index("-m") + 1 < len(rest) else ""
        return "build_test" if mod in _PY_MODULE_RUNNERS else "other"
    if base in _SUBCOMMAND_RUNNERS:
        return "build_test" if nxt in _SUBCOMMAND_RUNNERS[base] else "other"
    if base == "sed":
        return "mutate" if any(t == "-i" or t.startswith("-i") for t in rest) else "inspect"
    if base in _BUILD_TEST_TOKENS:
        return "build_test"
    if base in _MUTATE_TOKENS:
        return "mutate"
    if base in _INSPECT_TOKENS:
        return "inspect"
    return "other"


def split_segments(inner: str) -> list[str]:
    """Split a shell script on ``&&``/``||``/``;``/``|`` and newlines into
    non-empty segments (a newline separates commands just like ``;``)."""
    norm = inner.replace("||", "&&").replace(";", "&&").replace("|", "&&").replace("\n", "&&")
    return [s.strip() for s in norm.split("&&") if s.strip()]


def segment_head(seg: str) -> tuple[str, list[str]] | None:
    """Public alias of :func:`_segment_head` — ``(base, args)`` of one segment."""
    return _segment_head(seg)


def strip_shell_wrapper(cmd: str) -> str:
    """Public alias of :func:`_strip_shell_wrapper`."""
    return _strip_shell_wrapper(cmd)


# Commands whose exit status 1 means "no match" / "inputs differ" / "false",
# not an error. ``git diff`` / ``git grep`` behave the same way.
_EXIT1_SIGNAL_FREE = {"grep", "egrep", "fgrep", "rg", "ugrep", "ag", "diff", "cmp", "test", "[", "false"}
_EXIT1_SIGNAL_FREE_GIT = {"diff", "grep", "diff-index", "diff-files"}


def exit1_is_signal_free(cmd: str) -> bool:
    """True when exit status 1 from ``cmd``'s last segment carries no error
    signal (``rg``/``grep`` no match, ``diff``/``cmp`` files differ, ``test`` false,
    ``git diff --check``/``git grep``)."""
    segs = split_segments(_strip_shell_wrapper(cmd))
    if not segs:
        return False
    hr = _segment_head(segs[-1])
    if hr is None:
        return False
    base, rest = hr
    if base in _EXIT1_SIGNAL_FREE:
        return True
    if base == "git":
        sub = next((t for t in rest if not t.startswith("-")), "")
        return sub in _EXIT1_SIGNAL_FREE_GIT
    return False


def command_intents(cmd: str) -> set[str]:
    """Set of segment intents present in a (possibly compound) command."""
    inner = _strip_shell_wrapper(cmd)
    seen: set[str] = set()
    for seg in split_segments(inner):
        intent = segment_intent(seg)
        if intent:
            seen.add(intent)
    return seen


def classify_command(cmd: str) -> str:
    """Classify a shell command into ``inspect | build_test | mutate | other``.

    Compound commands are split and the highest-priority segment intent
    (build_test > mutate > inspect > other) wins.
    """
    inner = _strip_shell_wrapper(cmd)
    if not inner:
        return "other"
    seen = command_intents(cmd)
    for p in _INTENT_PRIORITY:
        if p in seen:
            return p
    return "other"


# Read-only commands whose nonzero exit is usually inconsequential when they are
# the trailing step of a compound line (the agent already got its data upstream).
_TAIL_READONLY = {"grep", "egrep", "rg", "ugrep", "ls", "find", "sed", "cat", "head", "tail"}


def failing_tail_head(cmd: str) -> str | None:
    """Base name of the LAST segment of a (possibly compound) command."""
    segs = split_segments(_strip_shell_wrapper(cmd))
    if not segs:
        return None
    hr = _segment_head(segs[-1])
    return hr[0] if hr else None


def is_compound(cmd: str) -> bool:
    return len(split_segments(_strip_shell_wrapper(cmd))) > 1
