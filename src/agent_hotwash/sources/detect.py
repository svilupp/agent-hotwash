"""Format auto-detection and the single public entry point ``iter_traces``.

``iter_traces(path)`` normalizes a file-or-dir path, fingerprints the format by
structure (not filename alone), expands container dirs, and yields ``Trace``
objects lazily. Detection precedence, per DESIGN §2.1:

- **code-bench** — a *run dir* (``run.json`` + a stdout stream), an *instance
  dir* (children are run dirs), an *experiment dir*, or a *runs root*. Dirs
  whose name ends in ``.quarantine`` / ``.interrupted`` / ``.crashed*`` are
  skipped with a recorded reason.
- **native Claude** — a project dir of ``<session>.jsonl`` files whose records
  carry ``uuid``/``parentUuid``, or a single such session file.
- **native Codex** — a ``rollout-*.jsonl`` (or a sessions tree/date dir of them)
  whose first record is a ``{timestamp,type,payload}`` ``session_meta``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from agent_hotwash.sources import claude_native, codebench, codex_native

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from agent_hotwash.events import Trace

# Run-dir suffixes that mark failed orchestration — skipped with a reason.
_SKIP_SUFFIXES = (".quarantine", ".interrupted", ".crashed")


def _skip_reason(path: Path) -> str | None:
    name = path.name
    for suffix in _SKIP_SUFFIXES:
        if suffix == ".crashed":
            if ".crashed" in name:
                return f"skipped {name}: crashed orchestration dir"
        elif name.endswith(suffix):
            return f"skipped {name}: {suffix.lstrip('.')} orchestration dir"
    return None


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def iter_traces(path: Path) -> Iterator[Trace]:
    """Detect the format at ``path`` and yield every analyzable ``Trace``."""
    path = path.expanduser()
    if not path.exists():
        _warn(f"path does not exist: {path}")
        return

    if path.is_file():
        yield from _iter_file(path)
        return

    reason = _skip_reason(path)
    if reason:
        _warn(reason)
        return

    yield from _iter_dir(path)


def _iter_file(path: Path) -> Iterator[Trace]:
    if codex_native.looks_like_native_codex(path):
        yield codex_native.load_rollout(path)
    elif claude_native.looks_like_native_claude(path):
        yield claude_native.load_session_file(path)
    else:
        _warn(f"unrecognized trace file: {path}")


def _iter_dir(path: Path) -> Iterator[Trace]:
    # code-bench run dir (leaf).
    if codebench.is_run_dir(path):
        trace = codebench.load_run_dir(path)
        if trace is not None:
            yield trace
        return

    # native Claude project dir: contains <session>.jsonl files.
    claude_files = [p for p in sorted(path.glob("*.jsonl")) if claude_native.looks_like_native_claude(p)]
    if claude_files:
        for session_file in claude_files:
            yield claude_native.load_session_file(session_file)
        return

    # native Codex date dir: rollout-*.jsonl directly in this dir. A non-recursive
    # glob (not rglob) is deliberate — a deeper sessions tree (``YYYY/MM/DD/``) is
    # reached by the container recursion below, and, crucially, it keeps detection
    # from descending into a code-bench run dir's harness-internal
    # ``traces/codex/sessions/`` rollout copy (that data is already parsed from the
    # run's stdout.jsonl; ``is_run_dir`` short-circuits before we ever recurse into
    # a run dir). A recursive glob here hijacked whole experiment dirs and re-walked
    # the tree at every level (~25x slower).
    codex_rollouts = [p for p in sorted(path.glob("rollout-*.jsonl")) if codex_native.looks_like_native_codex(p)]
    if codex_rollouts:
        for rollout in codex_rollouts:
            yield codex_native.load_rollout(rollout)
        return

    # code-bench container dir (instance / experiment / runs root): recurse into
    # child dirs looking for run dirs, skipping suffixed orchestration dirs.
    yielded = False
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        reason = _skip_reason(child)
        if reason:
            _warn(reason)
            continue
        for trace in _iter_dir(child):
            yielded = True
            yield trace
    if not yielded:
        _warn(f"no analyzable traces found under: {path}")
