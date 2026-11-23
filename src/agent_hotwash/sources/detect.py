"""Format auto-detection: turn a path into independent, loadable work units.

``discover(path)`` fingerprints the input by structure (not filename alone) and
returns :class:`WorkUnit`s — small, picklable descriptions of one analyzable
trace each. ``load_unit`` turns a unit into ``Trace`` objects; ``iter_traces``
chains the two for callers that just want traces. Splitting discovery from
loading is what lets the runner process units in parallel worker processes
without ever pickling a parsed trace.

Detection precedence, per DESIGN §2.1:

- **code-bench** — a *run dir* (``run.json`` + a stdout stream), an *instance
  dir* (children are run dirs), an *experiment dir*, or a *runs root*. Dirs
  whose name ends in ``.quarantine`` / ``.interrupted`` / ``.crashed*`` are
  skipped with a recorded reason.
- **native Claude** — a project dir of ``<session>.jsonl`` files whose records
  carry ``uuid``/``parentUuid``, or a single such session file.
- **native Codex** — ``rollout-*.jsonl`` files whose first record is a
  ``{timestamp,type,payload}`` ``session_meta``. All rollouts under the given
  directory (a date dir *or* a whole ``sessions/YYYY/MM`` tree) are indexed
  together so parent/child threads on different days still link (PLAN C2).
- **native pi** — ``<ISO-ts>_<uuid>.jsonl`` session files, grouped into
  parent→child trees via ``parentSession`` (PLAN C2 analogue).
- **Cursor Agent** — native ``store.db`` chat checkpoints or CLI stream-json
  captures, including Cloop's ``logs/cloop/*.jsonl``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.sources import (
    claude_native,
    codebench,
    codex_forest,
    codex_native,
    cursor_merge,
    cursor_native,
    cursor_stream,
    pi_native,
)
from agent_hotwash.sources._common import snapshot_fingerprint
from agent_hotwash.sources.codex_forest import CodexComponent, build_codex_forest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agent_hotwash.events import Trace

UnitKind = Literal[
    "codebench_run", "claude_session", "pi_session", "codex_tree", "codex_file", "cursor_store", "cursor_stream"
]

# Run-dir suffixes that mark failed orchestration — skipped with a reason.
_SKIP_SUFFIXES = (".quarantine", ".interrupted", ".crashed")


class WorkUnit(BaseModel):
    """One independently loadable trace source (picklable; no parsed data)."""

    model_config = ConfigDict(frozen=True)

    kind: UnitKind
    paths: list[Path] = Field(default_factory=list)
    alternate_paths: dict[str, list[Path]] = Field(default_factory=dict)
    alternate_snapshots: dict[str, tuple[int, str]] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    issue: str | None = None
    cursor_checkpoint: tuple[str, str] | None = None
    component: CodexComponent | None = None  # codex_tree only

    @property
    def label(self) -> str:
        if self.component is not None:
            return f"{self.kind}:{self.component.root_id} ({len(self.component.paths)} file(s))"
        return f"{self.kind}:{self.paths[0]}" if self.paths else self.kind

    @property
    def size_bytes(self) -> int:
        if self.component is not None:
            return self.component.size_bytes
        total = 0
        for p in self.paths:
            try:
                total += p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            except OSError:
                continue
        return total


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


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover(path: Path) -> list[WorkUnit]:
    """Detect the format at ``path`` and return every analyzable work unit."""
    path = path.expanduser()
    if not path.exists():
        _warn(f"path does not exist: {path}")
        return []
    if path.is_file():
        return _discover_file(path)
    reason = _skip_reason(path)
    if reason:
        _warn(reason)
        return []
    units = discover_many([path])
    if not units:
        _warn(f"no analyzable traces found under: {path}")
    return units


def discover_many(paths, *, diagnostics: list[dict] | None = None) -> list[WorkUnit]:
    """Discover every supported format across inputs and group native families.

    Exact duplicate exports and byte-prefix snapshots are loaded once while
    retaining alternate source coordinates. Non-prefix duplicate native IDs
    retain the forest's explicit ambiguous-lineage diagnostics.
    """
    diagnostics = diagnostics if diagnostics is not None else []
    files: set[Path] = set()
    runs: set[Path] = set()
    visited: set[Path] = set()
    for supplied in paths:
        path = Path(supplied).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"trace input does not exist: {path}")
        if path.is_file():
            files.add(path)
            continue
        stack = [path]
        while stack:
            directory = stack.pop()
            if directory in visited:
                continue
            visited.add(directory)
            reason = _skip_reason(directory)
            if reason:
                diagnostics.append({"source_path": str(directory), "status": "skipped", "reason": reason})
                continue
            if codebench.is_run_dir(directory):
                runs.add(directory)
                continue
            if (
                directory.name == "subagents"
                and directory.parent.with_suffix(".jsonl").is_file()
                and claude_native.looks_like_native_claude(directory.parent.with_suffix(".jsonl"))
            ):
                continue
            for child in sorted(directory.iterdir(), reverse=True):
                if child.is_symlink():
                    continue
                if child.is_dir():
                    stack.append(child)
                elif child.suffix in (".jsonl", ".ndjson") or child.name == "store.db":
                    files.add(child.resolve())
    # Explicit nested stdout or Claude child arguments remain owned by parents
    # already in this corpus, including overlapping directory selections.
    files = {path for path in files if not any(run in path.parents for run in runs)}
    claude_parents = [path for path in files if claude_native.looks_like_native_claude(path)]
    files = {
        path
        for path in files
        if not any(parent.with_suffix("") / "subagents" in path.parents for parent in claude_parents)
    }
    aliases: dict[str, list[Path]] = {}
    by_hash: dict[str, Path] = {}
    fingerprints: dict[Path, tuple[int, str]] = {}
    unique = []
    for path in sorted(files):
        if path.name == "store.db":
            # The committed SQLite state can include WAL data outside this file.
            unique.append(path)
            continue
        size, sha = snapshot_fingerprint(path)
        fingerprints[path] = size, sha
        if sha in by_hash:
            aliases.setdefault(str(by_hash[sha]), []).append(path)
        else:
            by_hash[sha] = path
            unique.append(path)
    codex_paths, pi_paths, units = [], [], [WorkUnit(kind="codebench_run", paths=[run]) for run in sorted(runs)]
    for path in unique:
        found = _discover_file(path)
        if not found:
            diagnostics.append({"source_path": str(path), "status": "unrecognized_trace"})
        for unit in found:
            if unit.kind == "codex_file":
                codex_paths.append(path)
            elif unit.kind == "pi_session":
                pi_paths.append(path)
            else:
                units.append(unit)
    by_native: dict[str, list[Path]] = {}
    for path in codex_paths:
        meta = codex_native.index_session_meta(path) or {}
        if meta.get("id"):
            by_native.setdefault(meta["id"], []).append(path)
    for snapshots in by_native.values():
        if len(snapshots) < 2:
            continue
        largest = max(snapshots, key=lambda path: (path.stat().st_size, str(path)))
        raw = largest.read_bytes()
        if all(raw.startswith(path.read_bytes()) for path in snapshots):
            for path in snapshots:
                if path != largest:
                    codex_paths.remove(path)
                    aliases.setdefault(str(largest), []).append(path)
                    aliases[str(largest)].extend(aliases.pop(str(path), []))
        else:
            diagnostics.append({"status": "ambiguous_duplicate_identity", "source_paths": [str(p) for p in snapshots]})
            for path in snapshots:
                codex_paths.remove(path)
                units.append(WorkUnit(kind="codex_file", paths=[path], degraded=["thread_linkage"]))
    units.extend(codex_units(sorted(codex_paths)))
    units.extend(pi_units(sorted(pi_paths)))
    units = _cursor_units(units, aliases, fingerprints, diagnostics)
    for index, unit in enumerate(units):
        alternatives = {str(path): aliases[str(path)] for path in unit.paths if str(path) in aliases}
        units[index] = unit.model_copy(
            update={
                "alternate_paths": alternatives,
                "alternate_snapshots": {
                    str(path): fingerprints[path]
                    for paths in alternatives.values()
                    for path in paths
                    if path in fingerprints
                },
            }
        )
    return units


def _byte_prefix(shorter: Path, longer: Path, size: int) -> bool:
    with shorter.open("rb") as source, longer.open("rb") as full:
        while size:
            chunk = source.read(min(size, 1 << 20))
            if not chunk or full.read(len(chunk)) != chunk:
                return False
            size -= len(chunk)
    return True


def _cursor_units(
    units: list[WorkUnit],
    aliases: dict[str, list[Path]],
    fingerprints: dict[Path, tuple[int, str]],
    diagnostics: list[dict],
) -> list[WorkUnit]:
    """Collapse proven snapshots and put same-chat Cursor evidence in one work unit."""
    stream_groups: dict[tuple[str | None, str | None], list[WorkUnit]] = {}
    stores: dict[str, list[tuple[str, Path]]] = {}
    output = []
    for unit in units:
        path = unit.paths[0]
        if unit.kind == "cursor_stream":
            stream_groups.setdefault(cursor_stream.stream_identity(path, fingerprints[path][0]), []).append(unit)
        elif unit.kind == "cursor_store" and unit.cursor_checkpoint is not None:
            chat, root = unit.cursor_checkpoint
            stores.setdefault(chat, []).append((root, path))
        else:
            output.append(unit)

    streams_by_chat: dict[str | None, list[WorkUnit]] = {}
    for (chat, anchor), group in stream_groups.items():
        selected: list[WorkUnit] = []
        for unit in sorted(group, key=lambda u: (-fingerprints[u.paths[0]][0], str(u.paths[0]))):
            path = unit.paths[0]
            full = next(
                (
                    u.paths[0]
                    for u in selected
                    if anchor and chat and _byte_prefix(path, u.paths[0], fingerprints[path][0])
                ),
                None,
            )
            if full is None:
                selected.append(unit)
            else:
                aliases.setdefault(str(full), []).extend([path, *aliases.pop(str(path), [])])
        streams_by_chat.setdefault(chat, []).extend(selected)

    for chat, snapshots in stores.items():
        committed = [(checkpoint, path) for checkpoint, path in snapshots if checkpoint]
        if not committed:
            continue
        _root, primary = committed[0]
        if len({checkpoint for checkpoint, _path in committed}) > 1:
            issue = "Selected Cursor stores contain divergent checkpoints for one chat. Select one saved checkpoint."
            diagnostics.append(
                {
                    "status": "ambiguous_cursor_checkpoint",
                    "source_paths": [str(path) for _checkpoint, path in committed],
                }
            )
            output.append(WorkUnit(kind="cursor_store", paths=[primary], issue=issue))
            continue
        aliases.setdefault(str(primary), []).extend(path for _checkpoint, path in committed[1:])
        raw = [unit.paths[0] for unit in streams_by_chat.pop(chat, [])]
        output.append(WorkUnit(kind="cursor_store", paths=[primary, *raw]))
    output.extend(unit for group in streams_by_chat.values() for unit in group)
    return output


def _discover_file(path: Path) -> list[WorkUnit]:
    if codex_native.looks_like_native_codex(path):
        return [WorkUnit(kind="codex_file", paths=[path])]
    if claude_native.looks_like_native_claude(path):
        return [WorkUnit(kind="claude_session", paths=[path])]
    if pi_native.looks_like_native_pi(path):
        return [WorkUnit(kind="pi_session", paths=[path])]
    if cursor_checkpoint := cursor_native.store_identity(path):
        return [WorkUnit(kind="cursor_store", paths=[path], cursor_checkpoint=cursor_checkpoint)]
    if cursor_stream.looks_like_stream(path):
        return [WorkUnit(kind="cursor_stream", paths=[path])]
    _warn(f"unrecognized trace file: {path}")
    return []


def _discover_dir(path: Path) -> list[WorkUnit]:
    # code-bench run dir (leaf).
    if codebench.is_run_dir(path):
        return [WorkUnit(kind="codebench_run", paths=[path])]

    # native Claude project dir: contains <session>.jsonl files.
    claude_files = [p for p in sorted(path.glob("*.jsonl")) if claude_native.looks_like_native_claude(p)]
    if claude_files:
        return [WorkUnit(kind="claude_session", paths=[p]) for p in claude_files]

    # native Codex: every rollout under this dir (date dir or whole tree) is
    # indexed together so cross-day parent/child threads link. The walk stops at
    # code-bench run dirs so a run's harness-internal ``traces/codex/sessions/``
    # rollout copy is never promoted to a second trace.
    codex_rollouts = _collect_codex_rollouts(path)
    if codex_rollouts:
        return codex_units(codex_rollouts)

    # native pi project dir: <ISO-ts>_<uuid>.jsonl session files directly here.
    # Group by parentSession so persisted subagents are one trace, not N.
    pi_sessions = [p for p in sorted(path.glob("*.jsonl")) if pi_native.looks_like_native_pi(p)]
    if pi_sessions:
        return pi_units(pi_sessions)

    # code-bench container dir (instance / experiment / runs root): recurse into
    # child dirs looking for run dirs, skipping suffixed orchestration dirs.
    units: list[WorkUnit] = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        reason = _skip_reason(child)
        if reason:
            _warn(reason)
            continue
        units.extend(_discover_dir(child))
    return units


def _collect_codex_rollouts(path: Path) -> list[Path]:
    """All ``rollout-*.jsonl`` under ``path``, not descending into run dirs."""
    out: list[Path] = []
    stack = [path]
    while stack:
        cur = stack.pop()
        if cur is not path and (codebench.is_run_dir(cur) or _skip_reason(cur)):
            continue
        try:
            children = sorted(cur.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                stack.append(child)
            elif (
                child.name.startswith("rollout-")
                and child.suffix == ".jsonl"
                and codex_native.looks_like_native_codex(child)
            ):
                out.append(child)
    return sorted(out)


def codex_units(paths: list[Path]) -> list[WorkUnit]:
    """Index + link Codex rollouts into one work unit per thread tree."""
    return [
        WorkUnit(kind="codex_tree", paths=list(comp.paths.values()), component=comp)
        for comp in codex_forest.components(codex_forest.index_rollouts(paths))
    ]


def pi_units(paths: list[Path]) -> list[WorkUnit]:
    """Group native pi session files into one work unit per parent→child tree."""
    return [
        WorkUnit(kind="pi_session", paths=[root, *children]) for root, children, _missing in pi_native.components(paths)
    ]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _load_unit(unit: WorkUnit) -> Iterator[Trace]:
    """Parse one work unit into its Trace(s)."""
    if unit.issue:
        raise ValueError(unit.issue)
    if unit.kind == "codebench_run":
        trace = codebench.load_run_dir(unit.paths[0])
        if trace is not None:
            yield trace
    elif unit.kind == "claude_session":
        yield claude_native.load_session_file(
            unit.paths[0], alternate_session_files=unit.alternate_paths.get(str(unit.paths[0]), [])
        )
    elif unit.kind == "pi_session":
        yield pi_native.load_paths(unit.paths)
    elif unit.kind == "codex_file":
        yield codex_native.load_rollout(unit.paths[0])
    elif unit.kind == "codex_tree":
        assert unit.component is not None
        yield codex_forest.load_component(unit.component)
    elif unit.kind == "cursor_store":
        yield from _load_cursor_store(unit.paths)
    elif unit.kind == "cursor_stream":
        yield cursor_stream.load_stream(unit.paths[0])


def _load_cursor_store(paths: list[Path]) -> Iterator[Trace]:
    if len(paths) == 1:
        trace = cursor_native.load_store(paths[0])
        if trace is not None:
            yield trace
        return
    streams, errors = [], []
    for path in paths[1:]:
        try:
            streams.append(cursor_stream.load_stream(path))
        except (OSError, ValueError):
            errors.append({"source_path": str(path), "status": "unreadable_cursor_stream"})
    try:
        trace = cursor_native.load_store(paths[0])
    except (OSError, ValueError):
        if not streams:
            raise
        trace = None
    if trace is None:
        for stream in streams:
            stream.provenance.notes.append("Saved Cursor checkpoint unavailable for this run log.")
            yield stream
    else:
        if errors:
            trace.root.source_diagnostics.extend(errors)
            trace.provenance.notes.append("Some selected Cursor run logs could not be read.")
        yield from cursor_merge.reconcile(trace, streams) if streams else [trace]


def load_unit(unit: WorkUnit) -> Iterator[Trace]:
    """Load canonical evidence with alternate export provenance attached."""
    from agent_hotwash.sources._common import SourceRecord, iter_jsonl, source_artifacts, source_diagnostics

    for trace in _load_unit(unit):
        if unit.degraded:
            trace.root.degraded.extend(item for item in unit.degraded if item not in trace.root.degraded)
            trace.provenance.thread_linkage = "partial"
            trace.provenance.notes.append("ambiguous duplicate identity across non-prefix exports")
        for selected, alternate_paths in unit.alternate_paths.items():
            owners = [
                session
                for session in (trace.root, *trace.subagents)
                if any(artifact["source_path"] == selected for artifact in session.source_artifacts)
            ]
            if not owners:
                continue
            for path in alternate_paths:
                if path.name == "store.db":
                    initial = trace.root.events[0].source
                    if initial is None or cursor_native.store_identity(path) != (
                        trace.root.session_id,
                        initial.item_id,
                    ):
                        trace.provenance.notes.append("An alternate Cursor store checkpoint changed after discovery.")
                        continue
                    artifact = {
                        **trace.root.source_artifacts[0],
                        "source_path": str(path),
                        "bytes": path.stat().st_size,
                    }
                    valid_offsets, diagnostics = None, []
                else:
                    snapshot = unit.alternate_snapshots.get(str(path))
                    try:
                        alternate_records = list(iter_jsonl(path, limit=snapshot[0] if snapshot else None))
                    except (OSError, ValueError):
                        trace.provenance.notes.append("An alternate trace snapshot could not be read.")
                        continue
                    artifacts = source_artifacts(alternate_records)
                    if not artifacts:
                        continue
                    artifact = artifacts[0]
                    if snapshot and (artifact["bytes"], artifact["source_sha256"]) != snapshot:
                        trace.provenance.notes.append("An alternate trace snapshot changed after discovery.")
                        continue
                    valid_offsets = {
                        record.source.byte_offset for record in alternate_records if isinstance(record, SourceRecord)
                    }
                    diagnostics = source_diagnostics(alternate_records)
                for session in owners:
                    session.source_artifacts.append(artifact)
                    session.source_diagnostics.extend(diagnostics)
                    for event in session.events:
                        for source in [event.source, *event.alternate_sources]:
                            if (
                                source
                                and source.path == selected
                                and (valid_offsets is None or source.byte_offset in valid_offsets)
                            ):
                                event.alternate_sources.append(
                                    source.model_copy(
                                        update={"path": str(path), "source_sha256": artifact["source_sha256"]}
                                    )
                                )
                if path not in trace.provenance.files:
                    trace.provenance.files.append(path)
        yield trace


def iter_traces(path: Path) -> Iterator[Trace]:
    """Detect the format at ``path`` and yield every analyzable ``Trace``."""
    for unit in discover(path):
        yield from load_unit(unit)


__all__ = [
    "WorkUnit",
    "build_codex_forest",
    "codex_units",
    "discover",
    "discover_many",
    "iter_traces",
    "load_unit",
    "pi_units",
]
