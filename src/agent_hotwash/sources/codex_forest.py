"""Codex thread trees: index rollouts, link them, load one tree at a time.

A *trace* is one connected component of the parent→child thread graph
(PLAN C2). Building it happens in two cheap-then-expensive steps so a
directory of hundreds of rollouts is never held in memory at once:

1. :func:`index_rollouts` reads only the head of each file (first
   ``session_meta`` plus, for ``agent_created_thread`` children, the first
   ``<codex_delegation>`` wrapper) — a few KB per file.
2. :func:`components` turns the index into :class:`CodexComponent`s: plain,
   picklable descriptions (paths + ranked edges) that a worker process can load
   independently with :func:`load_component`.

Linkage precedence (PLAN §4.4): ``parent_thread_id`` (spawn) >
``forked_from_id`` / ``history_base`` (fork) > delegation ``<source_thread_id>``
(created). Every edge records which evidence produced it. Children whose parent
is not in the input become roots with ``thread_linkage="partial"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.events import Capabilities, ThreadLink, ThreadLinkKind
from agent_hotwash.sources import codex_native

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from agent_hotwash.events import Trace


class Edge(BaseModel):
    """A ranked child→parent edge (lower ``rank`` = stronger evidence)."""

    model_config = ConfigDict(frozen=True)

    child_id: str
    parent_id: str
    kind: ThreadLinkKind
    rank: int
    evidence: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"


class CodexComponent(BaseModel):
    """One connected component: the root, its member files, and the edges.

    ``parent_missing`` is set when the root is itself a child whose parent was
    not among the indexed files.
    """

    root_id: str
    paths: dict[str, Path]  # thread id -> rollout path, root included
    edges: list[Edge] = Field(default_factory=list)  # child -> parent, all inside this component
    meta: dict[str, dict[str, Any]] = Field(default_factory=dict)  # thread id -> indexed session_meta fields
    parent_missing: str | None = None

    @property
    def size_bytes(self) -> int:
        total = 0
        for p in self.paths.values():
            try:
                total += p.stat().st_size
            except OSError:
                continue
        return total


# ---------------------------------------------------------------------------
# indexing + linkage
# ---------------------------------------------------------------------------


def index_rollouts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read the linkage-relevant head of every rollout (cheap)."""
    index: list[dict[str, Any]] = []
    for path in paths:
        meta = codex_native.index_session_meta(path)
        if meta is None:
            # No session_meta at all: still a trace, just unlinked.
            meta = {"id": None, "path": path}
        index.append(meta)
    return index


def _rank_edge(meta: dict[str, Any]) -> Edge | None:
    """Strongest edge a child's indexed head supports, or ``None`` for a root."""
    cid = meta.get("id")
    if not cid:
        return None
    parent = meta.get("parent_thread_id")
    forked = meta.get("forked_from_id")
    source = meta.get("thread_source")
    if parent and parent != cid:
        evidence = ["session_meta.parent_thread_id"]
        if source == "subagent":
            evidence.append("source.subagent")
        if forked:
            evidence.append("forked_history")
        return Edge(child_id=cid, parent_id=parent, kind=ThreadLinkKind.spawn, rank=1, evidence=evidence)
    hb = meta.get("history_base")
    fork_parent = forked or (hb.get("thread_id") if isinstance(hb, dict) else None)
    if fork_parent and fork_parent != cid:
        evidence = ["forked_from_id" if forked else "history_base"]
        return Edge(child_id=cid, parent_id=fork_parent, kind=ThreadLinkKind.fork, rank=2, evidence=evidence)
    delegated = meta.get("delegation_source_thread_id")
    if delegated and delegated != cid:
        return Edge(
            child_id=cid,
            parent_id=delegated,
            kind=ThreadLinkKind.created,
            rank=3,
            evidence=["delegation.source_thread_id"],
            confidence="medium",
        )
    return None


def _break_cycles(edges: dict[str, Edge]) -> list[str]:
    """Drop the weakest edge on every cycle (highest ``rank``), in place.

    Returns the child keys whose edge was dropped so callers can note it.
    """
    state: dict[str, int] = {}  # 0 = visiting, 1 = done
    dropped: list[str] = []

    def walk(start: str) -> None:
        path: list[str] = []
        node: str | None = start
        while node is not None and node in edges and state.get(node) is None:
            state[node] = 0
            path.append(node)
            node = edges[node].parent_id
        if node is not None and state.get(node) == 0:
            # cycle: nodes from `node` to end of path
            cycle = path[path.index(node) :]
            weakest = max(cycle, key=lambda n: edges[n].rank)
            edges.pop(weakest, None)
            dropped.append(weakest)
        for n in path:
            state[n] = 1

    for nid in list(edges):
        if state.get(nid) is None:
            walk(nid)
    return dropped


def components(index: list[dict[str, Any]]) -> list[CodexComponent]:
    """Group indexed rollouts into connected components (ordered by root path).

    Nodes are keyed by thread id; a file with a missing or duplicated id gets a
    path-based key and is never linked (a copied rollout must not steal or
    double a parent's children).
    """
    nodes: dict[str, dict[str, Any]] = {}
    keys_by_id: dict[str, list[str]] = {}
    for meta in index:
        tid = meta.get("id")
        key = tid if tid and tid not in nodes else f"{tid or 'no-id'}#{meta['path']}"
        if tid and tid in nodes:
            meta.setdefault("notes", []).append(f"duplicate thread id {tid}; not linked")
        nodes[key] = meta
        if tid:
            keys_by_id.setdefault(tid, []).append(key)

    edges: dict[str, Edge] = {}
    for key, meta in nodes.items():
        if key != meta.get("id"):
            continue  # duplicate / id-less files are standalone roots
        edge = _rank_edge(meta)
        if edge is None:
            continue
        parent_keys = keys_by_id.get(edge.parent_id, [])
        if len(parent_keys) == 1:
            edges[key] = edge
        elif parent_keys:
            meta.setdefault("notes", []).append(f"ambiguous parent {edge.parent_id}; not linked")
        else:
            meta["parent_missing"] = edge.parent_id
    for key in _break_cycles(edges):
        nodes[key].setdefault("notes", []).append("linkage cycle; weakest edge dropped")

    children_of: dict[str, list[str]] = {}
    for key, edge in edges.items():
        children_of.setdefault(edge.parent_id, []).append(key)

    out: list[CodexComponent] = []
    for root_key, meta in nodes.items():
        if root_key in edges:
            continue  # not a root
        members: dict[str, Path] = {root_key: meta["path"]}
        comp_edges: list[Edge] = []
        stack = [root_key]
        while stack:
            pid = stack.pop()
            for child_key in sorted(children_of.get(pid, [])):
                members[child_key] = nodes[child_key]["path"]
                comp_edges.append(edges[child_key])
                stack.append(child_key)
        out.append(
            CodexComponent(
                root_id=root_key,
                paths=members,
                edges=comp_edges,
                meta={k: nodes[k] for k in members},
                parent_missing=meta.get("parent_missing"),
            )
        )
    out.sort(key=lambda c: str(c.paths[c.root_id]))
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_component(comp: CodexComponent) -> Trace:
    """Load every rollout of one component and assemble the Trace."""
    loaded = {tid: codex_native.load_rollout(path) for tid, path in comp.paths.items()}
    root_trace = loaded[comp.root_id]
    root_trace.provenance.files = [comp.paths[tid] for tid in comp.paths]
    linkage_notes = False
    for tid, meta in comp.meta.items():
        for note in meta.get("notes", []):
            loaded[tid].provenance.notes.append(note)
            linkage_notes = True

    children_of: dict[str, list[Edge]] = {}
    for edge in comp.edges:
        children_of.setdefault(edge.parent_id, []).append(edge)

    subagents = []
    links: list[ThreadLink] = []
    stack = [comp.root_id]
    while stack:
        pid = stack.pop()
        for edge in children_of.get(pid, []):
            child = loaded[edge.child_id].root
            child.parent_session_id = pid
            subagents.append(child)
            child_meta = comp.meta.get(edge.child_id, {})
            links.append(
                ThreadLink(
                    child_id=edge.child_id,
                    parent_id=pid,
                    kind=edge.kind,
                    depth=child_meta.get("depth"),
                    history_base=child_meta.get("history_base"),
                    evidence=list(edge.evidence),
                    confidence=edge.confidence,
                )
            )
            stack.append(edge.child_id)

    root_trace.subagents = subagents
    root_trace.links = links
    # Child decoder notes travel with the root so they are visible in the report.
    for child in subagents:
        for note in loaded[child.session_id].provenance.notes:
            root_trace.provenance.notes.append(f"{child.session_id[:8]}: {note}")
    if subagents:
        root_trace.capabilities = Capabilities.merge_min(
            [root_trace.root.capabilities, *[s.capabilities for s in subagents]]
        )
        root_trace.provenance.thread_linkage = root_trace.provenance.thread_linkage or "full"
    if comp.parent_missing:
        root_trace.provenance.notes.append(f"parent not in input: {comp.parent_missing}")
    if comp.parent_missing or linkage_notes:
        # Duplicate/ambiguous ids, dropped cycle edges or an absent parent all
        # mean the tree may be incomplete or double-counted: say so.
        root_trace.provenance.thread_linkage = "partial"
        root_trace.root.degraded.append("thread_linkage")
    return root_trace


def build_codex_forest(paths: list[Path]) -> Iterator[Trace]:
    """Index, link and load every thread tree under ``paths`` (one at a time)."""
    for comp in components(index_rollouts(paths)):
        yield load_component(comp)


__all__ = [
    "CodexComponent",
    "Edge",
    "build_codex_forest",
    "components",
    "index_rollouts",
    "load_component",
]
