"""Project JeV state onto inspect/compare paths so the model only sees evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

Token = str | tuple[str, int | None]
_STRUCTURE_SIBLINGS = ("kind",)
_CONTEXT_SIBLINGS: dict[str, tuple[str, ...]] = {
    "deliverables": ("request",),
    "artifacts": ("request",),
    "last_answer": ("request",),
    "kind": ("cmd", "exit"),
}


def inspect_paths_of(questions: Mapping[str, Any]) -> list[str]:
    """Collect backticked inspect/compare paths from wired question bodies."""
    out: list[str] = []
    seen: set[str] = set()
    for question in questions.values():
        ins = question.get("instructions") if isinstance(question, dict) else None
        if not isinstance(ins, dict):
            continue
        for key in ("inspect", "compare"):
            for path in _path_list(ins.get(key)):
                if path not in seen:
                    seen.add(path)
                    out.append(path)
    return out


def group_wired_questions(questions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Mapping[str, Any]]]:
    """Keep questions with different inspect/compare sets off the same request.

    The cache hashes ``(model, request_state, question)``. Mixing inspect sets
    would bind one question's answer to another question's evidence.
    """
    groups: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for qid, question in questions.items():
        key = tuple(inspect_paths_of({qid: question}))
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][qid] = question
    return [groups[key] for key in order]


def is_heavy_inspect(path: str) -> bool:
    """True when this path would leak op bodies or whole-fact objects into a batch."""
    cleaned = path.strip().strip("`")
    if cleaned in {"episode.ops", "episode.facts", "episode.messages"}:
        return True
    return "out_head" in cleaned or "out_tail" in cleaned


def project_state(state: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """Keep only JSON nodes named by ``paths``. Missing paths are omitted.

    Questions without inspect/compare paths leave ``state`` unchanged so ad-hoc
    callers still work. Array indexes stay aligned so ``messages[0]`` /
    ``messages[-1]`` mean the same elements as on the source digest.
    """
    if not paths or not isinstance(state, dict):
        return state
    out: dict[str, Any] = {}
    copied = False
    for path in paths:
        tokens = tokenize_path(path)
        if not tokens:
            continue
        if _copy(state, out, tokens):
            copied = True
    return out if copied else {}


def tokenize_path(path: str) -> list[Token]:
    """Split ``episode.messages[-1].text`` into keys and index tokens."""
    raw = path.strip().strip("`")
    if not raw:
        return []
    tokens: list[Token] = []
    buf = ""
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
            continue
        if ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            close = raw.find("]", i)
            if close < 0:
                break
            inner = raw[i + 1 : close].strip()
            if inner == "":
                tokens.append(("all", None))
            else:
                try:
                    tokens.append(("idx", int(inner)))
                except ValueError:
                    return []
            i = close + 1
            continue
        buf += ch
        i += 1
    if buf:
        tokens.append(buf)
    return tokens


def _path_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        path = str(item).strip().strip("`")
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _copy(src: Any, dest: Any, tokens: list[Token]) -> bool:
    if not tokens:
        return False
    tok, rest = tokens[0], tokens[1:]
    if isinstance(tok, tuple):
        kind, idx = tok
        if not isinstance(src, list) or not isinstance(dest, list):
            return False
        if kind == "all":
            wrote = False
            _ensure_list_len(dest, len(src))
            for i, item in enumerate(src):
                if not rest:
                    dest[i] = item
                    wrote = True
                    continue
                dest[i] = _child_dest(dest[i], rest)
                if _copy(item, dest[i], rest):
                    wrote = True
            return wrote
        if kind == "idx" and idx is not None:
            i = idx if idx >= 0 else len(src) + idx
            if i < 0 or i >= len(src):
                return False
            _ensure_list_len(dest, i + 1)
            if not rest:
                dest[i] = src[i]
                return True
            dest[i] = _child_dest(dest[i], rest)
            return _copy(src[i], dest[i], rest)
        return False
    if not isinstance(src, dict) or tok not in src:
        return False
    if not isinstance(dest, dict):
        return False
    child = src[tok]
    if not rest:
        dest[tok] = child
        _copy_structure_siblings(src, dest, tok)
        return True
    dest[tok] = _child_dest(dest.get(tok), rest)
    return _copy(child, dest[tok], rest)


def _copy_structure_siblings(src: dict[str, Any], dest: dict[str, Any], leaf: str) -> None:
    """Keep small discriminator fields on the same object as an inspect leaf."""
    extra = _CONTEXT_SIBLINGS.get(leaf, ())
    for key in (*_STRUCTURE_SIBLINGS, *extra):
        if key == leaf or key not in src or key in dest:
            continue
        val = src[key]
        if val is None or isinstance(val, (str, int, float, bool)):
            dest[key] = val


def _ensure_list_len(dest: list[Any], n: int) -> None:
    while len(dest) < n:
        dest.append(None)


def _child_dest(current: Any, rest: list[Token]) -> Any:
    nxt = rest[0]
    want_list = isinstance(nxt, tuple)
    if want_list:
        return current if isinstance(current, list) else []
    return current if isinstance(current, dict) else {}


__all__ = [
    "group_wired_questions",
    "inspect_paths_of",
    "is_heavy_inspect",
    "project_state",
    "tokenize_path",
]
