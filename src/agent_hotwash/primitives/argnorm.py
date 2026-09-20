"""Argument normalizer + capped edit distance.

``norm_args`` produces a canonical string for a tool call's arguments so that
"the same call" is robust to key ordering and whitespace (RETRY_STORM #7).
``edit_distance`` gives a capped Levenshtein distance for near-identical
detection (NO_ADAPT_RETRY #8).
"""

from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")


def _norm_value(v: Any) -> str:
    if isinstance(v, str):
        return _WS.sub(" ", v.strip())
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}={_norm_value(v[k])}" for k in sorted(v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_norm_value(x) for x in v) + "]"
    return str(v)


def norm_args(args: dict[str, Any] | None) -> str:
    """Canonical, order-independent string form of a tool call's args.

    Keys are sorted, string values whitespace-collapsed and stripped; nested
    dicts/lists are normalized recursively. Two calls that differ only in key
    order or incidental whitespace produce identical output.
    """
    if not args:
        return ""
    return ";".join(f"{k}={_norm_value(args[k])}" for k in sorted(args))


def edit_distance(a: str, b: str, cap: int = 100) -> int:
    """Levenshtein distance between ``a`` and ``b``, capped at ``cap``.

    Only the diagonal band of cells that can still yield a distance below
    ``cap`` is computed (Ukkonen's bound), so the cost is ``O(cap * len)`` rather
    than ``O(len(a) * len(b))`` — a Codex command string can be a 10 kB script,
    and detectors compare many of them with ``cap`` in the single digits.
    Returns ``cap`` as soon as the distance is known to reach it.
    """
    if a == b:
        return 0
    if cap <= 0:
        return 0
    n, m = len(a), len(b)
    if abs(n - m) >= cap:
        return cap
    k = cap - 1  # widest band that can still finish below cap
    # prev[j] = distance between a[:i-1] and b[:j], for j within the band.
    prev: dict[int, int] = {j: j for j in range(0, min(m, k) + 1)}
    for i in range(1, n + 1):
        ca = a[i - 1]
        cur: dict[int, int] = {}
        row_min = cap
        lo, hi = max(0, i - k), min(m, i + k)
        for j in range(lo, hi + 1):
            if j == 0:
                val = i
            else:
                val = min(
                    prev.get(j, cap) + 1,  # deletion
                    cur.get(j - 1, cap) + 1,  # insertion
                    prev.get(j - 1, cap) + (0 if ca == b[j - 1] else 1),  # substitution
                )
            cur[j] = val if val < cap else cap
            row_min = min(row_min, cur[j])
        if row_min >= cap:
            return cap
        prev = cur
    return min(prev.get(m, cap), cap)
