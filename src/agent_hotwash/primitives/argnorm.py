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

    Returns ``cap`` as soon as the true distance is known to exceed it (both the
    length gap and the running row minimum are used to bail early), so this stays
    cheap on long, very different strings.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) >= cap:
        return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            row_min = min(row_min, val)
        if row_min >= cap:
            return cap
        prev = cur
    return min(prev[-1], cap)
