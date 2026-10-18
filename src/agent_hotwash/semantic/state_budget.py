"""Bound diagnostic evidence without changing source ledgers or hiding omissions."""

from __future__ import annotations

import hashlib
from typing import Any

from systemoneprompts.json_values import canonical_json

CLIPPING_KEY = "_diagnostic_state_clipping"


class DiagnosticStateTooLarge(ValueError):
    """Structure and small scalar fields alone exceed the diagnostic budget."""


def serialized_bytes(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def bound_diagnostic_state(state: dict[str, Any], max_bytes: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Retain whole structure and UTF-8-safe string heads/tails within a byte cap.

    Callers project and redact first. Small states are returned unchanged, keeping
    their existing native cache identity. Large states include verifiable clipping
    provenance in the provider state as well as in the report's feature values.
    """
    original_bytes = serialized_bytes(state)
    if original_bytes <= max_bytes:
        return state, None
    digest = hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()

    def candidate(cap: int) -> tuple[dict[str, Any], dict[str, Any]]:
        paths: list[str] = []
        collections: list[dict[str, Any]] = []

        def clip(value: Any, path: str) -> Any:
            if isinstance(value, str) and len(value) > cap:
                head = value[: (cap + 1) // 2]
                tail = value[-(cap // 2) :] if cap // 2 else ""
                omitted = len(value.encode("utf-8")) - len((head + tail).encode("utf-8"))
                paths.append(path)
                return head + f"\n[… {omitted} UTF-8 bytes omitted …]\n" + tail
            if isinstance(value, dict):
                return {
                    key: clip(item, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                limit = max(4, cap // 64)
                if len(value) > limit:
                    head_count = (limit + 1) // 2
                    tail_start = len(value) - limit // 2
                    omitted = len(value) - limit
                    collections.append(
                        {
                            "path": path,
                            "original_items": len(value),
                            "retained_items": limit,
                            "omitted_items": omitted,
                            "original_sha256": hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest(),
                        }
                    )
                    return (
                        [clip(item, f"{path}/{index}") for index, item in enumerate(value[:head_count])]
                        + [{"_diagnostic_omitted_items": omitted}]
                        + [clip(value[index], f"{path}/{index}") for index in range(tail_start, len(value))]
                    )
                return [clip(item, f"{path}/{index}") for index, item in enumerate(value)]
            return value

        bounded = clip(state, "")
        metadata = {
            "version": 1,
            "original_bytes": original_bytes,
            "retained_bytes": 0,
            "original_sha256": digest,
            "clipped_paths": paths,
            "clipped_collections": collections,
            "max_bytes": max_bytes,
        }
        bounded[CLIPPING_KEY] = metadata
        for _ in range(4):
            size = serialized_bytes(bounded)
            if metadata["retained_bytes"] == size:
                break
            metadata["retained_bytes"] = size
        return bounded, metadata

    # Do not erase short identifiers, statuses, flags, or structure to force a fit.
    best, metadata = candidate(64)
    if serialized_bytes(best) > max_bytes:
        raise DiagnosticStateTooLarge("Diagnostic structure exceeds byte budget after clipping long strings")
    low, high = 64, original_bytes
    while low <= high:
        middle = (low + high) // 2
        bounded, info = candidate(middle)
        if serialized_bytes(bounded) <= max_bytes:
            best, metadata = bounded, info
            low = middle + 1
        else:
            high = middle - 1
    return best, metadata
