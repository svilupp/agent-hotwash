"""Compatibility exports; all trace decoding belongs to the shared sources."""

from agent_hotwash.primitives.reads import read_paths
from agent_hotwash.semantic.redact import redact_audit_text as redact
from agent_hotwash.semantic.redact import redact_audit_value as _redact_value

TOOL_EDGE_CHARS = 1000
VERSION = "memory-evidence-v2"


def prepare_traces(paths):
    """Prepare supported trace sources through the canonical pipeline."""
    from .inputs import prepare_inputs

    return prepare_inputs(paths)


__all__ = ["TOOL_EDGE_CHARS", "VERSION", "_redact_value", "prepare_traces", "read_paths", "redact"]
