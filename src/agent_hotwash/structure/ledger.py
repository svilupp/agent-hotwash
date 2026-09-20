"""Active-task ledger: the relationship state for a task (§5.1)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.events import EventKind, TurnStatus
from agent_hotwash.primitives.commands import classify_command

if TYPE_CHECKING:
    from agent_hotwash.events import Event, Turn

AMENDMENT_CAP = 8

# Codex inter-agent envelopes with no payload body are not a task request.
_EMPTY_ENVELOPE_RE = re.compile(
    r"^Message Type:\s*\S+\s*\nTask name:\s*.+\nSender:\s*.+\nPayload:\s*(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_PATH_RE = re.compile(r"\S+\.\w{1,6}")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_DOTTED_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[.:][A-Za-z_][A-Za-z0-9_]*)+\b")


class Ledger(BaseModel):
    """Deterministic per-task state used for turn relationship and digests."""

    model_config = ConfigDict(extra="ignore")

    request: str = ""
    amendments: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    artifacts: set[str] = Field(default_factory=set)
    last_answer: str | None = None
    status: Literal["answered", "aborted", "open"] = "open"
    test_outcomes: list[dict[str, Any]] = Field(default_factory=list)


def normalize_request(text: str) -> str:
    """Empty delegation envelopes become an empty string so task features skip."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    match = _EMPTY_ENVELOPE_RE.match(stripped)
    if match is not None and not match.group(1).strip():
        return ""
    return text.strip()


def extract_deliverables(text: str) -> list[str]:
    """Lexically extract paths, symbols, and URLs from a user message."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def _add(item: str) -> None:
        item = item.strip().rstrip(".,;:)")
        if item and item not in seen:
            seen.add(item)
            out.append(item)

    for m in _URL_RE.finditer(text):
        _add(m.group(0))
    for m in _PATH_RE.finditer(text):
        val = m.group(0)
        if _URL_RE.match(val):
            continue
        _add(val)
    for m in _BACKTICK_RE.finditer(text):
        _add(m.group(1))
    for m in _DOTTED_SYMBOL_RE.finditer(text):
        _add(m.group(0))
    return out


def _command_of(event: Event) -> str:
    args = event.tool_args or {}
    for key in ("command", "cmd"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
    if event.classifications:
        parts = [c.cmd for c in event.classifications if c.cmd]
        if parts:
            return " ".join(parts)
    return event.text or ""


def _is_test_like(event: Event) -> bool:
    cmd = _command_of(event)
    if cmd and classify_command(cmd) == "build_test":
        return True
    name = (event.tool_name or event.op_kind or "").lower()
    return any(tok in name for tok in ("pytest", "jest", "vitest", "test"))


def _status_of(turn: Turn) -> Literal["answered", "aborted", "open"]:
    if turn.status is TurnStatus.aborted:
        return "aborted"
    if turn.status is TurnStatus.completed:
        return "answered"
    return "open"


def update_ledger(ledger: Ledger, turn: Turn, events: list[Event]) -> Ledger:
    """Accumulate ``turn`` + its events into ``ledger`` (mutates and returns it)."""
    text = normalize_request(turn.user_input.text or "")
    if text:
        if not ledger.request:
            ledger.request = text
        elif text != ledger.request and text not in ledger.amendments:
            ledger.amendments = [*ledger.amendments, text][-AMENDMENT_CAP:]
        for item in extract_deliverables(text):
            if item not in ledger.deliverables:
                ledger.deliverables.append(item)

    for ev in events:
        for art in ev.artifacts:
            if art.path:
                ledger.artifacts.add(art.path)
        if ev.path:
            ledger.artifacts.add(ev.path)
        if ev.kind is EventKind.assistant_msg and ev.phase == "final_answer" and ev.text:
            ledger.last_answer = ev.text
        if (
            _is_test_like(ev)
            and ev.kind in (EventKind.tool_call, EventKind.tool_result)
            and (ev.kind is EventKind.tool_result or ev.exit_code is not None or ev.ok is not None)
        ):
            ledger.test_outcomes.append(
                {
                    "cmd": _command_of(ev) or ev.tool_name or "",
                    "exit": ev.exit_code,
                    "ok": ev.ok,
                }
            )

    if turn.final_message:
        ledger.last_answer = turn.final_message
    ledger.status = _status_of(turn)
    return ledger


__all__ = [
    "AMENDMENT_CAP",
    "Ledger",
    "extract_deliverables",
    "normalize_request",
    "update_ledger",
]
