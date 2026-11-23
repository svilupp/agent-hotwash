"""Read Cursor's protobuf checkpoint fields and join UI timing to the model transcript."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_hotwash.events import EventKind
from agent_hotwash.sources._common import parse_ts

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from datetime import datetime

Fields = dict[int, list[bytes | int]]
if TYPE_CHECKING:
    type Timing = tuple[datetime | None, datetime | None]


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            break
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80 and (shift < 63 or byte <= 1):
            return value, offset
    raise ValueError("Invalid varint in Cursor checkpoint")


def protobuf(data: bytes) -> Fields:
    fields: Fields = {}
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("Invalid field in Cursor checkpoint")
        if wire == 0:
            value, offset = _varint(data, offset)
            fields.setdefault(number, []).append(value)
            continue
        if wire == 2:
            size, offset = _varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
        else:
            raise ValueError("Unsupported wire type in Cursor checkpoint")
        end = offset + size
        if end > len(data):
            raise ValueError("Truncated Cursor checkpoint")
        if wire == 2:
            fields.setdefault(number, []).append(data[offset:end])
        offset = end
    return fields


def bytes_fields(fields: Fields, number: int) -> Iterator[bytes]:
    return (value for value in fields.get(number, []) if isinstance(value, bytes))


def blob_ids(fields: Fields, number: int) -> Iterator[str]:
    return (value.hex() for value in bytes_fields(fields, number) if len(value) == 32)


def integer(fields: Fields, number: int) -> int | None:
    return next((value for value in fields.get(number, []) if isinstance(value, int)), None)


def text(blob: bytes | None) -> str | None:
    if blob is None:
        return None
    try:
        return blob.decode()
    except UnicodeDecodeError:
        return None


def text_field(fields: Fields, number: int) -> str | None:
    return text(next(bytes_fields(fields, number), None))


def read_blob(db: sqlite3.Connection, blob_id: str) -> bytes | None:
    row = db.execute("SELECT data FROM blobs WHERE id = ?", (blob_id,)).fetchone()
    return row[0] if row is not None and isinstance(row[0], bytes) else None


def _time(fields: Fields, number: int) -> datetime | None:
    value = integer(fields, number)
    return parse_ts(value) if value and value > 0 else None


@dataclass
class CheckpointTiming:
    prompts: dict[str, datetime] = field(default_factory=dict)
    tools: dict[str, Timing] = field(default_factory=dict)
    messages: dict[tuple[str, EventKind, str], deque[Timing]] = field(default_factory=lambda: defaultdict(deque))
    final_replies: dict[str, tuple[str, datetime]] = field(default_factory=dict)

    def message(self, request: str | None, kind: EventKind, content: str) -> Timing:
        timings = self.messages.get((request or "", kind, content))
        return timings.popleft() if timings else (None, None)


def _step_timing(timing: CheckpointTiming, request_id: str, step: Fields) -> tuple[str, datetime] | None:
    final = None
    for number, payloads in step.items():
        for payload in payloads:
            if not isinstance(payload, bytes) or number not in (1, 2, 3):
                continue
            detail = protobuf(payload)
            if number == 2:
                if call_id := text_field(detail, 57):
                    timing.tools[call_id] = _time(detail, 59), _time(detail, 60)
            elif content := text_field(detail, 1):
                kind = EventKind.assistant_msg if number == 1 else EventKind.thinking
                start_key = 2 if number == 1 else 3
                timing.messages[(request_id, kind, content)].append(
                    (_time(detail, start_key), _time(detail, start_key + 1))
                )
                if kind is EventKind.assistant_msg and (end := _time(detail, start_key + 1)):
                    final = content, end
    return final


def read_timing(db: sqlite3.Connection, checkpoint: Fields, diagnostics: list[dict]) -> CheckpointTiming:
    """Read turn timestamps by request ID and tool durations by the actual call ID."""
    timing = CheckpointTiming()
    for turn_id in blob_ids(checkpoint, 8):
        blob = read_blob(db, turn_id)
        if blob is None:
            diagnostics.append({"status": "missing_cursor_turn", "item_id": turn_id})
            continue
        try:
            detail = next(bytes_fields(protobuf(blob), 1), None)
            if detail is None:
                continue
            turn = protobuf(detail)
            request_id = text_field(turn, 3)
            if not request_id:
                continue
            for ref in blob_ids(turn, 1):
                if (raw := read_blob(db, ref)) is not None and (stamp := _time(protobuf(raw), 25)):
                    timing.prompts[request_id] = stamp
            latest = None
            for ref in blob_ids(turn, 2):
                raw = read_blob(db, ref)
                latest = _step_timing(timing, request_id, protobuf(raw)) if raw is not None else None
            if latest is not None:
                timing.final_replies[request_id] = latest
        except ValueError:
            diagnostics.append({"status": "unreadable_cursor_turn", "item_id": turn_id})
    return timing
