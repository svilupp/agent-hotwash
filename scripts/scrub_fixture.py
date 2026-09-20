#!/usr/bin/env python3
"""Deterministic irreversible fixture scrubber (PLAN §9.3).

Preserve JSON types, record order, ordinals, timestamps, numeric edge values,
id referential integrity, path relationships, and sentinel strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SENTINELS: tuple[str, ...] = (
    "item_completed",
    "UserMessage",
    "Reasoning",
    "AgentMessage",
    "CommandExecution",
    "FileChange",
    "McpToolCall",
    "FunctionCallOutput",
    "SubAgentActivity",
    "Extension",
    "ContextCompaction",
    "CollabAgentToolCall",
    "ImageView",
    "Script completed",
    "Script failed",
    "aborted by user",
    "turn_context",
    "task_started",
    "task_complete",
    "turn_aborted",
    "thread_settings_applied",
    "token_usage_record",
    "custom_tool_call",
    "exec",
    "subagent_history_start_ordinal",
    "parsed_cmd",
    "token_count",
    "response_id",
    "turn_id",
)

# Longest first so overlapping sentinels stash correctly.
_SENTINELS_SORTED = tuple(sorted(SENTINELS, key=len, reverse=True))

_PAYLOAD_KEYS = frozenset(
    {
        "text",
        "output",
        "stdout",
        "stderr",
        "aggregated_output",
        "input",
        "thinking",
        "message",
        "content",
        "summary_text",
        "encrypted_content",
        "base_instructions",
        "world_state",
        "replacement_history",
    }
)

_ID_KEYS = frozenset(
    {
        "id",
        "call_id",
        "turn_id",
        "response_id",
        "thread_id",
        "session_id",
        "parent_thread_id",
        "forked_from_id",
        "agent_thread_id",
        "item_id",
        "group_id",
        "sender_thread_id",
        "source_thread_id",
        "parentId",
        "toolCallId",
        "receiver_thread_id",
    }
)

_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}(?!\d)")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_USERINFO_RE = re.compile(r"^(https?://)([^/@\s]+)@")
_AKIA_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")
_BEARER_RE = re.compile(r"(?i)\b(bearer|token|api[_-]?key)\s+[A-Za-z0-9._\-]{8,}")
_ASSIGN_SECRET_RE = re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+")
_TOKENISH_QUERY_KEY = re.compile(r"(?i)token|key|secret|sig|auth|password|passwd|access")
_LONG_TOKEN_VAL = re.compile(r"^[A-Za-z0-9_\-]{16,}$")
_LOREM = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "


def _stash(text: str) -> tuple[str, list[str]]:
    held: list[str] = []
    out = text
    for item in _SENTINELS_SORTED:
        token = f"\x00S{len(held)}\x00"
        if item in out:
            out = out.replace(item, token)
            held.append(item)
    return out, held


def _unstash(text: str, held: list[str]) -> str:
    out = text
    for i, item in enumerate(held):
        out = out.replace(f"\x00S{i}\x00", item)
    return out


def _redact_url(url: str) -> str:
    url = _USERINFO_RE.sub(r"\1redacted@", url)
    if "?" in url:
        base, query = url.split("?", 1)
        if "#" in query:
            frag, hashpart = query.split("#", 1)
        else:
            frag, hashpart = query, ""
        parts: list[str] = []
        for piece in frag.split("&"):
            if "=" not in piece:
                parts.append(piece)
                continue
            key, val = piece.split("=", 1)
            if _TOKENISH_QUERY_KEY.search(key) or _LONG_TOKEN_VAL.match(val):
                parts.append(f"{key}=REDACTED")
            else:
                parts.append(piece)
        url = base + "?" + "&".join(parts)
        if hashpart:
            url += "#" + hashpart
    return re.sub(r"^(https?://(?:redacted@)?)[^/:]+", r"\1host.example.com", url)


def _redact_paths(text: str) -> str:
    text = re.sub(r"/Users/[^/]+", "/home/user", text)
    text = re.sub(r"/Volumes/[^/]+", "/home/user", text)
    text = re.sub(r"/home/(?!user(?:/|$))[^/]+", "/home/user", text)
    return text


def remap_uuid(value: str, mapping: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        old = match.group(0)
        mapped = mapping.get(old)
        if mapped is None:
            digest = hashlib.sha256(old.encode("utf-8")).hexdigest()
            mapped = f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
            mapping[old] = mapped
        return mapped

    return _UUID_RE.sub(repl, value)


def same_shape_lorem(text: str) -> str:
    """Replace non-sentinel, non-whitespace characters with deterministic lorem."""
    out: list[str] = []
    li = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\x00":
            end = text.find("\x00", i + 1)
            if end < 0:
                out.append(text[i])
                i += 1
                continue
            out.append(text[i : end + 1])
            i = end + 1
            continue
        ch = text[i]
        if ch.isspace() or ch in '{}[]":,./\\|_-':
            out.append(ch)
        else:
            out.append(_LOREM[li % len(_LOREM)])
            li += 1
        i += 1
    return "".join(out)


def scrub_text(text: str, mapping: dict[str, str], *, lorem: bool) -> str:
    work, held = _stash(text)
    work = remap_uuid(work, mapping)
    work = _URL_RE.sub(lambda m: _redact_url(m.group(0)), work)
    work = _EMAIL_RE.sub("user@example.com", work)
    work = _PHONE_RE.sub("555-0100", work)
    work = _AKIA_RE.sub("AKIAEXAMPLEKEY00000", work)
    work = _SK_RE.sub("sk-REDACTED", work)
    work = _BEARER_RE.sub(r"\1 REDACTED", work)
    work = _ASSIGN_SECRET_RE.sub(lambda m: f"{m.group(1)}=REDACTED", work)
    work = _redact_paths(work)
    if lorem:
        work = same_shape_lorem(work)
    return _unstash(work, held)


def scrub_value(obj: Any, mapping: dict[str, str], *, key: str | None = None) -> Any:
    if isinstance(obj, str):
        lorem = key in _PAYLOAD_KEYS and key not in _ID_KEYS
        return scrub_text(obj, mapping, lorem=lorem)
    if isinstance(obj, dict):
        return {k: scrub_value(v, mapping, key=str(k) if isinstance(k, str) else None) for k, v in obj.items()}
    if isinstance(obj, list):
        child_key = key if key in _PAYLOAD_KEYS else None
        return [scrub_value(v, mapping, key=child_key) for v in obj]
    return obj


def scrub_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping: dict[str, str] = {}
    return [scrub_value(rec, mapping) for rec in records]


def scrub_jsonl_text(raw: str) -> str:
    lines = raw.splitlines()
    mapping: dict[str, str] = {}
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        scrubbed = scrub_value(rec, mapping)
        out_lines.append(json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":")))
    trailing = "\n" if raw.endswith("\n") else ""
    return "\n".join(out_lines) + trailing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrub a JSONL fixture of secrets and PII.")
    parser.add_argument("input", type=Path, help="Input JSONL path.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--canary",
        action="append",
        default=[],
        help="Fail if STRING appears in the scrubbed output (repeatable).",
    )
    args = parser.parse_args(argv)
    raw = args.input.read_text(encoding="utf-8")
    text = scrub_jsonl_text(raw)
    hits = [c for c in args.canary if c and c in text]
    if hits:
        sys.stderr.write("canary hit: " + ", ".join(hits) + "\n")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
