"""Redact a digest after build and before send (C9)."""

from __future__ import annotations

import re
from typing import Any

REDACTION_VERSION = 1

_SENTINELS = ("item_completed", "Script completed", "token_usage_record")

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}(?!\d)")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_USERINFO_RE = re.compile(r"^(https?://)([^/@\s]+)@")
_USERNAME_ASSIGN_RE = re.compile(r"(?i)\b(user(?:name)?)\s*[:=]\s*\S+")
_HOME_TILDE_RE = re.compile(r"~[A-Za-z_][A-Za-z0-9_-]{0,31}")

_TOKENISH_QUERY_KEY = re.compile(r"(?i)token|key|secret|sig|auth|password|passwd|access")
_LONG_TOKEN_VAL = re.compile(r"^[A-Za-z0-9_\-]{16,}$")


def _stash(text: str, reserved: list[str]) -> tuple[str, list[str]]:
    held: list[str] = []
    out = text
    for item in reserved:
        if not item:
            continue
        token = f"\x00R{len(held)}\x00"
        if item in out:
            out = out.replace(item, token)
            held.append(item)
    return out, held


def _unstash(text: str, held: list[str]) -> str:
    out = text
    for i, item in enumerate(held):
        out = out.replace(f"\x00R{i}\x00", item)
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
    url = re.sub(r"^(https?://(?:redacted@)?)[^/:]+", r"\1host.example", url)
    return url


def _redact_paths(text: str) -> str:
    text = re.sub(r"/Users/[^/]+", "/home/user", text)
    text = re.sub(r"/Volumes/[^/]+", "/home/user", text)
    text = re.sub(r"/home/(?!user(?:/|$))[^/]+", "/home/user", text)
    return text


def _redact_text(text: str, secret_res: list[re.Pattern[str]], allowlist: list[str]) -> str:
    reserved = [*_SENTINELS, *allowlist]
    work, held = _stash(text, reserved)
    # URLs first: userinfo looks like an email and would otherwise split the match.
    work = _URL_RE.sub(lambda m: _redact_url(m.group(0)), work)
    work = _EMAIL_RE.sub("[email]", work)
    work = _PHONE_RE.sub("[phone]", work)
    for pat in secret_res:
        work = pat.sub("[secret]", work)
    work = _redact_paths(work)
    work = _USERNAME_ASSIGN_RE.sub(r"\1=user", work)
    work = _HOME_TILDE_RE.sub("~user", work)
    return _unstash(work, held)


def redact_state(
    state: dict[str, Any] | list[Any] | str | Any,
    lexicon_secret_patterns: list[str],
    *,
    allowlist: list[str] | None = None,
) -> Any:
    """Replace secrets/PII in a digest. Recursive on dict/list/str.

    Source-code tokens such as ``test_key`` are not matched. Sentinels
    (``item_completed``, ``Script completed``, ``token_usage_record``) are preserved.
    """
    secret_res = [re.compile(p) for p in lexicon_secret_patterns if p]
    allow = list(allowlist or [])

    def rec(obj: Any) -> Any:
        if isinstance(obj, str):
            return _redact_text(obj, secret_res, allow)
        if isinstance(obj, dict):
            return {rec(k) if isinstance(k, str) else k: rec(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rec(v) for v in obj]
        if isinstance(obj, tuple):
            return [rec(v) for v in obj]
        return obj

    return rec(state)


__all__ = ["REDACTION_VERSION", "redact_state"]
