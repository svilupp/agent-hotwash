"""System One asker: redaction, ≤N batching, cache + rate-limit transport stack."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx2
from systemoneprompts.answers import choice_label, noul_value, partition_answers, score_value, wire_questions
from systemoneprompts.cache import (
    CacheMissError,
    CacheMode,
    CacheStats,
    CachingFetch,
    FetchResponse,
    create_caching_fetch,
    question_hash,
    transport_headers,
)
from systemoneprompts.client import TypeSafeClient, TypeSafeClientError
from systemoneprompts.diagnostics import SystemOnePromptsError, diagnostic
from systemoneprompts.provider import wrap_caching_fetch

from agent_hotwash.semantic.bank import FeatureDef
from agent_hotwash.semantic.project import project_for_questions
from agent_hotwash.semantic.ratelimit import RateLimitedTransport, RateLimiter
from agent_hotwash.semantic.redact import REDACTION_VERSION, redact_state
from agent_hotwash.semantic.results import FeatureValue

_TYPESAFE_ENV = ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")


def _typesafe_environ() -> dict[str, str | None]:
    """Forward only Typesafe keys so a host Cloudflare account cannot hijack the client."""
    return {key: os.environ.get(key) for key in _TYPESAFE_ENV}


def _unwrap_cache_miss(exc: BaseException) -> CacheMissError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, CacheMissError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    if str(exc).startswith("cache miss in read-only mode"):
        return CacheMissError([])
    return None


class SystemOneAsker:
    """Capability-blind HTTP asker. Annotator owns gating; this owns the wire."""

    def __init__(
        self,
        model: str,
        cache_dir: str | Path,
        *,
        mode: str,
        max_questions: int = 15,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        limiter: RateLimiter | None = None,
        transport_override: httpx2.BaseTransport | None = None,
        secret_patterns: list[str] | None = None,
    ) -> None:
        if mode not in ("cached", "live"):
            raise ValueError(f"unsupported System One mode {mode!r}")
        self.model = model
        self.cache_dir = Path(cache_dir).expanduser() / f"r{REDACTION_VERSION}"
        self.mode = mode
        self.max_questions = max(1, max_questions)
        self.max_retries = max(0, max_retries)
        self.timeout_s = timeout_s
        self.limiter = limiter
        self._secrets = list(secret_patterns or [])
        override = transport_override is not None
        env_key = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
        if mode == "live" and not override and not env_key:
            raise TypeSafeClientError(
                diagnostic(
                    "error",
                    "missing-credentials",
                    "TYPESAFE_API_KEY is not set",
                    hint="export TYPESAFE_API_KEY or place it in a local .env for live commands only",
                )
            )
        inner = transport_override if override else httpx2.HTTPTransport()
        self._rate = RateLimitedTransport(inner, limiter)
        cache_mode: CacheMode = "read-only" if mode == "cached" else "read-write"
        self._caching, caching_transport = _caching_transport(self.cache_dir, self._rate, cache_mode)
        self._client = TypeSafeClient(
            api_key="test" if override else (env_key or None),
            model=model,
            timeout=timeout_s,
            max_retries=max_retries,
            transport=caching_transport,
            environ=_typesafe_environ(),
        )
        self._lock = threading.Lock()
        self._stats = {"questions_asked": 0.0, "cache_hits": 0.0, "retries": 0.0}

    def close(self) -> None:
        self._client.close()

    def stats(self) -> dict[str, float]:
        with self._lock:
            asked = self._stats["questions_asked"]
            hits = self._stats["cache_hits"]
            retries = self._stats["retries"]
        return {
            "requests": float(self._rate.stats["requests"]),
            "questions_asked": float(asked),
            "cache_hits": float(hits),
            "retries": float(retries),
            "rate_wait_s": float(self._rate.stats["rate_wait_s"]),
        }

    def ask(
        self,
        state: dict[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        redact: bool = True,
        allow_unredacted: bool = False,
    ) -> dict[str, Any]:
        """Answer ``questions`` against ``state``. Batch size ≤ ``max_questions``."""
        if self.mode == "live" and not redact and not allow_unredacted:
            raise ValueError("live mode refuses unredacted state unless allow_unredacted is set")
        if not questions:
            return {}
        payload = redact_state(state, self._secrets) if redact else state
        wired = wire_questions(questions)
        out: dict[str, Any] = {}
        ids = list(wired)
        for offset in range(0, len(ids), self.max_questions):
            batch_ids = ids[offset : offset + self.max_questions]
            batch = {qid: wired[qid] for qid in batch_ids}
            batch_state = project_for_questions(payload, batch)
            cache_before = self._caching.stats()
            limiter_before = self._rate.stats["requests"]
            try:
                resp = self._client.system_one_sync(state=batch_state, questions=batch)
            except Exception as exc:
                self._record_chunk(len(batch_ids), cache_before, limiter_before)
                miss = _unwrap_cache_miss(exc)
                if miss is not None:
                    raise miss from exc
                raise
            self._record_chunk(len(batch_ids), cache_before, limiter_before)
            answers = resp.get("answers") or {}
            partitioned = partition_answers(batch, answers)
            out.update(partitioned["valid"])
            missing = list(partitioned["missing"])
            malformed = list(partitioned["malformed"])
            if missing or malformed:
                bits = []
                if missing:
                    bits.append("missing=" + ",".join(sorted(missing)))
                if malformed:
                    bits.append("malformed=" + ",".join(sorted(malformed)))
                raise SystemOnePromptsError(
                    diagnostic("error", "answers-incomplete", "incomplete answers: " + "; ".join(bits))
                )
        return out

    def to_feature_value(
        self,
        feat: FeatureDef,
        answer: Any,
        state: dict[str, Any],
        *,
        redact: bool = True,
    ) -> FeatureValue:
        hashed_state = redact_state(state, self._secrets) if redact else state
        native = wire_questions({feat.id: feat.question}).get(feat.id) or {}
        hashed_state = project_for_questions(hashed_state, {feat.id: native})
        qh = question_hash({"model": self.model, "id": feat.id, "state": hashed_state, "question": native})
        value: Any = noul_value(answer)
        if value is None:
            value = choice_label(answer)
        if value is None:
            value = score_value(answer)
        confidence = None
        if isinstance(answer, dict) and answer.get("confidence") is not None:
            try:
                confidence = float(answer["confidence"])
            except (TypeError, ValueError):
                confidence = None
        stored = dict(answer) if isinstance(answer, dict) else None
        return FeatureValue(
            id=feat.id,
            version=feat.version,
            value=value,
            confidence=confidence,
            source="jev",
            model=self.model,
            question_hash=qh,
            answer=stored,
        )

    def _record_chunk(self, n_questions: int, cache_before: CacheStats, limiter_before: float) -> None:
        cache_after = self._caching.stats()
        limiter_after = self._rate.stats["requests"]
        req_delta = int(limiter_after - limiter_before)
        hits_delta = int(cache_after.hits - cache_before.hits)
        with self._lock:
            if req_delta <= 0:
                self._stats["cache_hits"] += float(min(max(hits_delta, 0), n_questions))
            else:
                first_hits = min(max(hits_delta // req_delta, 0), n_questions)
                self._stats["cache_hits"] += float(first_hits)
                self._stats["questions_asked"] += float(n_questions - first_hits)
                self._stats["retries"] += float(req_delta - 1)


def _caching_transport(
    directory: Path, inner: httpx2.BaseTransport, mode: CacheMode
) -> tuple[CachingFetch, httpx2.BaseTransport]:
    directory.mkdir(parents=True, exist_ok=True)

    def http_fetch(url: str, init: Mapping[str, Any] | None = None) -> FetchResponse:
        options = dict(init or {})
        body = options.get("body")
        content = body.encode("utf-8") if isinstance(body, str) else body
        request = httpx2.Request(
            str(options.get("method") or "GET"),
            url,
            headers=transport_headers(options),
            content=content,
        )
        response = inner.handle_request(request)
        response.read()
        return FetchResponse(
            status=int(response.status_code),
            status_text=getattr(response, "reason_phrase", "OK") or "OK",
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.content,
        )

    caching = create_caching_fetch(dir=str(directory), mode=mode, fetch=http_fetch)
    return caching, wrap_caching_fetch(caching, inner=inner)


__all__ = ["CacheMissError", "SystemOneAsker", "TypeSafeClientError"]
