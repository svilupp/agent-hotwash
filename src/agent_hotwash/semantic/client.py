"""System One asker: redaction, ≤N batching, cache + rate-limit transport stack."""

from __future__ import annotations

import hashlib
import json
import math
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
from systemoneprompts.client import TypeSafeClient, TypeSafeClientError, TypeSafeHttpError
from systemoneprompts.diagnostics import SystemOnePromptsError, diagnostic
from systemoneprompts.json_values import canonical_json
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


class StrictSemanticResponseError(ValueError):
    """Reject information the tolerant per-question cache would otherwise filter."""


class SemanticRequestTooLarge(ValueError):
    """A diagnostic singleton exceeds its explicit serialized request budget."""


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StrictSemanticResponseError("Duplicate System One response object key")
        value[key] = item
    return value


def _strict_nouls(questions, answers):
    for qid, question in questions.items():
        if question.get("type") != "noul":
            continue
        answer = answers.get(qid)
        value = answer.get("noul") if isinstance(answer, dict) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise StrictSemanticResponseError("Invalid finite numeric System One Noul probability")


class SystemOneAsker:
    """Capability-blind HTTP asker. Annotator owns gating; this owns the wire."""

    def __init__(
        self,
        model: str,
        cache_dir: str | Path,
        *,
        mode: str,
        max_questions: int | None = None,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        limiter: RateLimiter | None = None,
        transport_override: httpx2.BaseTransport | None = None,
        secret_patterns: list[str] | None = None,
        strict_answers: bool = False,
        max_request_bytes: int | None = None,
    ) -> None:
        if mode not in ("cached", "live"):
            raise ValueError(f"unsupported System One mode {mode!r}")
        self.max_request_bytes = max_request_bytes
        self.strict_answers = strict_answers
        self.model = model
        self.cache_dir = Path(cache_dir).expanduser() / f"r{REDACTION_VERSION}"
        if strict_answers:
            self.cache_dir /= "strict-v1"
        self.mode = mode
        self.max_questions = max_questions if max_questions is not None and max_questions > 0 else None
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
        self._caching, caching_transport = _caching_transport(
            self.cache_dir, self._rate, cache_mode, strict_answers=strict_answers
        )
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
        self._provider_usage = []

    def close(self) -> None:
        self._client.close()

    def usage(self):
        with self._lock:
            return list(self._provider_usage)

    def stats(self) -> dict[str, float]:
        with self._lock:
            asked = self._stats["questions_asked"]
            hits = self._stats["cache_hits"]
            retries = self._stats["retries"]
        return {
            "requests": float(self._rate.stats["requests"]),
            "questions_asked": float(asked),
            "cache_hits": float(hits),
            "cache_misses": float(asked),
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
        max_request_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Answer questions with the fewest requests.

        With no positive ``max_questions`` cap, every question sharing this
        state is sent together. A provider HTTP 413 recursively bisects only
        the rejected batch.
        """
        if self.mode == "live" and not redact and not allow_unredacted:
            raise ValueError("live mode refuses unredacted state unless allow_unredacted is set")
        if not questions:
            return {}
        payload = redact_state(state, self._secrets) if redact else state
        wired = wire_questions(questions)
        out: dict[str, Any] = {}
        ids = list(wired)
        batch_size = self.max_questions or len(ids)

        def run_batch(batch_ids: list[str]) -> None:
            batch = {qid: wired[qid] for qid in batch_ids}
            batch_state = project_for_questions(payload, batch)
            envelope = {"model": self.model, "state": batch_state, "questions": batch}
            limits = [limit for limit in (self.max_request_bytes, max_request_bytes) if limit is not None]
            if limits and len(canonical_json(envelope).encode("utf-8")) > min(limits):
                if max_request_bytes is not None:
                    if len(batch_ids) > 1:
                        midpoint = len(batch_ids) // 2
                        run_batch(batch_ids[:midpoint])
                        run_batch(batch_ids[midpoint:])
                        return
                    raise SemanticRequestTooLarge("Diagnostic singleton exceeds serialized request byte limit")
                raise ValueError("System One request exceeds configured serialized byte limit")
            if self.strict_answers:
                self._validate_native_cache(batch_state, batch)
            cache_before = self._caching.stats()
            self._rate.begin_chunk()
            try:
                resp = self._client.system_one_sync(state=batch_state, questions=batch)
            except Exception as exc:
                self._record_chunk(len(batch_ids), cache_before)
                current = exc
                seen: set[int] = set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    if isinstance(current, StrictSemanticResponseError):
                        raise current from exc
                    current = current.__cause__ or current.__context__
                if isinstance(exc, TypeSafeHttpError) and exc.status == 413 and len(batch_ids) > 1:
                    midpoint = len(batch_ids) // 2
                    run_batch(batch_ids[:midpoint])
                    run_batch(batch_ids[midpoint:])
                    return
                miss = _unwrap_cache_miss(exc)
                if miss is not None:
                    raise miss from exc
                raise
            self._record_chunk(len(batch_ids), cache_before)
            if isinstance(resp.get("usage"), dict):
                with self._lock:
                    self._provider_usage.append(resp["usage"])
            answers = resp.get("answers") or {}
            if self.strict_answers and (not isinstance(answers, dict) or set(answers) != set(batch)):
                raise ValueError("Incomplete or unexpected System One answer IDs")
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

        for offset in range(0, len(ids), batch_size):
            run_batch(ids[offset : offset + batch_size])
        return out

    def _validate_native_cache(self, state, questions):
        for qid, question in questions.items():
            key = question_hash({"model": self.model, "id": qid, "state": state, "question": question})
            path = self.cache_dir / key[:2] / (key + ".json")
            if not path.exists():
                continue
            try:
                saved = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError, OSError):
                raise StrictSemanticResponseError(
                    "Corrupt native semantic cache; preserve the object for inspection"
                ) from None
            if (
                not isinstance(saved, dict)
                or saved.get("hash") != key
                or not isinstance(saved.get("reportedModel"), str)
            ):
                raise StrictSemanticResponseError("Invalid native semantic cache contract")
            _strict_nouls({qid: question}, {qid: saved.get("answer")})
            parsed = partition_answers({qid: question}, {qid: saved.get("answer")})
            if qid not in parsed["valid"]:
                raise StrictSemanticResponseError("Invalid native semantic cache answer")

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
        state_hash = hashlib.sha256(canonical_json(hashed_state).encode("utf-8")).hexdigest()
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
        reason = None
        if (
            feat.primitive == "noul"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and feat.negative_threshold < float(value) < feat.positive_threshold
        ):
            reason = "low_support"
        return FeatureValue(
            id=feat.id,
            version=feat.version,
            value=value,
            confidence=confidence,
            reason=reason,
            source="jev",
            model=self.model,
            question_hash=qh,
            answer=stored,
            state_hash=state_hash,
            negative_threshold=feat.negative_threshold if feat.primitive == "noul" else None,
            positive_threshold=feat.positive_threshold if feat.primitive == "noul" else None,
        )

    def _record_chunk(self, n_questions: int, cache_before: CacheStats) -> None:
        requests, first_questions = self._rate.chunk_stats()
        with self._lock:
            if requests <= 0:
                hits = self._caching.stats().hits - cache_before.hits
                self._stats["cache_hits"] += float(min(max(hits, 0), n_questions))
            else:
                missed = n_questions if first_questions is None else min(first_questions, n_questions)
                self._stats["cache_hits"] += float(n_questions - missed)
                self._stats["questions_asked"] += float(missed)
                self._stats["retries"] += float(requests - 1)


def _caching_transport(
    directory: Path, inner: httpx2.BaseTransport, mode: CacheMode, *, strict_answers: bool = False
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
        if strict_answers and 200 <= response.status_code < 300:
            try:
                payload = json.loads(response.content.decode("utf-8"), object_pairs_hook=_unique_object)
                if not isinstance(content, (str, bytes, bytearray)):
                    raise StrictSemanticResponseError("Invalid System One request envelope")
                sent = json.loads(content)
                answers = payload.get("answers") if isinstance(payload, dict) else None
                if not isinstance(answers, dict) or set(answers) != set(sent["questions"]):
                    raise StrictSemanticResponseError("Incomplete or unexpected System One answer IDs")
                _strict_nouls(sent["questions"], answers)
                validated = partition_answers(sent["questions"], answers)
                if validated["missing"] or validated["malformed"]:
                    raise StrictSemanticResponseError("Malformed System One answers")
            except (json.JSONDecodeError, UnicodeError):
                raise StrictSemanticResponseError("Malformed System One response JSON") from None
        return FetchResponse(
            status=int(response.status_code),
            status_text=getattr(response, "reason_phrase", "OK") or "OK",
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.content,
        )

    caching = create_caching_fetch(dir=str(directory), mode=mode, fetch=http_fetch)
    return caching, wrap_caching_fetch(caching, inner=inner)


__all__ = ["CacheMissError", "SystemOneAsker", "TypeSafeClientError"]
