"""JeV HTTP client: stdlib urllib, per-question cache, batching, rate limit.

Throughput model (§6.6, C8):

- Questions are batched ≤ ``max_questions`` per request and answered from the
  per-question cache first; only misses hit the network.
- Every network request first takes a token from a :class:`RateLimiter`
  (token bucket, thread-safe). Callers running several worker processes give
  each process its share of the global budget (see ``runner``).
- 429/5xx are retried with exponential backoff (``Retry-After`` honoured when
  the server sends it); 401 and malformed answers fail fast.
"""

from __future__ import annotations

import contextlib
import email.utils
import hashlib
import json
import multiprocessing
import os
import random
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_hotwash.semantic.redact import REDACTION_VERSION
from agent_hotwash.structure.digest import DIGEST_SCHEMA_VERSION as DIGEST_SCHEMA_VERSION_DEFAULT

JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MAX_RETRIES = 3
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 30.0  # cap for our own exponential backoff
_RETRY_AFTER_CAP_S = 300.0  # cap for a server-supplied Retry-After
Transport = Callable[[dict[str, Any]], dict[str, Any]]


class _LocalBucket:
    """Token-bucket state for one process (threads share it under ``lock``)."""

    __slots__ = ("last", "lock", "tokens")

    def __init__(self, tokens: float, last: float) -> None:
        self.tokens = tokens
        self.last = last
        self.lock: Any = threading.Lock()


class _SharedBucket:
    """Token-bucket state in shared memory: one budget for every worker process.

    Built in the parent and handed to workers through the pool initializer
    (``multiprocessing`` primitives may only cross process boundaries by
    inheritance, which is exactly what ``initargs`` does). The lock is a
    ``SemLock`` so it also serialises threads inside one worker.
    """

    def __init__(self, tokens: float, last: float, ctx: Any = None) -> None:
        ctx = ctx or multiprocessing.get_context()
        self._tokens = ctx.RawValue("d", tokens)
        self._last = ctx.RawValue("d", last)
        self.lock = ctx.Lock()

    @property
    def tokens(self) -> float:
        return float(self._tokens.value)

    @tokens.setter
    def tokens(self, value: float) -> None:
        self._tokens.value = value

    @property
    def last(self) -> float:
        return float(self._last.value)

    @last.setter
    def last(self, value: float) -> None:
        self._last.value = value


class RateLimiter:
    """Token bucket: at most ``rate`` requests/second on average, with up to
    ``burst`` requests allowed back-to-back.

    Thread-safe within a process; :meth:`shared` builds one whose state lives
    in shared memory so *all* worker processes draw from a single budget
    (the configured ``burst`` is then a true global ceiling). ``rate <= 0``
    disables limiting. ``acquire`` blocks (sleeping) until a token is
    available and returns the seconds it waited.
    """

    def __init__(self, rate: float, burst: int | None = None, *, clock: Callable[[], float] | None = None) -> None:
        self.rate = max(0.0, float(rate))
        self.burst = max(1, int(burst if burst is not None else max(1, round(self.rate))))
        self._clock = clock or time.monotonic
        self._sleep: Callable[[float], None] = time.sleep
        self._bucket: _LocalBucket | _SharedBucket = _LocalBucket(float(self.burst), self._clock())

    @classmethod
    def shared(cls, rate: float, burst: int | None = None, *, ctx: Any = None) -> RateLimiter:
        """A limiter whose bucket is shared across processes spawned with ``ctx``.

        Uses ``time.monotonic`` (system-wide on macOS and Linux) so refill
        maths agree between processes.
        """
        limiter = cls(rate, burst)
        limiter._bucket = _SharedBucket(float(limiter.burst), limiter._clock(), ctx)
        return limiter

    def _refill(self) -> None:
        now = self._clock()
        bucket = self._bucket
        bucket.tokens = min(float(self.burst), bucket.tokens + (now - bucket.last) * self.rate)
        bucket.last = now

    def acquire(self) -> float:
        if self.rate <= 0:
            return 0.0
        waited = 0.0
        while True:
            with self._bucket.lock:
                self._refill()
                if self._bucket.tokens >= 1.0:
                    self._bucket.tokens -= 1.0
                    return waited
                wait = (1.0 - self._bucket.tokens) / self.rate
            self._sleep(wait)
            waited += wait

    def share(self, parts: int) -> RateLimiter:
        """A private limiter for one of ``parts`` consumers of this budget.

        Fallback for callers that cannot pass a :meth:`shared` bucket to their
        workers. The aggregate *rate* is exact; the aggregate *burst* becomes
        ``max(burst, parts)`` because each consumer needs at least one token.
        """
        parts = max(1, parts)
        return RateLimiter(self.rate / parts, max(1, self.burst // parts), clock=self._clock)


class JeVError(Exception):
    """Exit-style JeV failure. ``code`` ∈ {401, 429, 500, cache_miss, no_key}."""

    def __init__(self, code: int | str, message: str = "", retry_after: float | None = None) -> None:
        self.code = code
        self.retry_after = retry_after
        super().__init__(message or str(code))


class CacheMiss(JeVError):
    """``cached`` mode found no answer for ``missing`` question ids."""

    def __init__(self, message: str = "cache miss", missing: list[str] | None = None) -> None:
        super().__init__("cache_miss", message)
        self.missing = list(missing or [])


def _nfc(obj: Any) -> Any:
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, dict):
        return {_nfc(k): _nfc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nfc(x) for x in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, NFC-normalised strings."""
    return json.dumps(_nfc(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(
    model_version: str,
    feature_id: str,
    feature_version: int,
    question_def: Any,
    digest_schema_version: int,
    redaction_version: int,
    state: Any,
) -> str:
    """C8: sha256 of canonical JSON of the cache-identity tuple."""
    payload = [
        model_version,
        feature_id,
        feature_version,
        _sha(canonical_json(question_def)),
        digest_schema_version,
        redaction_version,
        _sha(canonical_json(state)),
    ]
    return _sha(canonical_json(payload))


def _retry_after(exc: urllib.error.HTTPError, *, now: Callable[[], float] = time.time) -> float | None:
    """Seconds from a ``Retry-After`` header: delta-seconds or an HTTP-date."""
    raw = exc.headers.get("Retry-After") if exc.headers is not None else None
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - now())


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    """Server hint when given (capped at ``_RETRY_AFTER_CAP_S``), else
    exponential backoff with jitter capped at ``_BACKOFF_CAP_S``."""
    if retry_after is not None:
        return min(_RETRY_AFTER_CAP_S, retry_after)
    base = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
    return base * (0.5 + random.random() / 2)  # jitter


def _normalize_http_code(code: int) -> int | str:
    if code == 401:
        return 401
    if code == 429:
        return 429
    if code >= 500:
        return 500
    return code


def _as_unit_float(val: Any) -> float | None:
    """``val`` as a float in ``[0, 1]``; ``None`` when it is not one."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    num = float(val)
    if num != num or num < 0.0 or num > 1.0:  # NaN or out of range
        return None
    return num


def _allowed_values(qdef: dict[str, Any] | None) -> set[str] | None:
    """The option/level labels a ``choice``/``score`` answer must come from."""
    if not qdef:
        return None
    for key in ("options", "levels", "choices"):
        raw = qdef.get(key)
        if isinstance(raw, list) and raw:
            return {str(o.get("id") if isinstance(o, dict) else o) for o in raw}
    return None


def _validate_answer(qtype: str, raw: Any, qdef: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Typed answer payload for one primitive, or ``None`` when malformed.

    ``noul`` → ``{"noul": p}`` with ``p`` a number in [0, 1]; ``choice`` →
    ``{"choice": str, confidence?, probabilities?}``; ``score`` →
    ``{"score": str|int, confidence?, probabilities?}``. When ``qdef`` lists
    ``options``/``levels`` the value must be one of them. Bare scalars are
    accepted for compatibility. Anything else (missing key, wrong type,
    ``None``) is malformed and must not be cached.
    """
    if qtype == "noul":
        val = raw.get("noul") if isinstance(raw, dict) else raw
        num = _as_unit_float(val)
        return None if num is None else {"noul": num}
    if qtype in ("choice", "score"):
        if isinstance(raw, dict):
            val = raw.get(qtype)
            confidence = raw.get("confidence")
            probabilities = raw.get("probabilities")
        else:
            val, confidence, probabilities = raw, None, None
        if qtype == "choice" and not (isinstance(val, str) and val.strip()):
            return None
        if qtype == "score" and (
            isinstance(val, bool) or not ((isinstance(val, str) and val.strip()) or isinstance(val, int))
        ):
            return None
        allowed = _allowed_values(qdef)
        # An integer score may index the ordered levels.
        indexes_level = qtype == "score" and isinstance(val, int) and allowed is not None and 0 <= val < len(allowed)
        if allowed is not None and str(val) not in allowed and not indexes_level:
            return None
        if confidence is not None and _as_unit_float(confidence) is None:
            return None
        if probabilities is not None and not isinstance(probabilities, dict):
            return None
        return {qtype: val, "confidence": confidence, "probabilities": probabilities}
    # Unknown primitive: accept an object payload verbatim, reject bare None.
    if isinstance(raw, dict):
        return dict(raw)
    return None if raw is None else {"value": raw}


class JeVClient:
    """POST ``JEV_URL`` with optional injectable ``transport`` for tests."""

    def __init__(
        self,
        model: str,
        cache_dir: str | Path,
        max_questions: int = 15,
        transport: Transport | None = None,
        *,
        limiter: RateLimiter | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir).expanduser()
        self.max_questions = max(1, max_questions)
        self.transport = transport
        self.limiter = limiter
        self.max_retries = max(0, max_retries)
        self.timeout_s = timeout_s
        self._sleep = time.sleep
        # Counters for progress/telemetry (per client; a client is per process,
        # shared by the annotator's threads — hence the lock).
        self.stats = {"requests": 0, "questions_asked": 0, "cache_hits": 0, "retries": 0, "rate_wait_s": 0.0}
        self._stats_lock = threading.Lock()

    def _count(self, key: str, amount: float = 1) -> None:
        with self._stats_lock:
            self.stats[key] += amount

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        fd, tmp = tempfile.mkstemp(prefix="jev-", suffix=".json", dir=str(self.cache_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _http(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(body)
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise JeVError("no_key", "TYPESAFE_API_KEY is not set")
        data = canonical_json(body).encode("utf-8")
        req = urllib.request.Request(
            JEV_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            code = _normalize_http_code(int(exc.code))
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                detail = str(exc)
            raise JeVError(code, detail or str(exc), retry_after=_retry_after(exc)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JeVError(500, str(getattr(exc, "reason", exc))) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JeVError(500, "malformed JeV response") from exc
        if not isinstance(parsed, dict):
            raise JeVError(500, "JeV response is not an object")
        return parsed

    def _post_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST once per rate-limit token; retry 429/5xx with capped backoff."""
        attempts = 1 + self.max_retries
        for attempt in range(attempts):
            if self.limiter is not None:
                self._count("rate_wait_s", self.limiter.acquire())
            self._count("requests")
            try:
                return self._http(body)
            except JeVError as exc:
                retryable = exc.code == 429 or (isinstance(exc.code, int) and exc.code >= 500)
                if not retryable or attempt >= attempts - 1:
                    if isinstance(exc.code, int) and exc.code >= 500:
                        raise JeVError(500, str(exc)) from exc
                    raise
                self._count("retries")
                self._sleep(_backoff_seconds(attempt, exc.retry_after))
        raise AssertionError("unreachable")  # pragma: no cover

    def ask(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        *,
        mode: str,
        redact: bool = True,
        allow_unredacted: bool = False,
        digest_schema_version: int = DIGEST_SCHEMA_VERSION_DEFAULT,
        redaction_version: int = REDACTION_VERSION,
        feature_versions: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Answer ``questions`` against ``state``. Batch size ≤ ``max_questions``.

        ``mode=cached`` never touches transport (raises ``CacheMiss`` on a miss).
        ``mode=live`` refuses if ``redact`` is off unless ``allow_unredacted``.
        """
        if mode not in ("cached", "live"):
            raise ValueError(f"unsupported JeV mode {mode!r}")
        if mode == "live" and not redact and not allow_unredacted:
            raise ValueError("live mode refuses unredacted state unless allow_unredacted is set")
        if not questions:
            return {}
        versions = feature_versions or {}
        keys: dict[str, str] = {}
        cached: dict[str, Any] = {}
        missing: list[str] = []
        for qid, qdef in questions.items():
            key = cache_key(
                self.model,
                qid,
                versions.get(qid, int(qdef.get("version") or 1)),
                qdef,
                digest_schema_version,
                redaction_version,
                state,
            )
            keys[qid] = key
            hit = self._read_cache(key)
            if hit is not None:
                cached[qid] = hit
            else:
                missing.append(qid)
        self._count("cache_hits", len(cached))

        if mode == "cached":
            if missing:
                raise CacheMiss(f"missing {len(missing)} cached question(s)", missing=missing)
            return cached

        if missing and self.transport is None and not os.environ.get("TYPESAFE_API_KEY"):
            raise JeVError("no_key", "TYPESAFE_API_KEY is not set")

        for offset in range(0, len(missing), self.max_questions):
            batch_ids = missing[offset : offset + self.max_questions]
            batch_qs = {qid: questions[qid] for qid in batch_ids}
            body = {"model": self.model, "state": state, "questions": batch_qs}
            self._count("questions_asked", len(batch_ids))
            resp = self._post_with_retry(body)
            answers = resp.get("answers", resp)
            if not isinstance(answers, dict):
                raise JeVError(500, "JeV answers are not an object")
            validated: dict[str, dict[str, Any]] = {}
            missing_ids: list[str] = []
            malformed_ids: list[str] = []
            for qid in batch_ids:
                if qid not in answers:
                    missing_ids.append(qid)
                    continue
                qtype = str(questions[qid].get("type") or questions[qid].get("primitive") or "")
                parsed = _validate_answer(qtype, answers[qid], questions[qid])
                if parsed is None:
                    malformed_ids.append(qid)
                    continue
                validated[qid] = parsed
            # Well-formed answers are cached even when siblings are bad (a later
            # run re-asks only the misses); bad ones are never cached and the
            # batch is reported as a transport failure so callers degrade to
            # ``api_error`` instead of reading a null as an answer.
            for qid, parsed in validated.items():
                cached[qid] = parsed
                self._write_cache(keys[qid], parsed)
            if missing_ids or malformed_ids:
                bits = []
                if missing_ids:
                    bits.append(f"missing={sorted(missing_ids)}")
                if malformed_ids:
                    bits.append(f"malformed={sorted(malformed_ids)}")
                raise JeVError(500, "missing/malformed answers: " + ", ".join(bits))
        return cached


def ask(
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    *,
    mode: str,
    model: str,
    cache_dir: str | Path,
    max_questions: int = 15,
    transport: Transport | None = None,
    redact: bool = True,
    allow_unredacted: bool = False,
) -> dict[str, Any]:
    """Module-level JeV entry: construct a client and ``ask``."""
    client = JeVClient(model, cache_dir, max_questions=max_questions, transport=transport)
    return client.ask(state, questions, mode=mode, redact=redact, allow_unredacted=allow_unredacted)


__all__ = [
    "JEV_URL",
    "CacheMiss",
    "JeVClient",
    "JeVError",
    "RateLimiter",
    "ask",
    "cache_key",
    "canonical_json",
]
