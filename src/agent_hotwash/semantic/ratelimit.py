"""Cross-process token bucket and an httpx2 transport that spends it."""

from __future__ import annotations

import multiprocessing
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx2


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


class RateLimitedTransport(httpx2.BaseTransport):
    """Acquire one rate-limit token per HTTP attempt, then delegate."""

    def __init__(self, inner: httpx2.BaseTransport, limiter: RateLimiter | None) -> None:
        self._inner = inner
        self.limiter = limiter
        self.stats: dict[str, float] = {"requests": 0.0, "rate_wait_s": 0.0}
        self._lock = threading.Lock()

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        waited = self.limiter.acquire() if self.limiter is not None else 0.0
        with self._lock:
            self.stats["requests"] += 1.0
            self.stats["rate_wait_s"] += waited
        return self._inner.handle_request(request)

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()


__all__ = ["RateLimitedTransport", "RateLimiter"]
