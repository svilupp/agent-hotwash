"""Cross-process token bucket."""

from __future__ import annotations

from typing import Any

from agent_hotwash.semantic.ratelimit import RateLimiter

_SHARED: dict[str, Any] = {}


def test_rate_limiter_token_bucket_waits_for_refill() -> None:
    now = [0.0]
    slept: list[float] = []
    lim = RateLimiter(rate=2.0, burst=2, clock=lambda: now[0])

    def fake_sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    lim._sleep = fake_sleep
    assert lim.acquire() == 0.0
    assert lim.acquire() == 0.0
    waited = lim.acquire()
    assert abs(waited - 0.5) < 1e-9
    assert slept == [0.5]


def test_rate_limiter_disabled_and_share() -> None:
    assert RateLimiter(0).acquire() == 0.0
    shared = RateLimiter(8.0, burst=8).share(4)
    assert shared.rate == 2.0
    assert shared.burst == 2
    assert RateLimiter(1.0, burst=1).share(10).burst == 1


def test_shared_bucket_is_one_budget_for_many_limiters() -> None:
    """Two limiter objects over one shared bucket behave as a single bucket."""
    now = [0.0]
    a = RateLimiter.shared(rate=2.0, burst=2)
    a._clock = lambda: now[0]
    a._bucket.last = 0.0
    b = RateLimiter(rate=2.0, burst=2, clock=lambda: now[0])
    b._bucket = a._bucket
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    a._sleep = b._sleep = sleep
    assert a.acquire() == 0.0
    assert b.acquire() == 0.0
    assert abs(b.acquire() - 0.5) < 1e-9
    assert slept == [0.5]


def _init_shared(limiter: RateLimiter, barrier: Any) -> None:
    _SHARED["limiter"] = limiter
    _SHARED["barrier"] = barrier


def _acquire_after_barrier(_i: int) -> float:
    _SHARED["barrier"].wait(timeout=30)
    return float(_SHARED["limiter"].acquire())


def test_shared_rate_limiter_bounds_burst_across_processes() -> None:
    """Four workers hitting one ``RateLimiter.shared`` at the same instant get
    exactly ``burst`` free tokens between them; the rest wait for refill.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    limiter = RateLimiter.shared(rate=4.0, burst=2, ctx=ctx)
    barrier = ctx.Barrier(4)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=ctx, initializer=_init_shared, initargs=(limiter, barrier)
    ) as pool:
        waits = sorted(pool.map(_acquire_after_barrier, range(4)))
    assert waits[0] == 0.0 and waits[1] == 0.0
    assert waits[2] > 0.1 and waits[3] > 0.1
    assert sum(waits) > 0.5
    assert limiter._bucket.tokens < 1.0
