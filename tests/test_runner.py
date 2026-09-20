"""Parallel runner, rate limiter and work-unit discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.runner import RunOptions, resolve_jobs, run_paths, run_units
from agent_hotwash.semantic.jev import JeVClient, JeVError, RateLimiter
from agent_hotwash.sources.detect import WorkUnit, discover

FIXTURES = Path(__file__).parent / "fixtures"
TREE = FIXTURES / "codex_native" / "v0153" / "tree"
CLAUDE_RUN = FIXTURES / "codebench" / "claude_run"


# --------------------------------------------------------------------------- rate limiter


def test_rate_limiter_token_bucket_waits_for_refill() -> None:
    now = [0.0]
    slept: list[float] = []
    lim = RateLimiter(rate=2.0, burst=2, clock=lambda: now[0])

    def fake_sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    lim._sleep = fake_sleep
    assert lim.acquire() == 0.0  # burst token 1
    assert lim.acquire() == 0.0  # burst token 2
    waited = lim.acquire()  # bucket empty -> wait 1/rate
    assert abs(waited - 0.5) < 1e-9
    assert slept == [0.5]


def test_rate_limiter_disabled_and_share() -> None:
    assert RateLimiter(0).acquire() == 0.0
    shared = RateLimiter(8.0, burst=8).share(4)
    assert shared.rate == 2.0
    assert shared.burst == 2
    assert RateLimiter(1.0, burst=1).share(10).burst == 1


_SHARED: dict[str, Any] = {}


def _init_shared(limiter: RateLimiter, barrier: Any) -> None:
    _SHARED["limiter"] = limiter
    _SHARED["barrier"] = barrier


def _acquire_after_barrier(_i: int) -> float:
    _SHARED["barrier"].wait(timeout=30)
    return float(_SHARED["limiter"].acquire())


def test_shared_rate_limiter_bounds_burst_across_processes() -> None:
    """Four workers hitting one ``RateLimiter.shared`` at the same instant get
    exactly ``burst`` free tokens between them; the rest wait for refill.
    (The old per-worker split gave every worker its own free token.)"""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    limiter = RateLimiter.shared(rate=4.0, burst=2, ctx=ctx)
    barrier = ctx.Barrier(4)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=ctx, initializer=_init_shared, initargs=(limiter, barrier)
    ) as pool:
        waits = sorted(pool.map(_acquire_after_barrier, range(4)))
    assert waits[0] == 0.0 and waits[1] == 0.0  # the two burst tokens
    assert waits[2] > 0.1 and waits[3] > 0.1  # everyone else refilled at 4/s
    assert sum(waits) > 0.5
    assert limiter._bucket.tokens < 1.0  # parent sees the drained shared bucket


def test_client_uses_limiter_and_retries_with_backoff(tmp_path: Path) -> None:
    calls: list[dict] = []
    fails = [429, 503]

    def transport(body: dict) -> dict:
        calls.append(body)
        if fails:
            code = fails.pop(0)
            raise JeVError(code, "boom", retry_after=0.25 if code == 429 else None)
        return {"answers": {qid: {"noul": 0.9} for qid in body["questions"]}}

    lim = RateLimiter(100.0, burst=1)
    lim._sleep = lambda _s: None
    client = JeVClient("jev-1.13.0", tmp_path, transport=transport, limiter=lim, max_retries=3)
    sleeps: list[float] = []
    client._sleep = lambda s: sleeps.append(float(s))
    out = client.ask({"x": 1}, {"q1": {"type": "noul", "instructions": "?", "criteria": {}}}, mode="live")
    assert out == {"q1": {"noul": 0.9}}
    assert len(calls) == 3
    assert client.stats["requests"] == 3
    assert client.stats["retries"] == 2
    assert sleeps[0] == 0.25  # Retry-After honoured on the 429
    assert 0.5 <= sleeps[1] <= 1.0  # exponential backoff with jitter on the 503


def test_client_gives_up_after_max_retries(tmp_path: Path) -> None:
    def transport(_body: dict) -> dict:
        raise JeVError(500, "down")

    client = JeVClient("jev-1.13.0", tmp_path, transport=transport, max_retries=1)
    client._sleep = lambda _s: None
    try:
        client.ask({"x": 1}, {"q1": {"type": "noul", "instructions": "?", "criteria": {}}}, mode="live")
    except JeVError as exc:
        assert exc.code == 500
    else:  # pragma: no cover
        raise AssertionError("expected JeVError")
    assert client.stats["requests"] == 2


# --------------------------------------------------------------------------- discovery


def test_discover_units_are_picklable_and_labelled() -> None:
    import pickle

    units = discover(TREE)
    assert units and all(isinstance(u, WorkUnit) for u in units)
    assert all(u.kind == "codex_tree" for u in units)
    roundtrip = pickle.loads(pickle.dumps(units))
    assert [u.label for u in roundtrip] == [u.label for u in units]
    assert all(u.size_bytes > 0 for u in units)


def test_resolve_jobs() -> None:
    assert resolve_jobs(0, 0) == 1
    assert resolve_jobs(0, 1) == 1
    assert resolve_jobs(8, 3) == 3
    assert resolve_jobs(2, 10) == 2
    assert resolve_jobs(0, 10) >= 1


# --------------------------------------------------------------------------- runner


def _strip_generated(report: Report) -> dict:
    data = json.loads(report.model_dump_json())
    data["meta"].pop("generated_at", None)
    return data


def test_run_paths_parallel_equals_sequential() -> None:
    seq_runs, seq_errors, _ = run_paths([TREE, CLAUDE_RUN], RunOptions(jobs=1))
    par_runs, par_errors, _ = run_paths([TREE, CLAUDE_RUN], RunOptions(jobs=2))
    assert not seq_errors and not par_errors
    assert len(seq_runs) == len(par_runs) == 2  # one codex tree + one code-bench run
    meta = ReportMeta(tool_version="t")
    assert _strip_generated(Report.build(seq_runs, meta)) == _strip_generated(Report.build(par_runs, meta))


def test_run_units_reports_unit_error_without_aborting(tmp_path: Path) -> None:
    good = discover(TREE)
    # A claude_session unit pointing at a missing file raises inside the loader.
    units = [*good, WorkUnit(kind="claude_session", paths=[tmp_path / "missing.jsonl"])]
    outcomes = run_units(units, RunOptions(jobs=1))
    assert len(outcomes) == len(units)
    assert all(o.error is None for o in outcomes[:-1])
    last = outcomes[-1]
    assert last.runs == [] and last.error is not None
    assert "missing.jsonl" in last.error.label
    assert last.error.error and last.error.traceback
