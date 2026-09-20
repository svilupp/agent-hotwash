"""Per-trace pipeline and the parallel runner that maps it over work units.

Throughput model
----------------
- **Discovery is cheap and sequential** (``sources.detect.discover``): one
  picklable :class:`WorkUnit` per trace, never a parsed trace.
- **Loading + analysis is CPU-bound** → units are mapped over a
  ``ProcessPoolExecutor`` (``jobs`` workers, largest unit first for balance).
  Each worker loads the config once, registers detectors once, and builds one
  :class:`SystemOneAsker` whose :class:`RateLimiter` draws from one token bucket
  in shared memory (``RateLimiter.shared``) — so the aggregate rate *and*
  burst are bounded no matter how many workers run.
- **System One is I/O-bound** → inside one trace, independent questions
  (episodes, tasks) are asked concurrently up to ``semantic.max_concurrency``
  threads sharing that per-process limiter (see ``semantic.pipeline``).
- A failing unit never aborts the batch: it is returned as a
  :class:`UnitError` and reported by the caller.

``run_trace`` is the single-trace pipeline (analytics → detectors → structure
→ System One → diagnostics) used both by the runner and by callers holding a
Trace.
"""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_hotwash.analytics import analyze
from agent_hotwash.config import Config, load_config
from agent_hotwash.report.model import RunResult, StructureSection
from agent_hotwash.sources.detect import WorkUnit, discover, load_unit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from agent_hotwash.events import Trace
    from agent_hotwash.semantic.client import SystemOneAsker
    from agent_hotwash.semantic.ratelimit import RateLimiter


@dataclass(frozen=True)
class RunOptions:
    """Everything a worker needs to reproduce the caller's pipeline settings."""

    config_path: Path | None = None
    detectors: bool = True
    semantic_mode: str = "off"  # off | cached | live
    allow_unredacted: bool = False
    jobs: int = 0  # 0 = auto (min(cpu_count, units)); 1 = in-process
    since: date | None = None
    until: date | None = None
    model_families: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnitError:
    """A unit that raised; the batch continues without it."""

    label: str
    error: str
    traceback: str


@dataclass(frozen=True)
class UnitOutcome:
    index: int  # position in the discovery order
    unit: WorkUnit
    runs: list[RunResult]
    error: UnitError | None
    seconds: float
    jev_stats: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# single-trace pipeline
# ---------------------------------------------------------------------------


def make_asker(config: Config, *, limiter: RateLimiter | None = None, mode: str) -> SystemOneAsker:
    """One asker per process. ``limiter`` is the (possibly process-shared)
    global request budget; when omitted a private one is built from config."""
    from agent_hotwash.semantic.client import SystemOneAsker
    from agent_hotwash.semantic.ratelimit import RateLimiter

    if limiter is None:
        limiter = RateLimiter(config.semantic.requests_per_second, config.semantic.burst)
    if mode not in {"cached", "live"}:
        raise ValueError(f"unsupported System One mode {mode!r}")
    return SystemOneAsker(
        config.semantic.model,
        config.semantic.cache_dir,
        mode=mode,
        max_questions=config.semantic.max_questions_per_request,
        limiter=limiter,
        max_retries=config.semantic.max_retries,
        timeout_s=config.semantic.timeout_s,
        secret_patterns=list(config.lexicons.secret),
    )


def run_trace(
    trace: Trace,
    config: Config,
    *,
    detectors: bool = True,
    semantic_mode: str = "off",
    allow_unredacted: bool = False,
    asker: SystemOneAsker | None = None,
) -> RunResult:
    """Analytics → detectors → (semantic ≠ off) structure, System One, diagnostics."""
    import agent_hotwash.detectors  # noqa: F401 -- registers all detectors
    from agent_hotwash.detectors.registry import run_detectors

    findings = run_detectors(trace, config) if detectors else []
    analysis = analyze(trace, config)
    if semantic_mode == "off":
        return RunResult(analysis=analysis, findings=findings)

    from agent_hotwash.diagnostics.cost_views import build_cost_views
    from agent_hotwash.diagnostics.engine import attach_counterfactual, diagnose
    from agent_hotwash.semantic.pipeline import annotate_trace

    if asker is None:
        asker = make_asker(config, mode=semantic_mode)
    tasks, episodes, feature_sets, caps = annotate_trace(
        trace, config, mode=semantic_mode, allow_unredacted=allow_unredacted, asker=asker
    )
    views = build_cost_views(trace, config, episodes=episodes, tasks=tasks)
    diagnoses = diagnose(trace, tasks, episodes, feature_sets, findings, config)
    views = attach_counterfactual(views, diagnoses)
    return RunResult(
        analysis=analysis,
        findings=findings,
        structure=StructureSection(tasks=list(tasks), episodes=list(episodes)),
        features=list(feature_sets),
        capabilities=caps,
        cost_views=views,
    )


# ---------------------------------------------------------------------------
# worker plumbing
# ---------------------------------------------------------------------------

# Per-process state, initialised once by ``_init_worker`` (or lazily in-process).
_STATE: dict[str, Any] = {}


def _init_worker(options: RunOptions, limiter: RateLimiter | None = None) -> None:
    _STATE.clear()
    _STATE["options"] = options
    _STATE["config"] = load_config(options.config_path)
    _STATE["asker"] = (
        make_asker(_STATE["config"], limiter=limiter, mode=options.semantic_mode)
        if options.semantic_mode != "off"
        else None
    )


def _global_limiter(options: RunOptions, ctx: Any) -> RateLimiter | None:
    """One token bucket in shared memory for every worker (exact global burst)."""
    if options.semantic_mode == "off":
        return None
    from agent_hotwash.semantic.ratelimit import RateLimiter

    cfg = load_config(options.config_path).semantic
    return RateLimiter.shared(cfg.requests_per_second, cfg.burst, ctx=ctx)


def _normalize_model_family(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _model_matches(model: str | None, families: tuple[str, ...]) -> bool:
    normalized = _normalize_model_family(model or "")
    return any(family in normalized for family in families)


def _trace_in_date_range(trace: Trace, *, since: date | None, until: date | None) -> bool:
    if since is None and until is None:
        return True
    timestamps = [event.ts for event in trace.root.events if event.ts is not None]
    if not timestamps:
        return False
    trace_date = min(timestamps).date()
    return (since is None or trace_date >= since) and (until is None or trace_date <= until)


def _keep_trace(trace: Trace, options: RunOptions) -> bool:
    if not _trace_in_date_range(trace, since=options.since, until=options.until):
        return False
    return not options.model_families or _model_matches(trace.model, options.model_families)


def _run_unit(index: int, unit: WorkUnit) -> UnitOutcome:
    options: RunOptions = _STATE["options"]
    config: Config = _STATE["config"]
    asker = _STATE.get("asker")
    start = time.monotonic()
    before = dict(asker.stats()) if asker is not None else {}
    runs: list[RunResult] = []
    try:
        for trace in load_unit(unit):
            if not _keep_trace(trace, options):
                continue
            runs.append(
                run_trace(
                    trace,
                    config,
                    detectors=options.detectors,
                    semantic_mode=options.semantic_mode,
                    allow_unredacted=options.allow_unredacted,
                    asker=asker,
                )
            )
        error = None
    except Exception as exc:  # one bad unit must not sink the batch
        error = UnitError(label=unit.label, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    # Client counters are cumulative per process; report this unit's delta so
    # the parent can simply sum across units.
    stats = {k: v - before.get(k, 0) for k, v in asker.stats().items()} if asker is not None else None
    return UnitOutcome(
        index=index, unit=unit, runs=runs, error=error, seconds=time.monotonic() - start, jev_stats=stats
    )


def resolve_jobs(requested: int, n_units: int) -> int:
    """``0`` → auto: one worker per CPU, never more than there are units."""
    if n_units <= 1:
        return 1
    if requested and requested > 0:
        return min(requested, n_units)
    return max(1, min(os.cpu_count() or 1, n_units))


def run_units(
    units: Sequence[WorkUnit],
    options: RunOptions,
    *,
    on_done: Callable[[UnitOutcome, int, int], None] | None = None,
) -> list[UnitOutcome]:
    """Map the pipeline over ``units``; results come back in discovery order.

    ``on_done(outcome, completed, total)`` is called as each unit finishes (from
    the parent process), for progress reporting.
    """
    units = list(units)
    total = len(units)
    if total == 0:
        return []
    jobs = resolve_jobs(options.jobs, total)
    outcomes: list[UnitOutcome | None] = [None] * total
    done = 0

    if jobs == 1:
        _init_worker(options)
        for i, unit in enumerate(units):
            out = _run_unit(i, unit)
            outcomes[i] = out
            done += 1
            if on_done is not None:
                on_done(out, done, total)
        return [o for o in outcomes if o is not None]

    # Largest units first so a 400 MB rollout does not start last.
    order = sorted(range(total), key=lambda i: units[i].size_bytes, reverse=True)
    ctx = multiprocessing.get_context("spawn")
    limiter = _global_limiter(options, ctx)
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=ctx, initializer=_init_worker, initargs=(options, limiter)
    ) as pool:
        futures = {pool.submit(_run_unit, i, units[i]): i for i in order}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:  # worker crashed / result unpicklable
                out = UnitOutcome(
                    index=i,
                    unit=units[i],
                    runs=[],
                    error=UnitError(label=units[i].label, error=f"{type(exc).__name__}: {exc}", traceback=""),
                    seconds=0.0,
                )
            outcomes[i] = out
            done += 1
            if on_done is not None:
                on_done(out, done, total)
    return [o for o in outcomes if o is not None]


def run_paths(
    paths: Iterable[Path],
    options: RunOptions,
    *,
    on_done: Callable[[UnitOutcome, int, int], None] | None = None,
) -> tuple[list[RunResult], list[UnitError], dict[str, Any]]:
    """Discover every path, run all units, return ``(runs, errors, jev_stats)``."""
    units: list[WorkUnit] = []
    for path in paths:
        units.extend(discover(Path(path)))
    outcomes = run_units(units, options, on_done=on_done)
    runs = [r for o in outcomes for r in o.runs]
    errors = [o.error for o in outcomes if o.error is not None]
    return runs, errors, _merge_jev_stats(outcomes)


def _merge_jev_stats(outcomes: Sequence[UnitOutcome]) -> dict[str, Any]:
    """Sum the per-unit System One counter deltas (requests, questions, cache hits…)."""
    totals: dict[str, float] = {}
    for o in outcomes:
        for k, v in (o.jev_stats or {}).items():
            totals[k] = totals.get(k, 0.0) + float(v)
    return totals


__all__ = [
    "RunOptions",
    "UnitError",
    "UnitOutcome",
    "make_asker",
    "resolve_jobs",
    "run_paths",
    "run_trace",
    "run_units",
]
