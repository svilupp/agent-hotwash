"""Command-line entry point for agent-hotwash (typer app).

AI-friendly output contract:
- **stdout** carries the structured result (JSON/CSV/HTML/rendered table).
- **stderr** carries all human-facing progress and diagnostics.
- exit **0** on success, **1** on a usage/runtime error, **2** when no
  analyzable traces were found (or ``--fail-on`` tripped the CI gate).

The module keeps the ``agent_hotwash.cli:main`` entry point working via the
``raise SystemExit(main())`` pattern; ``main`` invokes the typer app.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import click  # explicit runtime dep (typer requires it); used for its exception types
import typer
from rich.console import Console
from rich.table import Table

import agent_hotwash.detectors  # noqa: F401 -- import registers all detectors
from agent_hotwash import __version__
from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import Severity, get_registry, severity_rank
from agent_hotwash.labels import (
    append_records,
    burned_path,
    collect_drafts,
    eval_store,
    labelled_keys,
    load_records,
    record_key,
)
from agent_hotwash.report.csv_writer import render_csv
from agent_hotwash.report.highlights import HighlightsData, build_highlights_data, render_highlights
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import render_json
from agent_hotwash.report.model import Report, ReportMeta
from agent_hotwash.report.table import render_table
from agent_hotwash.runner import RunOptions, UnitError, UnitOutcome, run_paths
from agent_hotwash.sources.detect import iter_traces

if TYPE_CHECKING:
    from collections.abc import Iterable


class Format(StrEnum):
    json = "json"
    brief_json = "brief-json"
    table = "table"
    csv = "csv"
    html = "html"


class SemanticMode(StrEnum):
    off = "off"
    cached = "cached"
    live = "live"


class ReviewMode(StrEnum):
    cached = "cached"
    live = "live"


# Human/progress messages always go to stderr so stdout stays machine-clean.
_err = Console(stderr=True)

app = typer.Typer(
    name="agent-hotwash",
    help=(
        "Analyze coding-agent traces to surface improvement opportunities and bad patterns. "
        "Commands: analyze, threads, detectors, config-show, version, label, eval."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _default_format() -> Format:
    """Table for an interactive TTY, JSON when piped (agent/CI friendly)."""
    return Format.table if sys.stdout.isatty() else Format.json


_SEMANTIC_MODES = frozenset({"off", "cached", "live"})
_SEMANTIC_ENV = "AGENT_HOTWASH_SEMANTIC"


def _resolve_semantic_mode(flag: SemanticMode | None, cfg_mode: str) -> str:
    """``--semantic`` wins; else ``AGENT_HOTWASH_SEMANTIC`` (pytest/CI); else config."""
    if flag is not None:
        return flag.value
    raw = (os.environ.get(_SEMANTIC_ENV) or "").strip().lower()
    if not raw:
        return cfg_mode
    if raw not in _SEMANTIC_MODES:
        _err.print(f"[red]error:[/red] {_SEMANTIC_ENV} must be off, cached, or live (got {raw!r})")
        raise typer.Exit(1)
    return raw


def _require_live_key(mode: str) -> None:
    if mode != "live":
        return
    if (os.environ.get("TYPESAFE_API_KEY") or "").strip():
        return
    _err.print("[red]error:[/red] TYPESAFE_API_KEY is not set")
    _err.print("live JeV is the default. Export TYPESAFE_API_KEY, or pass --semantic off.")
    raise typer.Exit(1)


def _progress(out: UnitOutcome, done: int, total: int) -> None:
    """One stderr line per finished unit when there is more than one."""
    if total <= 1:
        return
    status = "[red]failed[/red]" if out.error else f"{len(out.runs)} trace(s)"
    _err.print(f"[dim]\\[{done}/{total}][/dim] {out.unit.label} — {status} in {out.seconds:.1f}s")


def _build_report(paths: Iterable[Path], options: RunOptions) -> tuple[Report, int, list[UnitError]]:
    """Discover -> (parallel) load/analyze/detect -> aggregate.

    Returns the Report, the number of traces analyzed and the per-unit errors
    (a failing unit never aborts the batch)."""
    paths = list(paths)
    runs, errors, jev_stats = run_paths(paths, options, on_done=_progress)
    if jev_stats.get("requests"):
        keys = ("requests", "questions_asked", "cache_hits", "cache_misses", "retries", "rate_wait_s")
        r, q, h, m, t, w = (jev_stats.get(k, 0) for k in keys)
        _err.print(
            f"[dim]jev:[/dim] {r:.0f} request(s), {q:.0f} question(s), "
            f"{h:.0f} cache hit(s), {m:.0f} cache miss(es), "
            f"{t:.0f} retry(ies), {w:.1f}s rate-limit wait"
        )
    monthly = None
    if options.semantic_mode != "off":
        from agent_hotwash.aggregate import monthly_rollup

        monthly = monthly_rollup(runs, timezone=load_config(options.config_path).diagnostics.timezone)
    from importlib.metadata import version

    from agent_hotwash.semantic.redact import REDACTION_VERSION

    config = load_config(options.config_path)
    meta = ReportMeta(
        tool_version=__version__,
        config_path=str(options.config_path) if options.config_path else None,
        inputs=[str(p) for p in paths],
        detectors_enabled=options.detectors,
        jev_stats=jev_stats,
        analysis_parameters={
            "jev_model": config.semantic.model,
            "jev_package_version": version("systemoneprompts"),
            "redaction_version": REDACTION_VERSION,
            "handover_negative_threshold": 0.2,
            "handover_positive_threshold": 0.8,
            "cache_rewrite_min_write_tokens": config.diagnostics.cache_rewrite_min_write_tokens,
            "cache_wait_bands_seconds": config.diagnostics.cache_wait_bands_seconds,
            "price_rows_as_of": {name: row.as_of for name, row in config.pricing.items()},
        },
        filters={
            **({"since": options.since.isoformat()} if options.since else {}),
            **({"until": options.until.isoformat()} if options.until else {}),
            **({"model_families": ", ".join(options.model_families)} if options.model_families else {}),
        },
    )
    report = Report.build(runs, meta, monthly=monthly)
    if options.semantic_mode != "off" and report.expense_tail.runs and config.tails.max_expense_reviews:
        from agent_hotwash.runner import make_asker
        from agent_hotwash.semantic.expensive import review_expensive

        asker = make_asker(config, mode=options.semantic_mode)
        try:
            review_expensive(
                report, config, asker, mode=options.semantic_mode, allow_unredacted=options.allow_unredacted
            )
            for key, value in asker.stats().items():
                report.meta.jev_stats[key] = report.meta.jev_stats.get(key, 0) + value
        finally:
            asker.close()
    return report, len(runs), errors


def _normalize_model_family(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _parse_date(value: str | None, option_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("expected YYYY-MM-DD", param_hint=option_name) from exc


def _render(report: Report, fmt: Format, *, full_html: bool = False) -> str:
    if fmt is Format.json:
        return render_json(report)
    if fmt is Format.brief_json:
        return build_highlights_data(report).model_dump_json(indent=2)
    if fmt is Format.csv:
        return render_csv(report)
    if fmt is Format.html:
        return render_html(report) if full_html else render_highlights(report)
    return render_table(report)


def _emit(text: str, out: Path | None, *, default_name: str) -> None:
    """Write rendered output to ``--out`` (file or dir) or to stdout."""
    if out is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    target = out / default_name if out.is_dir() else out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _err.print(f"[green]wrote[/green] {target}")


_EXT = {
    Format.json: "json",
    Format.brief_json: "json",
    Format.csv: "csv",
    Format.html: "html",
    Format.table: "txt",
}


@app.command(name="render-brief")
def render_brief_cmd(
    brief: Path = typer.Argument(..., help="A previously generated brief-json file."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write HTML to a file or directory."),
) -> None:
    """Render highlights from a compact brief without analyzing traces."""
    try:
        data = HighlightsData.model_validate_json(brief.read_text(encoding="utf-8"))
        _emit(render_highlights(data), out, default_name="report.html")
    except Exception as exc:
        _err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command(name="review-brief")
def review_brief_cmd(
    brief: Path = typer.Argument(..., help="A previously generated brief-json file."),
    out: Path = typer.Option(..., "--out", "-o", help="Write the reviewed brief-json file."),
    review_out: Path | None = typer.Option(None, "--review-out", help="Write JeV feature judgments for audit."),
    mode: ReviewMode = typer.Option(
        ReviewMode.live, "--semantic", help="JeV mode: live (reuse cache, fetch misses) or cached (fail on miss)."
    ),
) -> None:
    """Review grouped actions and add at most one strong JeV discovery card."""
    from agent_hotwash.report.jev_review import review_brief
    from agent_hotwash.semantic.client import SystemOneAsker

    _require_live_key(mode.value)
    asker: SystemOneAsker | None = None
    try:
        data = HighlightsData.model_validate_json(brief.read_text(encoding="utf-8"))
        asker = SystemOneAsker("jev-1.13.0", ".cache/agent-hotwash/priority-review-v3", mode=mode.value)
        reviewed, details = review_brief(data, asker)
        _emit(reviewed.model_dump_json(indent=2), out, default_name="reviewed-brief.json")
        if review_out is not None:
            _emit(details.model_dump_json(indent=2), review_out, default_name="jev-review.json")
        stats = asker.stats()
        _err.print(
            f"[dim]jev review:[/dim] {stats['requests']:.0f} request(s), "
            f"{stats['questions_asked']:.0f} question(s), "
            f"{len(details.rescued_ids)} discovery card(s)"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        if asker is not None:
            asker.close()


@app.command(name="analyze")
def analyze_cmd(
    paths: list[Path] = typer.Argument(..., help="Trace files or directories to analyze."),
    fmt: Format | None = typer.Option(
        None, "--format", "-f", help="Output format (default: table on TTY, json when piped)."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write output to a file or directory; HTML also writes linked run pages."
    ),
    full_html: bool = typer.Option(
        False, "--full-html", help="Write the legacy single-file evidence view instead of a linked HTML report."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="User TOML config merged over defaults."),
    no_detectors: bool = typer.Option(False, "--no-detectors", help="Skip detectors (analytics + aggregate only)."),
    fail_on: Severity | None = typer.Option(
        None, "--fail-on", help="Exit non-zero if a finding at/above this severity is present."
    ),
    semantic: SemanticMode | None = typer.Option(
        None,
        "--semantic",
        help="JeV mode: off, cached (miss exits 1), or live (needs TYPESAFE_API_KEY). Default: live.",
    ),
    allow_unredacted: bool = typer.Option(
        False,
        "--allow-unredacted",
        help="Let live JeV run when [semantic] redact = false (C9 override; does not disable redaction).",
    ),
    jobs: int = typer.Option(
        0,
        "--jobs",
        "-j",
        min=0,
        help="Worker processes for loading/analyzing traces (0 = one per CPU, capped at the number of traces).",
    ),
    since: str | None = typer.Option(None, help="Include traces starting on/after YYYY-MM-DD."),
    until: str | None = typer.Option(None, help="Include traces starting on/before YYYY-MM-DD."),
    model_family: list[str] = typer.Option(
        [], "--model-family", help="Case-insensitive model substring; repeat to include multiple families."
    ),
) -> None:
    """Detect, parse, analyze, run detectors, aggregate, and render a report.

    Traces are analyzed in parallel worker processes (``--jobs``). In live
    semantic mode the JeV request budget (``[semantic] requests_per_second``)
    is one shared token bucket across workers. A trace that fails to parse is
    reported on stderr and skipped; the report still covers every other trace
    and the exit code is 1.
    """
    fmt = fmt or _default_format()
    since_date = _parse_date(since, "--since")
    until_date = _parse_date(until, "--until")
    start = time.monotonic()
    try:
        cfg = load_config(config)  # fail fast on a bad user config, before spawning workers
        mode = _resolve_semantic_mode(semantic, cfg.semantic.mode)
        _require_live_key(mode)
        options = RunOptions(
            config_path=config,
            detectors=not no_detectors,
            semantic_mode=mode,
            allow_unredacted=allow_unredacted,
            jobs=jobs,
            since=since_date,
            until=until_date,
            model_families=tuple(_normalize_model_family(value) for value in model_family if value.strip()),
        )
        report, n, errors = _build_report(paths, options)
    except typer.Exit:
        raise
    except Exception as exc:  # surface any pipeline error as exit 1
        _err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    for err in errors:
        _err.print(f"[red]error:[/red] {err.label}: {err.error}")
    if n == 0:
        if errors:
            raise typer.Exit(1)
        _err.print("[yellow]no analyzable traces found[/yellow]")
        raise typer.Exit(2)

    elapsed = time.monotonic() - start
    _err.print(f"[green]analyzed[/green] {n} trace(s), {report.total_findings()} finding(s) in {elapsed:.2f}s")

    if fmt is Format.html and out is not None and not full_html:
        from agent_hotwash.report.site import write_html_site

        try:
            target = write_html_site(report, out)
        except Exception as exc:
            _err.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(1) from exc
        _err.print(f"[green]wrote[/green] {target} and linked run evidence")
    else:
        _emit(_render(report, fmt, full_html=full_html), out, default_name=f"report.{_EXT[fmt]}")

    if errors:
        _err.print(f"[red]{len(errors)} unit(s) failed[/red] (report covers the rest)")
        raise typer.Exit(1)
    if fail_on is not None:
        top = report.max_severity()
        if top is not None and severity_rank(top) >= severity_rank(fail_on):
            _err.print(f"[red]fail-on:[/red] found {top.value} finding (threshold {fail_on.value})")
            raise typer.Exit(2)


def _thread_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in iter_traces(path):
        rows.append({"id": trace.root.session_id, "parent": None, "kind": "root", "evidence": []})
        for link in trace.links:
            kind = link.kind.value if hasattr(link.kind, "value") else str(link.kind)
            rows.append(
                {
                    "id": link.child_id,
                    "parent": link.parent_id,
                    "kind": kind,
                    "evidence": list(link.evidence),
                }
            )
    return rows


@app.command(name="threads")
def threads_cmd(
    path: Path = typer.Argument(..., help="Trace file or directory of connected-component thread trees."),
    fmt: Format | None = typer.Option(
        None, "--format", "-f", help="json or table (default: table on TTY, json when piped)."
    ),
) -> None:
    """List connected-component thread trees (id, parent, kind, evidence)."""
    fmt = fmt or _default_format()
    if fmt not in {Format.json, Format.table}:
        _err.print("[red]error:[/red] threads supports --format json or table")
        raise typer.Exit(1)
    try:
        rows = _thread_rows(path)
    except Exception as exc:
        _err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not rows:
        _err.print("[yellow]no analyzable traces found[/yellow]")
        raise typer.Exit(2)
    if fmt is Format.json:
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        return
    t = Table(title=f"Thread trees ({len(rows)})")
    for col in ("id", "parent", "kind", "evidence"):
        t.add_column(col, overflow="fold")
    for row in rows:
        evidence = row["evidence"]
        ev = ", ".join(str(x) for x in evidence) if isinstance(evidence, list) else str(evidence)
        t.add_row(str(row["id"]), str(row["parent"] or ""), str(row["kind"]), ev)
    Console().print(t)


@app.command()
def detectors(
    fmt: Format = typer.Option(Format.table, "--format", "-f", help="json or table."),
) -> None:
    """List registered detectors (id, kind, tier, severity) for discovery."""
    reg = get_registry()
    if fmt is Format.json:
        rows = [
            {
                "id": s.id,
                "kind": s.kind,
                "tier": s.tier,
                "severity": s.default_severity.value,
                "confidence": s.default_confidence,
                "llm_candidate": s.llm_candidate,
                "doc": s.doc,
            }
            for s in reg.values()
        ]
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        return

    t = Table(title=f"Registered detectors ({len(reg)})")
    for col in ("id", "kind", "tier", "severity", "confidence"):
        t.add_column(col)
    for s in reg.values():
        t.add_row(s.id, s.kind, s.tier, s.default_severity.value, s.default_confidence)
    Console().print(t)


@app.command(name="config-show")
def config_show(
    config: Path | None = typer.Option(None, "--config", "-c", help="User TOML config merged over defaults."),
) -> None:
    """Dump the effective merged config as JSON to stdout."""
    cfg = load_config(config)
    sys.stdout.write(cfg.model_dump_json(indent=2) + "\n")


@app.command()
def version() -> None:
    """Print the agent-hotwash version."""
    sys.stdout.write(f"{__version__}\n")


@app.command(name="label")
def label_cmd(
    path: Path = typer.Argument(..., help="Trace file or directory to label."),
    store: Path = typer.Option(..., "--store", help="JSONL label store path (created/resumed)."),
    annotator: str = typer.Option("unknown", "--annotator", help="Annotator id written on each record."),
    split: str = typer.Option("dev", "--split", help="Split label (dev or held-out); pinned by root_trace_id."),
    answer: str | None = typer.Option(None, "--answer", help="Answer to write (non-interactive)."),
    confidence: float | None = typer.Option(None, "--confidence", help="Optional confidence for --answer."),
    feature: list[str] | None = typer.Option(None, "--feature", help="Limit to this feature id (repeatable)."),
    provenance: str = typer.Option("real", "--provenance", help="real or synthetic."),
    config: Path | None = typer.Option(None, "--config", "-c", help="User TOML config merged over defaults."),
) -> None:
    """Write or resume a JSONL label store (PLAN §9.7)."""
    if provenance not in {"real", "synthetic"}:
        _err.print("[red]error:[/red] provenance must be real or synthetic")
        raise typer.Exit(1)
    from agent_hotwash.labels import RecordKey, SplitConflict, check_split_pinned, pinned_splits

    cfg = load_config(config)
    existing = load_records(store)
    seen = labelled_keys(existing)  # keyed by (item, feature, version, criteria, annotator)
    pinned = pinned_splits(existing)  # split is pinned per root_trace_id (§9.7)
    feature_ids = set(feature) if feature else None
    tty = sys.stdin.isatty()
    written = 0
    skipped_resume = 0
    try:
        traces = list(iter_traces(path))
    except Exception as exc:
        _err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not traces:
        _err.print("[yellow]no analyzable traces found[/yellow]")
        raise typer.Exit(2)
    batch = []
    this_run: set[RecordKey] = set()
    for trace in traces:
        try:
            check_split_pinned(pinned, trace.trace_id, split)
        except SplitConflict as exc:
            _err.print(f"[red]error:[/red] {exc} (splits are pinned by root_trace_id; use --split {exc.pinned})")
            raise typer.Exit(1) from exc
        drafts = collect_drafts(
            trace,
            cfg,
            annotator=annotator,
            split=split,
            provenance=cast(Literal["real", "synthetic"], provenance),
            feature_ids=feature_ids,
        )
        for rec in drafts:
            key = record_key(rec)
            if key in seen or key in this_run:
                skipped_resume += 1
                continue
            this_run.add(key)
            if answer is not None:
                rec.answer = answer
                rec.confidence = confidence
            elif not tty:
                rec.skip_reason = "non_tty"
            else:
                prompt = f"{rec.feature_id} on {rec.object_id} [{rec.scope}]: "
                rec.answer = input(prompt) or None
                if rec.answer is None:
                    rec.skip_reason = "empty"
            batch.append(rec)
    if batch:
        append_records(store, batch)
        written = len(batch)
    _err.print(f"[green]labelled[/green] wrote {written}, resumed-skip {skipped_resume}")


@app.command(name="eval")
def eval_cmd(
    store: Path = typer.Option(..., "--store", help="JSONL label store to evaluate."),
    held_out: bool = typer.Option(False, "--held-out", help="Refuse if the held-out split was previously inspected."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the JSON report to a file."),
) -> None:
    """Compute per-feature agreement and positive rates from a label store."""
    marker = burned_path(store)
    if held_out and marker.is_file():
        _err.print("[red]error:[/red] held-out store was previously inspected (burned marker present)")
        raise typer.Exit(1)
    if not store.is_file():
        _err.print(f"[red]error:[/red] label store not found: {store}")
        raise typer.Exit(1)
    records = load_records(store)
    report = eval_store(records)
    text = json.dumps(report, indent=2)
    _emit(text, out, default_name="eval.json")
    if held_out:
        marker.write_text("burned\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0/1/2)."""
    try:
        # standalone_mode=False makes click return the Exit code instead of
        # calling sys.exit, so we can hand a plain int back to callers.
        rv = app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code
    except SystemExit as exc:  # e.g. --help
        return int(exc.code or 0) if isinstance(exc.code, int) else 0
    except (typer.BadParameter, click.ClickException) as exc:
        _err.print(f"[red]error:[/red] {exc}")
        return 1
    return rv if isinstance(rv, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
