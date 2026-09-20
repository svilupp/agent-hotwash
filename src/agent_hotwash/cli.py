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
import sys
import time
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import click  # explicit runtime dep (typer requires it); used for its exception types
import typer
from rich.console import Console
from rich.table import Table

import agent_hotwash.detectors  # noqa: F401 -- import registers all detectors
from agent_hotwash import __version__
from agent_hotwash.analytics import analyze
from agent_hotwash.config import load_config
from agent_hotwash.detectors.registry import Severity, get_registry, run_detectors, severity_rank
from agent_hotwash.report.csv_writer import render_csv
from agent_hotwash.report.html import render_html
from agent_hotwash.report.json_writer import render_json
from agent_hotwash.report.model import Report, ReportMeta, RunResult
from agent_hotwash.report.table import render_table
from agent_hotwash.sources.detect import iter_traces

if TYPE_CHECKING:
    from collections.abc import Iterable


class Format(StrEnum):
    json = "json"
    table = "table"
    csv = "csv"
    html = "html"


# Human/progress messages always go to stderr so stdout stays machine-clean.
_err = Console(stderr=True)

app = typer.Typer(
    name="agent-hotwash",
    help="Analyze coding-agent traces to surface improvement opportunities and bad patterns.",
    no_args_is_help=True,
    add_completion=False,
)


def _default_format() -> Format:
    """Table for an interactive TTY, JSON when piped (agent/CI friendly)."""
    return Format.table if sys.stdout.isatty() else Format.json


def _build_report(
    paths: Iterable[Path],
    *,
    config_path: Path | None,
    run_detectors_enabled: bool,
    since: date | None = None,
    until: date | None = None,
    model_families: Iterable[str] = (),
) -> tuple[Report, int]:
    """Parse -> analyze -> detect over every path; return the Report and the
    number of traces analyzed."""
    paths = list(paths)
    config = load_config(config_path)
    runs: list[RunResult] = []
    families = tuple(_normalize_model_family(value) for value in model_families if value.strip())
    for path in paths:
        for trace in iter_traces(path):
            if not _trace_in_date_range(trace, since=since, until=until):
                continue
            if families and not _model_matches(trace.model, families):
                continue
            analysis = analyze(trace, config)
            findings = run_detectors(trace, config) if run_detectors_enabled else []
            runs.append(RunResult(analysis=analysis, findings=findings))

    meta = ReportMeta(
        tool_version=__version__,
        config_path=str(config_path) if config_path else None,
        inputs=[str(p) for p in paths],
        detectors_enabled=run_detectors_enabled,
        filters={
            **({"since": since.isoformat()} if since else {}),
            **({"until": until.isoformat()} if until else {}),
            **({"model_families": ", ".join(model_families)} if families else {}),
        },
    )
    return Report.build(runs, meta), len(runs)


def _normalize_model_family(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _model_matches(model: str | None, families: tuple[str, ...]) -> bool:
    normalized = _normalize_model_family(model or "")
    return any(family in normalized for family in families)


def _trace_in_date_range(trace, *, since: date | None, until: date | None) -> bool:
    if since is None and until is None:
        return True
    timestamps = [event.ts for event in trace.root.events if event.ts is not None]
    if not timestamps:
        return False
    trace_date = min(timestamps).date()
    return (since is None or trace_date >= since) and (until is None or trace_date <= until)


def _parse_date(value: str | None, option_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("expected YYYY-MM-DD", param_hint=option_name) from exc


def _render(report: Report, fmt: Format) -> str:
    if fmt is Format.json:
        return render_json(report)
    if fmt is Format.csv:
        return render_csv(report)
    if fmt is Format.html:
        return render_html(report)
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


_EXT = {Format.json: "json", Format.csv: "csv", Format.html: "html", Format.table: "txt"}


@app.command(name="analyze")
def analyze_cmd(
    paths: list[Path] = typer.Argument(..., help="Trace files or directories to analyze."),
    fmt: Format | None = typer.Option(
        None, "--format", "-f", help="Output format (default: table on TTY, json when piped)."
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write output to a file or directory (default: stdout)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="User TOML config merged over defaults."),
    no_detectors: bool = typer.Option(False, "--no-detectors", help="Skip detectors (analytics + aggregate only)."),
    fail_on: Severity | None = typer.Option(
        None, "--fail-on", help="Exit non-zero if a finding at/above this severity is present."
    ),
    since: str | None = typer.Option(None, help="Include traces starting on/after YYYY-MM-DD."),
    until: str | None = typer.Option(None, help="Include traces starting on/before YYYY-MM-DD."),
    model_family: list[str] = typer.Option(
        [], "--model-family", help="Case-insensitive model substring; repeat to include multiple families."
    ),
) -> None:
    """Detect, parse, analyze, run detectors, aggregate, and render a report."""
    fmt = fmt or _default_format()
    since_date = _parse_date(since, "--since")
    until_date = _parse_date(until, "--until")
    start = time.monotonic()
    try:
        report, n = _build_report(
            paths,
            config_path=config,
            run_detectors_enabled=not no_detectors,
            since=since_date,
            until=until_date,
            model_families=model_family,
        )
    except Exception as exc:  # surface any pipeline error as exit 1
        _err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if n == 0:
        _err.print("[yellow]no analyzable traces found[/yellow]")
        raise typer.Exit(2)

    elapsed = time.monotonic() - start
    _err.print(f"[green]analyzed[/green] {n} trace(s), {report.total_findings()} finding(s) in {elapsed:.2f}s")

    _emit(_render(report, fmt), out, default_name=f"report.{_EXT[fmt]}")

    if fail_on is not None:
        top = report.max_severity()
        if top is not None and severity_rank(top) >= severity_rank(fail_on):
            _err.print(f"[red]fail-on:[/red] found {top.value} finding (threshold {fail_on.value})")
            raise typer.Exit(2)


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
