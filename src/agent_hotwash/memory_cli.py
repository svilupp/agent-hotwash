"""Memory analysis orchestration behind the shared ``analyze`` command."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


def load_memory_environment() -> None:
    """Read local credentials without replacing values already in the environment."""
    selected = Path.cwd() / ".env"
    if selected.is_file():
        load_dotenv(selected, override=False)


def audit_directory(paths, options) -> Path:
    """Stable private artifact location shared by offline, cached and live runs."""
    identity = {
        "paths": sorted({str(Path(path).expanduser().resolve()) for path in paths}),
        "since": options.since.isoformat() if options.since else None,
        "until": options.until.isoformat() if options.until else None,
        "model_families": sorted(set(options.model_families)),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return Path(".cache/agent-hotwash/memory") / key


def run_memory_analysis(paths, *, config, options, guidance_paths=()):
    """Run the audit with the same input selection and provider mode as analyze."""
    from agent_hotwash.memory.pipeline import load_saved, run_audit

    output = audit_directory(paths, options)
    summary = run_audit(
        paths,
        output,
        config=config,
        extract=options.semantic_mode != "off",
        allow_paid=options.semantic_mode == "live",
        guidance_paths=guidance_paths,
        run_options=options,
    )
    evidence, result = load_saved(output)
    return output, summary, evidence, result


def render_memory_analysis(evidence, result, format_name: str) -> str:
    from agent_hotwash.report.memory import render_memory_html, render_memory_markdown

    if format_name == "html":
        return render_memory_html(evidence, result)
    if format_name == "json":
        return json.dumps({"evidence": evidence, "memory": result}, ensure_ascii=False, indent=2, allow_nan=False)
    if format_name == "table":
        return render_memory_markdown(evidence, result)
    raise ValueError("Memory analysis supports --format html, json, or table")


def memory_sidecar(out: Path | None, audit_dir: Path) -> Path:
    if out is None:
        return audit_dir / "report.html"
    if out.is_dir():
        return out / "memory.html"
    return out.with_name(out.stem + ".memory.html")


def write_memory_report(path: Path, text: str) -> None:
    """Publish reports atomically, keeping potentially sensitive audit text private."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".memory-report-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
