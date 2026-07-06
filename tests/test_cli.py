"""CLI tests via typer's CliRunner + end-to-end over the trace fixtures.

Asserts the stdout/stderr contract, exit codes (0 ok / 1 error / 2 no traces),
and that the full parse -> analyze -> detect -> report pipeline runs on the
committed fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_hotwash.cli import app, main

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures"
CLAUDE_RUN = FIXTURES / "codebench" / "claude_run"
CODEX_ROLLOUT = FIXTURES / "codex_native" / "rollout-fixture.jsonl"
CLAUDE_NATIVE = FIXTURES / "claude_native" / "proj"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_detectors_json_lists_registry() -> None:
    result = runner.invoke(app, ["detectors", "--format", "json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    ids = {r["id"] for r in rows}
    assert "EDIT_THRASH" in ids or "EDIT_WITHOUT_READ" in ids
    assert all({"id", "kind", "tier", "severity"} <= set(r) for r in rows)


def test_config_show_is_valid_json() -> None:
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    cfg = json.loads(result.stdout)
    assert "smells" in cfg
    assert "pricing" in cfg


def test_analyze_json_default(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] == 1
    assert data["runs"][0]["analysis"]["agent"] == "claude"


def test_analyze_no_traces_exit_2(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["analyze", str(empty), "--format", "json"])
    assert result.exit_code == 2


def test_analyze_missing_path_exit_2(tmp_path) -> None:
    # A non-existent path yields no traces (detect warns) -> exit 2.
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope"), "--format", "json"])
    assert result.exit_code == 2


def test_analyze_csv_shape() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "csv"])
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert "trace_id" in lines[0]
    assert len(lines) >= 2


def test_analyze_html_self_contained() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "html"])
    assert result.exit_code == 0
    assert "<!doctype html>" in result.stdout
    assert "https://" not in result.stdout


def test_analyze_out_file(tmp_path) -> None:
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    json.loads(out.read_text())


def test_analyze_out_dir(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "html", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "report.html").exists()


def test_analyze_no_detectors_zero_findings() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--no-detectors"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["finding_histogram"] == {}
    assert data["meta"]["detectors_enabled"] is False


def test_analyze_codex_native() -> None:
    result = runner.invoke(app, ["analyze", str(CODEX_ROLLOUT), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["runs"][0]["analysis"]["agent"] == "codex"


def test_analyze_claude_native() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_NATIVE), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] >= 1


def test_analyze_multiple_paths() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), str(CODEX_ROLLOUT), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] == 2


def test_fail_on_gate() -> None:
    # If any finding is present, --fail-on info must trip the CI gate (exit 2).
    probe = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json"])
    total = json.loads(probe.stdout)["finding_histogram"]
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--fail-on", "info"])
    if total:
        assert result.exit_code == 2
    else:
        assert result.exit_code == 0


def test_main_entrypoint_returns_int() -> None:
    assert main(["version"]) == 0
    assert main(["analyze", "/does/not/exist/xyz", "--format", "json"]) == 2
