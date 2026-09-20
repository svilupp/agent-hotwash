"""CLI tests via typer's CliRunner + end-to-end over the trace fixtures.

Asserts the stdout/stderr contract, exit codes (0 ok / 1 error / 2 no traces),
and that the full parse -> analyze -> detect -> report pipeline runs on the
committed fixtures.

Analyze invocations that must stay deterministic pass ``--semantic off``
(``SEM_OFF``). Default config is live and needs a TypeSafe key; CI has none.
The missing-key path is ``test_analyze_live_without_key_exits_1``. Network
JeV lives under ``tests/live/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_hotwash.cli import app, main

runner = CliRunner()
SEM_OFF = ("--semantic", "off")

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
    assert cfg["semantic"]["mode"] == "live"


def test_analyze_json_default(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", *SEM_OFF])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] == 1
    assert data["runs"][0]["analysis"]["agent"] == "claude"


def test_analyze_no_traces_exit_2(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["analyze", str(empty), "--format", "json", *SEM_OFF])
    assert result.exit_code == 2


def test_analyze_missing_path_exit_2(tmp_path) -> None:
    # A non-existent path yields no traces (detect warns) -> exit 2.
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope"), "--format", "json", *SEM_OFF])
    assert result.exit_code == 2


def test_analyze_csv_shape() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "csv", *SEM_OFF])
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert "trace_id" in lines[0]
    assert len(lines) >= 2


def test_analyze_html_self_contained() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "html", *SEM_OFF])
    assert result.exit_code == 0
    assert "<!doctype html>" in result.stdout
    assert "https://" not in result.stdout


def test_analyze_out_file(tmp_path) -> None:
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--out", str(out), *SEM_OFF])
    assert result.exit_code == 0
    assert out.exists()
    json.loads(out.read_text())


def test_analyze_out_dir(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "html", "--out", str(tmp_path), *SEM_OFF])
    assert result.exit_code == 0
    assert (tmp_path / "report.html").exists()


def test_analyze_no_detectors_zero_findings() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--no-detectors", *SEM_OFF])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["finding_histogram"] == {}
    assert data["meta"]["detectors_enabled"] is False


def test_analyze_codex_native() -> None:
    result = runner.invoke(app, ["analyze", str(CODEX_ROLLOUT), "--format", "json", *SEM_OFF])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["runs"][0]["analysis"]["agent"] == "codex"


def test_analyze_claude_native() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_NATIVE), "--format", "json", *SEM_OFF])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] >= 1


def test_analyze_multiple_paths() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), str(CODEX_ROLLOUT), "--format", "json", *SEM_OFF])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] == 2


def test_analyze_date_and_model_filters() -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            str(CLAUDE_RUN),
            "--format",
            "json",
            "--since",
            "2026-01-01",
            "--until",
            "2026-01-02",
            "--model-family",
            "opus 4 8",
            *SEM_OFF,
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["aggregate"]["total_traces"] == 1
    assert data["meta"]["filters"] == {
        "since": "2026-01-01",
        "until": "2026-01-02",
        "model_families": "opus 4 8",
    }


def test_fail_on_gate() -> None:
    # If any finding is present, --fail-on info must trip the CI gate (exit 2).
    probe = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", *SEM_OFF])
    total = json.loads(probe.stdout)["finding_histogram"]
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--fail-on", "info", *SEM_OFF])
    if total:
        assert result.exit_code == 2
    else:
        assert result.exit_code == 0


def test_main_entrypoint_returns_int() -> None:
    assert main(["version"]) == 0
    assert main(["analyze", "/does/not/exist/xyz", "--format", "json", "--semantic", "off"]) == 2


def test_analyze_semantic_off_omits_structure() -> None:
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--semantic", "off"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    run0 = data["runs"][0]
    assert "structure" not in run0
    assert "features" not in run0
    assert "capabilities" not in run0
    assert "cost_views" not in run0
    assert "monthly" not in data


def test_threads_json_tree_fixture() -> None:
    tree = FIXTURES / "codex_native" / "v0153" / "tree"
    result = runner.invoke(app, ["threads", str(tree), "--format", "json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)
    ids = {r["id"] for r in rows}
    parent = "bbbbbbbb-0000-0000-0000-000000000002"
    spawn = "cccccccc-0000-0000-0000-000000000003"
    fork = "dddddddd-0000-0000-0000-000000000004"
    assert parent in ids
    spawn_row = next(r for r in rows if r["id"] == spawn)
    assert spawn_row["parent"] == parent
    assert spawn_row["kind"] == "spawn"
    assert spawn_row["evidence"]
    fork_row = next(r for r in rows if r["id"] == fork)
    assert fork_row["parent"] == parent
    assert fork_row["kind"] == "fork"


def test_threads_rejects_csv_and_html() -> None:
    tree = FIXTURES / "codex_native" / "v0153" / "tree"
    csv_result = runner.invoke(app, ["threads", str(tree), "--format", "csv"])
    assert csv_result.exit_code == 1
    html_result = runner.invoke(app, ["threads", str(tree), "--format", "html"])
    assert html_result.exit_code == 1


def test_analyze_live_without_key_exits_1(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HOTWASH_SEMANTIC", raising=False)
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json"])
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY" in result.stderr
    assert "--semantic off" in result.stderr
    explicit = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json", "--semantic", "live"])
    assert explicit.exit_code == 1
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    whitespace = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json"])
    assert whitespace.exit_code == 1


def test_analyze_env_off_does_not_need_key(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_HOTWASH_SEMANTIC", "off")
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json"])
    assert result.exit_code == 0, result.stderr
    assert "structure" not in json.loads(result.stdout)["runs"][0]


def test_analyze_invalid_semantic_env_exits_1(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_HOTWASH_SEMANTIC", "nope")
    result = runner.invoke(app, ["analyze", str(CLAUDE_RUN), "--format", "json"])
    assert result.exit_code == 1
    assert "AGENT_HOTWASH_SEMANTIC" in result.stderr


def test_analyze_cached_miss_exits_1(tmp_path: Path) -> None:
    """A cold cache in ``cached`` mode is an error (exit 1) but the report for
    the traces that did succeed is still written."""
    cfg = tmp_path / "semantic.toml"
    cache = tmp_path / "jev-cache"
    cfg.write_text(f'[semantic]\ncache_dir = "{cache}"\n', encoding="utf-8")
    multiturn = FIXTURES / "codex_native" / "v0153" / "user_multiturn.jsonl"
    result = runner.invoke(
        app,
        ["analyze", str(multiturn), "--format", "json", "--semantic", "cached", "--config", str(cfg)],
    )
    assert result.exit_code == 1
    assert "cache miss" in result.stderr or "missing" in result.stderr


def test_analyze_jobs_parallel_matches_sequential() -> None:
    tree = FIXTURES / "codex_native" / "v0153" / "tree"
    seq = runner.invoke(app, ["analyze", str(tree), str(CLAUDE_RUN), "--format", "json", "--jobs", "1", *SEM_OFF])
    par = runner.invoke(app, ["analyze", str(tree), str(CLAUDE_RUN), "--format", "json", "--jobs", "2", *SEM_OFF])
    assert seq.exit_code == 0, seq.stderr
    assert par.exit_code == 0, par.stderr
    a, b = json.loads(seq.stdout), json.loads(par.stdout)
    a["meta"].pop("generated_at")
    b["meta"].pop("generated_at")
    assert a == b
