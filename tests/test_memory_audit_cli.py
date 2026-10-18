"""Native CLI audit integration; all providers are mocked or replayed offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_hotwash.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_memory_audit_native_and_codebench_directories_preserve_supported_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "analyze",
            "--analysis",
            "memory",
            "--semantic",
            "off",
            str(FIXTURES / "claude_native" / "proj"),
            str(FIXTURES / "codebench" / "codex_run"),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["memory"]["status"] == "preprocessed"
    output = next((tmp_path / ".cache/agent-hotwash/memory").iterdir())
    prepared = json.loads((output / "prepared.json").read_text())
    assert prepared["records"]
    assert any(record["role"] == "assistant" for record in prepared["records"])
    assert (output / "report.md").is_file()
    assert not (output / "calls").exists()


def test_memory_audit_empty_directory_persists_explicit_blocked_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "empty"
    source.mkdir()
    result = runner.invoke(app, ["analyze", str(source), "--analysis", "memory", "--semantic", "off"])
    assert result.exit_code == 2
    output = next((tmp_path / ".cache/agent-hotwash/memory").iterdir())
    assert json.loads((output / "memory.json").read_text())["status"] == "blocked"
    assert "blocked" in (output / "report.html").read_text().lower()


def test_memory_audit_mocked_extraction_and_cache_only_replay(tmp_path, monkeypatch):
    from agent_hotwash.memory.extraction import ProviderCache

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "native-cli-test", "cwd": "/repo"}})
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "No, use shared guidance."}],
                },
            }
        )
        + "\n"
    )
    calls = []

    def mocked_transport(provider, request):
        calls.append(provider)
        if provider == "jev":
            return {"answers": {key: {"noul": 0.99} for key in request["questions"]}}
        return {"status": "completed", "output": [{"content": [{"type": "output_text", "text": '{"candidates":[]}'}]}]}

    original_init = ProviderCache.__init__

    def mock_cache(self, directory, *, allow_paid=False, transport=None, config=None, asker=None):
        original_init(self, directory, allow_paid=allow_paid, transport=mocked_transport, config=config, asker=asker)

    monkeypatch.setattr(ProviderCache, "__init__", mock_cache)
    command = ["analyze", str(source), "--analysis", "memory"]
    live = runner.invoke(app, [*command, "--semantic", "live"])
    assert live.exit_code == 0, live.output
    assert json.loads(live.stdout)["memory"]["status"] == "complete"
    output = next((tmp_path / ".cache/agent-hotwash/memory").iterdir())
    first = json.loads((output / "memory.json").read_text())
    assert calls == ["jev", "openai"]
    replay = runner.invoke(app, [*command, "--semantic", "cached"])
    assert replay.exit_code == 0, replay.output
    second = json.loads((output / "memory.json").read_text())
    assert first["features"] == second["features"]
    assert first["candidates"] == second["candidates"]
    assert calls == ["jev", "openai"]
    assert second["calls"] == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stages"]["synthesis"]["resumed"]


@pytest.mark.parametrize("analysis", [None, "diagnostics"])
def test_diagnostic_analysis_never_runs_memory(tmp_path, monkeypatch, analysis):
    from agent_hotwash import memory_cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostics must not execute memory analysis")

    monkeypatch.setattr(memory_cli, "run_memory_analysis", forbidden)
    command = ["analyze", str(FIXTURES / "codex_native/rollout-fixture.jsonl"), "--semantic", "off"]
    if analysis:
        command.extend(["--analysis", analysis])
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["aggregate"]["total_traces"] == 1


def test_memory_only_skips_diagnostic_engine_and_respects_input_filters(tmp_path, monkeypatch):
    from agent_hotwash import cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Memory-only must not build diagnostic reports")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_build_report", forbidden)
    source = str(FIXTURES / "codex_native/rollout-fixture.jsonl")
    base = ["analyze", source, "--analysis", "memory", "--semantic", "off"]
    included = runner.invoke(app, [*base, "--since", "2026-01-01", "--model-family", "GPT-5.5"])
    assert included.exit_code == 0, included.output
    payload = json.loads(included.stdout)
    assert payload["evidence"]["sessions"] and payload["memory"]["status"] == "preprocessed"
    for selection in (["--since", "2027-01-01"], ["--until", "2025-12-31"], ["--model-family", "claude"]):
        excluded = runner.invoke(app, [*base, *selection])
        assert excluded.exit_code == 2, excluded.output
        assert excluded.stdout == ""


@pytest.mark.parametrize("fmt", ["html", "json", "table"])
@pytest.mark.parametrize("directory", [False, True])
def test_memory_only_output_file_or_directory(tmp_path, monkeypatch, fmt, directory):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "review"
    if directory:
        out.mkdir()
    result = runner.invoke(
        app,
        [
            "analyze",
            str(FIXTURES / "codex_native/rollout-fixture.jsonl"),
            "--analysis",
            "memory",
            "--semantic",
            "off",
            "--format",
            fmt,
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    target = out / {"html": "memory.html", "json": "memory.json", "table": "memory.md"}[fmt] if directory else out
    assert target.is_file()
    if fmt == "json":
        assert json.loads(target.read_text())["memory"]["status"] == "preprocessed"
    else:
        assert "memory" in target.read_text().lower()


@pytest.mark.parametrize("directory", [False, True])
def test_all_writes_separate_memory_report(tmp_path, monkeypatch, directory):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "review.json"
    if directory:
        out.mkdir()
    result = runner.invoke(
        app,
        [
            "analyze",
            str(FIXTURES / "claude_native/proj"),
            str(FIXTURES / "codebench/codex_run"),
            "--analysis",
            "all",
            "--semantic",
            "off",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    normal = out / "report.json" if directory else out
    memory = out / "memory.html" if directory else tmp_path / "review.memory.html"
    assert json.loads(normal.read_text())["aggregate"]["total_traces"] >= 2
    assert "<!doctype html>" in memory.read_text()
    assert result.stdout == ""


def test_all_keeps_stdout_diagnostics_and_partial_memory_report_on_cached_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent_hotwash import cli

    original = cli._build_report

    def deterministic(paths, options):
        from dataclasses import replace

        return original(paths, replace(options, semantic_mode="off"))

    monkeypatch.setattr(cli, "_build_report", deterministic)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(FIXTURES / "codex_native/rollout-fixture.jsonl"),
            "--analysis",
            "all",
            "--semantic",
            "cached",
        ],
    )
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["aggregate"]["total_traces"] == 1
    assert "wrote memory report" in result.stderr
    reports = list((tmp_path / ".cache/agent-hotwash/memory").glob("*/report.html"))
    assert len(reports) == 1 and "Incomplete audit" in reports[0].read_text()
    assert "--allow-paid" not in reports[0].read_text()
    assert "--semantic live" in reports[0].read_text()


@pytest.mark.parametrize(
    "extra",
    [
        ["--analysis", "invalid"],
        ["--analysis", "memory", "--format", "csv"],
        ["--analysis", "memory", "--format", "brief-json"],
        ["--guidance", "CLAUDE.md"],
        ["--memory"],
        ["--memory-only"],
    ],
)
def test_analysis_selector_rejects_invalid_or_conflicting_options(extra):
    result = runner.invoke(app, ["analyze", "placeholder", "--semantic", "off", *extra])
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("command", ["memory-audit", "memory-compare", "memory-render"])
def test_pipeline_stage_commands_are_removed(command):
    assert runner.invoke(app, [command, "--help"]).exit_code == 2
    assert command not in runner.invoke(app, ["--help"]).stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--fail-on", "high"],
        ["--no-detectors"],
        ["--full-html"],
        ["--allow-unredacted"],
        ["--jobs", "1"],
    ],
)
def test_memory_only_rejects_diagnostic_options(extra):
    result = runner.invoke(app, ["analyze", "placeholder", "--analysis", "memory", "--semantic", "off", *extra])
    assert result.exit_code == 2, result.output
    assert "requires diagnostics or all analysis" in result.output


@pytest.mark.parametrize("mode,extract,paid", [("off", False, False), ("cached", True, False), ("live", True, True)])
def test_memory_uses_shared_config_and_provider_permission(tmp_path, monkeypatch, mode, extract, paid):
    from agent_hotwash.memory import pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    config = tmp_path / "config.toml"
    config.write_text('[memory]\nmodel = "configured-model"\nretain_threshold = 0.7\n')
    seen = {}

    def audit(paths, output, **kwargs):
        seen.update(kwargs)
        return {"status": "preprocessed", "sessions": 1, "candidates": 0}

    monkeypatch.setattr(pipeline, "run_audit", audit)
    monkeypatch.setattr(
        pipeline,
        "load_saved",
        lambda output: (
            {"sessions": [], "records": [], "analytics": {}},
            {"status": "preprocessed", "candidates": []},
        ),
    )
    result = runner.invoke(
        app,
        [
            "analyze",
            "placeholder",
            "--analysis",
            "memory",
            "--semantic",
            mode,
            "--config",
            str(config),
            "--guidance",
            "CLAUDE.md",
            "--guidance",
            "SKILL.md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["config"].memory.model == "configured-model"
    assert seen["config"].memory.retain_threshold == 0.7
    assert seen["extract"] is extract and seen["allow_paid"] is paid
    assert seen["guidance_paths"] == [Path("CLAUDE.md"), Path("SKILL.md")]
    assert seen["run_options"].semantic_mode == mode


@pytest.mark.parametrize(
    "arguments",
    [
        ["memory-audit", "--help"],
        ["memory-compare", "--help"],
        ["memory-render", "--help"],
        ["analyze", "placeholder", "--analysis", "invalid"],
    ],
)
def test_process_entrypoint_returns_usage_exit_for_removed_or_invalid_commands(arguments, capsys):
    from agent_hotwash.cli import main

    assert main(arguments) == 2
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["json", "brief-json", "csv", "table", "html"])
def test_all_retains_diagnostic_formats_and_separate_memory_html(tmp_path, monkeypatch, fmt):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(FIXTURES / "codex_native/rollout-fixture.jsonl"),
            "--analysis",
            "all",
            "--semantic",
            "off",
            "--format",
            fmt,
        ],
    )
    assert result.exit_code == 0, result.output
    if fmt in {"json", "brief-json"}:
        assert "memory" not in json.loads(result.stdout)
    assert "wrote memory report" in result.stderr
    report = next((tmp_path / ".cache/agent-hotwash/memory").glob("*/report.html"))
    assert "<!doctype html>" in report.read_text()
