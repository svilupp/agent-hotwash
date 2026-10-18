"""Memory policy boundaries and provider-free review presentation."""

from __future__ import annotations

import json
import stat

import pytest
from pydantic import ValidationError

from agent_hotwash.config import MemoryConfig, load_config
from agent_hotwash.report.memory import render_memory_html, render_memory_markdown


@pytest.mark.parametrize(
    "override",
    [
        {"retain_threshold": float("nan")},
        {"coverage_threshold": 1.01},
        {"comparison_max_questions": 0},
        {"guidance_chunk_chars": 500, "guidance_overlap_chars": 500},
        {"guidance_max_total_bytes": 100},
        {"synthesis_repair_reserve_bytes": 120000},
        {"uncertain_threshold": 0.8},
        {"tool_edge_chars": 1001},
    ],
)
def test_incompatible_policy_is_rejected_at_configuration_boundary(override):
    with pytest.raises(ValidationError):
        MemoryConfig.model_validate(override)


def test_memory_config_merges_without_changing_semantic_defaults(tmp_path):
    config_file = tmp_path / "hotwash.toml"
    config_file.write_text('[memory]\nretain_threshold = 0.7\nmodel = "alternate"\n')
    config = load_config(config_file)
    assert config.memory.retain_threshold == 0.7
    assert config.memory.model == "alternate"
    assert config.memory.screening_max_questions == 120
    assert config.semantic == load_config().semantic
    assert json.loads(config.model_dump_json())["memory"]["retain_threshold"] == 0.7


def test_saved_render_is_offline_and_replaces_report_with_private_file(tmp_path):
    from agent_hotwash.memory_cli import render_memory_analysis, write_memory_report

    destination = tmp_path / "review.html"
    destination.write_text("stale report")
    destination.chmod(0o644)
    text = render_memory_analysis(
        {"sessions": [], "records": [], "analytics": {}},
        {"status": "blocked", "error": "Comparison incomplete.", "candidates": []},
        "html",
    )
    write_memory_report(destination, text)
    assert "Comparison incomplete." in destination.read_text()
    assert "stale report" not in destination.read_text()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_offline_reports_separate_blocked_comparison_from_existing_drafts():
    traces = {
        "sessions": [
            {"session_id": "session", "source_path": None, "reasoning_visibility": "encrypted_only_unavailable"}
        ],
        "records": [
            {
                "id": "e1",
                "session_id": "session",
                "source_path": None,
                "line_number": None,
                "canonical_event_index": 3,
                "text": "<script>untrusted_trace()</script>",
                "independence_group_id": "family",
                "independence_confidence": "known",
            }
        ],
        "analytics": {"most_opened_files": []},
    }
    result = {
        "status": "blocked",
        "failed_stage": "comparison",
        "error": "Missing comparison answer.",
        "provider_usage": {"jev": {"requests": 0, "cache_hits": 2}},
        "candidates": [
            {
                "id": "p1",
                "category": "user_corrections",
                "statement": "Check the executable.",
                "reason": "Observed correction.",
                "applicability": "Source task only.",
                "intervention": "instruction",
                "supporting_sessions": 1,
                "draft": "<img src=x onerror=hostile()>",
                "evidence_ids": ["e1"],
            }
        ],
    }
    html = render_memory_html(traces, result)
    markdown = render_memory_markdown(traces, result)
    assert "<!doctype html>" in html
    assert "Incomplete audit" in html and "comparison" in html
    assert "Check the executable." in html and "Check the executable." in markdown
    assert "&lt;script&gt;untrusted_trace()&lt;/script&gt;" in html
    assert "<script>" not in html and "<img src=x" not in html
    assert "physical line unavailable" in html and "source line unavailable" in markdown
    assert "Known independent evidence groups: 1." in markdown
    assert "encrypted_only_unavailable" in html
    assert "jev network requests 0; native cache hits 2." in html
    assert "jev: 0 network requests; 2 native cache hits." in markdown
