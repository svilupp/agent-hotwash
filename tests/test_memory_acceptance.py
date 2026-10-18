"""Independent integrated-stage acceptance, with providers kept deterministic."""

from __future__ import annotations

import json

import pytest

from agent_hotwash.config import Config, MemoryConfig
from agent_hotwash.memory.extraction import ProviderCache
from agent_hotwash.memory.persistence import atomic_text
from agent_hotwash.memory.pipeline import compare_saved, load_saved, run_audit


def corpus(tmp_path):
    source = tmp_path / "rollout-acceptance.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"type": "session_meta", "payload": {"id": "acceptance", "cwd": "/repo"}},
                {"type": "event_msg", "payload": {"type": "agent_message", "message": "I'll write a new parser."}},
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "No, reuse canonical adapters for future trace work.",
                    },
                },
            ]
        )
        + "\n"
    )
    return source


def provider(tmp_path, config=None, proposals=1):
    calls = []

    def transport(name, request):
        calls.append(name)
        if name == "jev":
            return {
                "answers": {key: {"noul": 0.1 if key.endswith("conflict") else 0.8} for key in request["questions"]}
            }
        records = json.loads(request["input"])["records"]
        user = next(row for row in records if row["role"] == "user")
        candidates = [
            {
                "category": "user_corrections",
                "statement": f"Reuse canonical adapters {index}.",
                "applicability": "This repository",
                "evidence_ids": [user["id"]],
                "paths": [],
                "intervention": "instruction",
                "draft": "Reuse canonical adapters.",
                "reason": "Explicit correction.",
            }
            for index in range(proposals)
        ]
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": candidates})}]}]}

    cache = ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config or Config())
    return cache, calls


def test_changed_guidance_and_threshold_reassessment_reuse_saved_synthesis(tmp_path):
    source = corpus(tmp_path)
    cache, calls = provider(tmp_path)
    audit = tmp_path / "audit"
    assert run_audit([source], audit, config=Config(), extract=True, cache=cache)["status"] == "complete"
    proposals = load_saved(audit)[1]["candidates"]
    guide = tmp_path / "CLAUDE.md"
    guide.write_text("Reuse canonical adapters.\n")
    comparison = tmp_path / "comparison"
    assert (
        compare_saved(audit, comparison, config=Config(), guidance_paths=[guide], cache=cache)["status"] == "complete"
    )
    before = len(calls)
    first = load_saved(comparison)[1]
    assert first["candidates"] == proposals
    assert next(iter(first["guidance_assessments"].values()))["guidance_status"] == "needs_review"
    lower = Config(memory=MemoryConfig(coverage_threshold=0.75))
    replay = ProviderCache(tmp_path / "calls", config=lower)
    assert compare_saved(audit, comparison, config=lower, guidance_paths=[guide], cache=replay)["status"] == "complete"
    second = load_saved(comparison)[1]
    assert next(iter(second["guidance_assessments"].values()))["guidance_status"] == "already_covered"
    assert len(calls) == before
    assert all(call["cache_hit"] for call in replay.calls)
    guide.write_text("Use shared adapters for all trace work.\n")
    assert (
        compare_saved(audit, comparison, config=Config(), guidance_paths=[guide], cache=cache)["status"] == "complete"
    )
    third = load_saved(comparison)[1]
    assert third["candidates"] == proposals
    assert first["guidance_assessments"] != third["guidance_assessments"]
    assert all(name == "jev" for name in calls[before:])


def test_stricter_synthesis_candidate_limit_cannot_resume_out_of_policy_proposals(tmp_path):
    source = corpus(tmp_path)
    audit = tmp_path / "audit"
    cache, _ = provider(tmp_path, proposals=2)
    assert run_audit([source], audit, config=Config(), extract=True, cache=cache)["status"] == "complete"
    assert len(load_saved(audit)[1]["candidates"]) == 2
    strict = Config(memory=MemoryConfig(max_candidates=1))
    replay = ProviderCache(tmp_path / "calls", config=strict)
    result = run_audit([source], audit, config=strict, extract=True, cache=replay)
    assert result["status"] == "blocked"
    assert load_saved(audit)[1]["failing_stage"] == "synthesis"


def test_saved_rendering_is_provider_free_and_validates_hashes(tmp_path, monkeypatch):
    from agent_hotwash.report.memory import render_memory_html

    cache, _ = provider(tmp_path)
    audit = tmp_path / "audit"
    run_audit([corpus(tmp_path)], audit, config=Config(), extract=True, cache=cache)

    def no_providers(*args, **kwargs):
        raise AssertionError("Rendering called a provider")

    monkeypatch.setattr(ProviderCache, "call", no_providers)
    evidence, result = load_saved(audit)
    html = render_memory_html(evidence, result)
    assert "Reuse canonical adapters" in html and "unchecked" in html
    features = audit / "features.json"
    features.write_text("[]")
    with pytest.raises(ValueError, match=r"(?i)(hash|corrupt)"):
        load_saved(audit)


def test_failed_atomic_replace_keeps_original_and_removes_temporary(tmp_path, monkeypatch):
    import agent_hotwash.memory.persistence as persistence

    target = tmp_path / "artifact.json"
    target.write_text("original")

    def interrupted(*args):
        raise OSError("simulated disk interruption")

    monkeypatch.setattr(persistence.os, "replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        atomic_text(target, "replacement")
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_stricter_comparison_pair_budget_cannot_resume_excess_pairs(tmp_path):
    audit = tmp_path / "audit"
    cache, _ = provider(tmp_path, proposals=2)
    run_audit([corpus(tmp_path)], audit, config=Config(), extract=True, cache=cache)
    guide = tmp_path / "rules.md"
    guide.write_text("Reuse canonical adapters.\n")
    output = tmp_path / "comparison"
    assert compare_saved(audit, output, config=Config(), guidance_paths=[guide], cache=cache)["status"] == "complete"
    strict = Config(memory=MemoryConfig(max_pairs=1))
    replay = ProviderCache(tmp_path / "calls", config=strict)
    result = compare_saved(audit, output, config=strict, guidance_paths=[guide], cache=replay)
    assert result["status"] == "blocked"
    _, blocked = load_saved(output)
    assert blocked["failing_stage"] == "comparison" and len(blocked["candidates"]) == 2
    assert replay.calls == []


@pytest.mark.parametrize("budget", [1, 100])
def test_impossible_screening_envelope_blocks_before_provider_calls(tmp_path, budget):
    config = Config(memory=MemoryConfig(screening_max_request_bytes=budget))
    cache, calls = provider(tmp_path, config=config)
    summary = run_audit([corpus(tmp_path)], tmp_path / "out", config=config, extract=True, cache=cache)
    assert summary["status"] == "blocked"
    assert calls == []
    assert load_saved(tmp_path / "out")[1]["failing_stage"] == "screening"
