import json
import subprocess
import sys

from agent_hotwash.memory.extraction import ProviderCache, evidence_batches, extract_memory
from agent_hotwash.memory.guidance import prepare_guidance


def fixture():
    agent = {
        "id": "agent",
        "session_id": "s1",
        "role": "assistant",
        "kind": "message",
        "text": "I will put all learning in MEMORY.md.",
        "paths": [],
        "line_number": 1,
    }
    user = {
        "id": "user",
        "session_id": "s1",
        "role": "user",
        "kind": "message",
        "text": "No, use mem instead.",
        "paths": [],
        "line_number": 2,
    }
    return {
        "records": [agent, user],
        "sessions": [{"session_id": "s1", "source_path": "trace.jsonl"}],
        "analytics": {"most_opened_files": []},
    }


def transport(calls):
    def respond(provider, request):
        calls.append((provider, request))
        if provider == "jev":
            return {
                "answers": {key: {"noul": 0.01 if key.endswith("_conflict") else 0.98} for key in request["questions"]}
            }
        candidate = {
            "category": "user_corrections",
            "statement": "Use mem for durable knowledge",
            "applicability": "This repository",
            "evidence_ids": ["user"],
            "paths": [],
            "intervention": "instruction",
            "draft": "Use mem for durable knowledge.",
            "reason": "The user corrected the storage choice.",
        }
        evidence = json.loads(request["input"])
        correction = next(r for r in evidence["records"] if r["id"] == "user")
        assert correction["preceding_agent_context"]["id"] == "agent"
        return {
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": [candidate]})}]}],
        }

    return respond


def test_selected_guidance_pipeline_reports_citations_and_replays(tmp_path):
    source = tmp_path / "CLAUDE.md"
    source.write_text("Use mem for durable knowledge.\n")
    snapshot = prepare_guidance([source])
    calls = []
    output = tmp_path / "run"
    cache_dir = tmp_path / "calls"
    first = extract_memory(
        fixture(),
        output,
        guidance=snapshot,
        cache=ProviderCache(cache_dir, allow_paid=True, transport=transport(calls)),
    )
    assert first["status"] == "complete"
    saved = json.loads((output / "memory.json").read_text())
    assessment = next(iter(saved["proposal_assessments"].values()))
    assert assessment["guidance_status"] == "already_covered"
    assert assessment["novelty_status"] == "not_established"
    assert assessment["context_evidence_ids"] == ["agent"]
    assert json.loads((output / "guidance.json").read_text()) == snapshot
    report = (output / "report.md").read_text()
    assert "selected files only" in report and "already_covered" in report
    assert snapshot["chunks"][0]["id"] in report and snapshot["files"][0]["source_sha256"] in report
    assert source.read_text() == "Use mem for durable knowledge.\n"
    assert len(calls) == 3
    replay = extract_memory(fixture(), tmp_path / "replay", guidance=snapshot, cache=ProviderCache(cache_dir))
    assert replay["status"] == "complete"
    replay_saved = json.loads((tmp_path / "replay" / "memory.json").read_text())
    assert replay_saved["proposal_assessments"] == saved["proposal_assessments"]
    assert all(call["cache_hit"] for call in replay_saved["calls"])
    assert len(calls) == 3


def test_missing_guidance_keeps_unchecked_status(tmp_path):
    calls = []
    extract_memory(
        fixture(),
        tmp_path / "run",
        cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport(calls)),
    )
    saved = json.loads((tmp_path / "run" / "memory.json").read_text())
    assert len(calls) == 2
    assert saved["guidance_comparison"]["status"] == "not_requested"
    assert next(iter(saved["proposal_assessments"].values()))["novelty_status"] == "not_established"


def test_bad_guidance_answer_blocks_preserves_generated_draft(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("Use mem for durable knowledge.")
    calls = []
    good = transport(calls)

    def bad(provider, request):
        if provider == "jev" and any(key.endswith("_coverage") for key in request["questions"]):
            return {"answers": {}}
        return good(provider, request)

    output = tmp_path / "run"
    result = extract_memory(
        fixture(),
        output,
        guidance=prepare_guidance([source]),
        cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=bad),
    )
    assert result["status"] == "blocked" and "guidance JEV answers" in result["error"]
    saved = json.loads((output / "memory.json").read_text())
    assert len(saved["candidates"]) == 1
    assert "Status: blocked" in (output / "report.md").read_text()


def test_cli_preprocess_snapshots_selected_guidance_without_model_calls(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "guidance-cli", "cwd": "/repo"}})
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "No, use mem."}],
                },
            }
        )
        + "\n"
    )
    guide = tmp_path / "guide.md"
    guide.write_text("Use mem.")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_hotwash.cli",
            "analyze",
            "--analysis",
            "memory",
            "--semantic",
            "off",
            str(trace),
            "--guidance",
            str(guide),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["memory"]["status"] == "preprocessed"
    output = next((tmp_path / ".cache/agent-hotwash/memory").iterdir())
    assert json.loads((output / "guidance.json").read_text())["files"][0]["path"] == str(guide)
    assert not (output / "calls").exists()


def test_correction_context_kept_even_if_antecedent_was_not_retained():
    traces = fixture()
    features = [{"record_id": "agent", "retained": False}, {"record_id": "user", "retained": True}]
    batches = list(evidence_batches(traces["records"], features))
    assert len(batches[0]) == 1
    assert batches[0][0]["preceding_agent_context"]["evidence_use"] == "interpretation_only"
