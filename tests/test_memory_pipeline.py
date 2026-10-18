"""Stage recovery, semantic reuse and private persistence regressions."""

from __future__ import annotations

import json
import stat

import pytest

from agent_hotwash.config import Config, MemoryConfig, SemanticConfig
from agent_hotwash.memory.extraction import ProviderCache
from agent_hotwash.memory.persistence import atomic_text, audit_lock
from agent_hotwash.memory.pipeline import compare_saved, load_saved, run_audit


def trace(tmp_path):
    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"type": "session_meta", "payload": {"id": "session-1", "cwd": "/repo"}},
                {
                    "type": "event_msg",
                    "payload": {"type": "agent_message", "message": "I'll create a separate parser."},
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "No, reuse the existing adapter for future trace analysis.",
                    },
                },
            ]
        )
        + "\n"
    )
    return source


def provider(tmp_path, config=None):
    def transport(name, request):
        if name == "jev":
            return {
                "answers": {
                    qid: {"noul": 0.99 if qid.endswith("correction") or qid.endswith("reusable") else 0.01}
                    for qid in request["questions"]
                }
            }
        rows = json.loads(request["input"])["records"]
        user = next(row for row in rows if row["role"] == "user")
        candidate = {
            "category": "user_corrections",
            "statement": "Reuse canonical adapters.",
            "applicability": "Trace analysis in this repository",
            "evidence_ids": [user["id"]],
            "paths": [],
            "intervention": "instruction",
            "draft": "Reuse the existing adapter for trace analysis.",
            "reason": "Explicit user correction.",
        }
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": [candidate]})}]}]}

    return ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config or Config())


def test_preparation_and_atomic_private_artifacts(tmp_path):
    source = trace(tmp_path)
    output = tmp_path / "out"
    assert run_audit([source], output, config=Config())["status"] == "preprocessed"
    evidence, result = load_saved(output)
    assert evidence["records"] and result["status"] == "preprocessed"
    assert (output / "report.html").is_file()
    assert stat.S_IMODE((output / "memory.json").stat().st_mode) == 0o600
    assert json.loads((output / "manifest.json").read_text())["stages"]["screening"]["status"] == "skipped"


def test_resume_and_guidance_change_preserve_synthesis(tmp_path):
    source = trace(tmp_path)
    output = tmp_path / "out"
    cache = provider(tmp_path)
    assert run_audit([source], output, config=Config(), extract=True, cache=cache)["status"] == "complete"
    first_calls = len(cache.calls)
    replay = ProviderCache(tmp_path / "calls", config=Config())
    assert run_audit([source], output, config=Config(), extract=True, cache=replay)["status"] == "complete"
    assert replay.calls == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stages"]["screening"]["resumed"] and manifest["stages"]["synthesis"]["resumed"]
    guidance = tmp_path / "CLAUDE.md"
    guidance.write_text("Reuse the existing adapter for trace analysis.\n")
    assert (
        compare_saved(output, tmp_path / "compare", config=Config(), guidance_paths=[guidance], cache=cache)["status"]
        == "complete"
    )
    assert len(cache.calls) > first_calls and all(c["provider"] == "jev" for c in cache.calls[first_calls:])
    _, compared = load_saved(tmp_path / "compare")
    assert compared["candidates"] == load_saved(output)[1]["candidates"]


def test_failure_replaces_stale_success_and_preserves_proposals(tmp_path):
    source = trace(tmp_path)
    output = tmp_path / "out"
    config = Config()
    cache = provider(tmp_path)
    assert run_audit([source], output, config=config, extract=True, cache=cache)["status"] == "complete"
    guidance = tmp_path / "guidance.md"
    guidance.write_text("A different rule.")
    blocked = run_audit(
        [source],
        output,
        config=config,
        extract=True,
        guidance_paths=[guidance],
        cache=ProviderCache(tmp_path / "empty", config=config),
    )
    assert blocked["status"] == "blocked"
    _, result = load_saved(output)
    assert result["candidates"] and result["failing_stage"] == "comparison"
    assert "blocked" in (output / "report.html").read_text().lower()


def test_lock_versions_and_corrupt_stage_are_explicit(tmp_path):
    output = tmp_path / "out"
    with audit_lock(output), pytest.raises(RuntimeError, match="already being written"), audit_lock(output):
        pass
    source = trace(tmp_path)
    cache = provider(tmp_path)
    run_audit([source], output, config=Config(), extract=True, cache=cache)
    (output / "features.json").write_text("[]")
    summary = run_audit([source], output, config=Config(), extract=True, cache=cache)
    assert summary["status"] == "blocked" and "Corrupt saved screening" in summary["error"]
    prepared = json.loads((output / "prepared.json").read_text())
    prepared.pop("schema_version")
    (output / "prepared.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="Unsupported"):
        load_saved(output)
    symlink = tmp_path / "link"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="symbolic"):
        atomic_text(symlink, "bad")


def test_configured_screening_bounds_and_zero_repair(tmp_path):
    from agent_hotwash.memory.extraction import canonical, feature_requests, propose_candidates

    rows = [
        {"id": "r1", "session_id": "s1", "role": "assistant", "kind": "reasoning", "text": '界\\"' * 20000, "paths": []}
    ]
    requests = list(feature_requests(rows, max_questions=3, max_request_bytes=12000))
    assert all(len(canonical(request)) <= 12000 for request, _ in requests)
    config = Config(memory=MemoryConfig(synthesis_max_repairs=0), semantic=SemanticConfig(max_retries=0))
    cache = ProviderCache(tmp_path, allow_paid=True, transport=lambda *_: {"output": []}, config=config)
    with pytest.raises(ValueError, match="no structured output"):
        propose_candidates(cache, "gpt-6.1-sol", {"records": []}, [])
    assert len(cache.calls) == 1


def test_shared_native_question_cache_serves_new_audit_without_paid_calls(tmp_path):
    import httpx2

    from agent_hotwash.memory.extraction import feature_requests
    from agent_hotwash.semantic.client import SystemOneAsker

    requests = []

    def transport(request):
        request.read()
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(
            200,
            json={"model": "jev-1.14.0", "answers": {key: {"type": "noul", "noul": 0.8} for key in body["questions"]}},
        )

    native = tmp_path / "native"
    config = Config(semantic=SemanticConfig(cache_dir=str(native), max_retries=0))
    asker = SystemOneAsker(
        "jev-1.14.0", native, mode="live", strict_answers=True, transport_override=httpx2.MockTransport(transport)
    )
    cache = ProviderCache(tmp_path / "first", allow_paid=True, config=config, asker=asker)
    row = {
        "id": "r1",
        "session_id": "s1",
        "role": "assistant",
        "kind": "reasoning",
        "text": "The repository uses shared adapters.",
        "paths": [],
    }
    request, _ = next(feature_requests([row]))
    cache.call("jev", request)
    replay = ProviderCache(tmp_path / "second", config=config)
    response = replay.call("jev", request)
    assert len(requests) == 1
    assert response["native_usage"]["requests"] == 0
    assert response["native_usage"]["cache_hits"] == 3
    assert replay.calls[0]["cache_hit"] is False
    assert replay.calls[0]["native_usage"]["requests"] == 0
    changed = {**request, "state": {"r0": {"text": "An uncached finding", "role": "assistant", "kind": "reasoning"}}}
    with pytest.raises(RuntimeError, match="cache miss"):
        replay.call("jev", changed)
    assert len(requests) == 1
    corrupt = next((native / "r1" / "strict-v1").rglob("*.json"))
    corrupt.write_text("{broken")
    retry = ProviderCache(tmp_path / "third", allow_paid=True, config=config, asker=asker)
    with pytest.raises(RuntimeError, match="Corrupt native semantic cache"):
        retry.call("jev", request)
    assert corrupt.read_text() == "{broken" and len(requests) == 1


def test_shared_client_strict_unknown_answer_and_wire_envelope(tmp_path):
    import httpx2

    from agent_hotwash.semantic.client import SystemOneAsker

    calls = []

    def transport(request):
        request.read()
        body = json.loads(request.content)
        calls.append(body)
        answers = {key: {"type": "noul", "noul": 0.8} for key in body["questions"]}
        answers["unknown"] = {"type": "noul", "noul": 0.8}
        return httpx2.Response(200, json={"model": "jev-1.14.0", "answers": answers})

    question = {
        "q": {"type": "noul", "instructions": {"question": "Is the target a correction?", "inspect": ["`target.text`"]}}
    }
    asker = SystemOneAsker(
        "jev-1.14.0",
        tmp_path / "native",
        mode="live",
        strict_answers=True,
        transport_override=httpx2.MockTransport(transport),
    )
    with pytest.raises(ValueError, match="unexpected"):
        asker.ask({"target": {"text": "No, reuse it."}}, question)
    bounded = SystemOneAsker(
        "jev-1.14.0",
        tmp_path / "bounded",
        mode="live",
        strict_answers=True,
        max_request_bytes=100,
        transport_override=httpx2.MockTransport(transport),
    )
    with pytest.raises(ValueError, match="serialized byte limit"):
        bounded.ask({"target": {"text": "界" * 1000}}, question)
    assert len(calls) == 1


def test_screening_ten_thousand_records_batched_without_loss():
    from agent_hotwash.memory.extraction import canonical, feature_requests

    rows = [
        {
            "id": f"r{i}",
            "session_id": "session",
            "role": "assistant",
            "kind": "reasoning",
            "text": "Confirmed a repository constraint.",
            "paths": [],
        }
        for i in range(10000)
    ]
    mapping = []
    max_bytes = 0
    count = 0
    for request, refs in feature_requests(rows):
        assert len(request["questions"]) <= 120
        max_bytes = max(max_bytes, len(canonical(request)))
        count += 1
        mapping.extend(refs.values())
    assert max_bytes <= 56000
    assert len(mapping) == 30000
    assert len({record_id for record_id, _ in mapping}) == 10000
    assert count < 1000


@pytest.mark.parametrize("bad", [True, None, float("inf"), -0.1, 1.1])
def test_shared_strict_probability_rejects_before_sdk_coercion(tmp_path, bad):
    import httpx2

    from agent_hotwash.semantic.client import SystemOneAsker

    def transport(request):
        request.read()
        body = json.loads(request.content)
        # Serialize Infinity deliberately: a tolerant SDK may normalize/coerce it.
        payload = {"model": "jev-1.14.0", "answers": {key: {"type": "noul", "noul": bad} for key in body["questions"]}}
        return httpx2.Response(200, content=json.dumps(payload).encode())

    asker = SystemOneAsker(
        "jev-1.14.0", tmp_path, mode="live", strict_answers=True, transport_override=httpx2.MockTransport(transport)
    )
    question = {"q": {"type": "noul", "instructions": {"question": "Correction?", "inspect": ["`target.text`"]}}}
    with pytest.raises(ValueError, match="finite numeric"):
        asker.ask({"target": {"text": "No, reuse it."}}, question)
    assert not list((tmp_path / "r1" / "strict-v1").rglob("*.json"))


@pytest.mark.parametrize("name", ["prepared.json", "memory.json"])
def test_saved_artifact_arrays_rejected_with_readable_error(tmp_path, name):
    output = tmp_path / "out"
    run_audit([trace(tmp_path)], output, config=Config())
    (output / name).write_text("[]")
    with pytest.raises(ValueError, match="must be JSON objects"):
        load_saved(output)


def test_blocked_report_exposes_the_failing_stage(tmp_path):
    output = tmp_path / "out"
    config = Config(semantic=SemanticConfig(cache_dir=str(tmp_path / "native")))
    summary = run_audit([trace(tmp_path)], output, config=config, extract=True)
    assert summary["status"] == "blocked"
    assert "screening" in (output / "report.md").read_text()
    assert "screening" in (output / "report.html").read_text()


@pytest.mark.parametrize("text", ["visible", "界" * 1000, '\\"\n' * 1000])
def test_impossibly_small_screening_budget_terminates_in_subprocess(text):
    import subprocess
    import sys

    code = (
        "from agent_hotwash.memory.extraction import feature_requests\nrow="
        + repr({"id": "r1", "session_id": "s", "role": "assistant", "kind": "reasoning", "text": text})
        + "\ntry:\n list(feature_requests([row],max_request_bytes=1))\n"
        + "except ValueError as exc:\n print(exc)\nelse:\n"
        + " raise AssertionError('expected bounded rejection')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], timeout=3, capture_output=True, text=True)
    assert result.returncode == 0 and "byte limit" in result.stdout


def test_tiny_screening_budget_blocks_with_manifest_without_provider_call(tmp_path):
    config = Config(memory=MemoryConfig(screening_max_request_bytes=1))
    cache = provider(tmp_path, config)
    summary = run_audit([trace(tmp_path)], tmp_path / "out", config=config, extract=True, cache=cache)
    assert summary["status"] == "blocked" and cache.calls == []
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stages"]["screening"]["status"] == "blocked"


@pytest.mark.parametrize("broken", [b"{", b"[]", b"\xff"])
def test_corrupt_manifest_preserved_and_old_success_replaced(tmp_path, broken):
    output = tmp_path / "out"
    source = trace(tmp_path)
    run_audit([source], output, config=Config())
    (output / "manifest.json").write_bytes(broken)
    with pytest.raises(ValueError, match="preserved for inspection"):
        run_audit([source], output, config=Config())
    assert next(output.glob(".manifest-corrupt-*.json")).read_bytes() == broken
    assert json.loads((output / "memory.json").read_text())["status"] == "blocked"
    assert "blocked" in (output / "report.md").read_text()
    assert "blocked" in (output / "report.html").read_text()
    assert json.loads((output / "manifest.json").read_text())["stages"]["preparation"]["status"] == "blocked"
