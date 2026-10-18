import json

import pytest

from agent_hotwash.memory.extraction import (
    ProviderCache,
    decode_candidates,
    eligible,
    extract_memory,
    feature_requests,
    screen_features,
)


def record(id="r1", role="user", session="s1", text="No, use the existing navigation guide."):
    return {
        "id": id,
        "role": role,
        "kind": "message",
        "session_id": session,
        "text": text,
        "paths": ["docs/guide.md"],
        "line_number": 1,
    }


def traces(records):
    return {
        "records": records,
        "sessions": [{"session_id": "s1", "source_path": "trace.jsonl"}],
        "analytics": {"most_opened_files": []},
    }


def test_batch_independent_features_role_boundary_and_uncertainty(tmp_path):
    records = [record(), record("r2", "assistant"), record("r3", "tool")]
    requests = list(feature_requests(records))
    assert len(requests) == 1
    _request, mapping = requests[0]
    assert len(mapping) == 8
    assert ("r1", "correction") in mapping.values()
    assert ("r2", "correction") not in mapping.values()
    cache = ProviderCache(
        tmp_path, allow_paid=True, transport=lambda p, r: {"answers": {k: {"noul": 0.5} for k in r["questions"]}}
    )
    features = screen_features(traces(records), cache)
    assert all(f["retained"] for f in features)
    assert features[1]["correction"] is None
    assert not eligible(record(text="# AGENTS.md instructions for /project"))


def test_cache_replay_identity_and_paid_boundary(tmp_path):
    calls = []

    def transport(p, r):
        calls.append(p)
        return {"answers": {k: {"noul": 0.01} for k in r["questions"]}}

    cache = ProviderCache(tmp_path, allow_paid=True, transport=transport)
    first = screen_features(traces([record()]), cache)
    second = screen_features(traces([record()]), ProviderCache(tmp_path))
    assert first == second and calls == ["jev"]
    with pytest.raises(RuntimeError, match="cache miss"):
        screen_features(traces([record(text="changed")]), ProviderCache(tmp_path))


def candidate_response(category="user_corrections", refs=None):
    candidate = {
        "category": category,
        "statement": "Use navigation guidance",
        "applicability": "Repository research",
        "evidence_ids": refs or ["r1"],
        "paths": ["docs/guide.md"],
        "intervention": "navigation_pointer",
        "draft": "Read docs/guide.md before research.",
        "reason": "The user corrected the search approach.",
    }
    return {
        "status": "completed",
        "output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": [candidate]})}]}],
    }


def test_proposal_evidence_validation():
    assert decode_candidates(candidate_response(), [record()])[0]["supporting_sessions"] == 1
    with pytest.raises(ValueError, match="unknown"):
        decode_candidates(candidate_response(refs=["invented"]), [record()])
    with pytest.raises(ValueError, match=r"evidence\.records"):
        decode_candidates(candidate_response(refs=["invented"]), [record()], evidence_collection="records")
    with pytest.raises(ValueError, match=r"evidence\.sources"):
        decode_candidates(candidate_response(refs=["invented"]), [record()], evidence_collection="sources")
    with pytest.raises(ValueError, match="pure user"):
        decode_candidates(candidate_response(), [record(role="assistant")])
    with pytest.raises(ValueError, match="independent sessions"):
        decode_candidates(candidate_response(category="repeated_discovery"), [record()])
    with pytest.raises(ValueError, match="did not complete"):
        decode_candidates({"status": "incomplete"}, [record()])


def test_pipeline_outputs_and_explicit_blocked_mode(tmp_path):
    blocked = extract_memory(traces([record()]), tmp_path / "blocked")
    assert blocked["status"] == "blocked" and "cache miss" in blocked["error"]

    def transport(provider, request):
        if provider == "jev":
            return {"answers": {k: {"noul": 0.95} for k in request["questions"]}}
        return candidate_response()

    result = extract_memory(
        traces([record()]),
        tmp_path / "run",
        cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport),
    )
    assert result["status"] == "complete" and result["candidates"] == 1
    report = (tmp_path / "run" / "report.md").read_text()
    assert "trace.jsonl#L1" in report and "Draft:" in report
    assert json.loads((tmp_path / "run" / "memory.json").read_text())["calls"][0]["provider"] == "jev"


def test_invalid_noul_rejected(tmp_path):
    cache = ProviderCache(
        tmp_path, allow_paid=True, transport=lambda p, r: {"answers": {k: {"noul": True} for k in r["questions"]}}
    )
    with pytest.raises(ValueError, match="invalid Noul"):
        screen_features(traces([record()]), cache)


@pytest.mark.parametrize("bad", [None, [], "bad"])
def test_malformed_jev_answers_write_blocked_report(tmp_path, bad):
    cache = ProviderCache(
        tmp_path / "calls", allow_paid=True, transport=lambda p, r: {"answers": dict.fromkeys(r["questions"], bad)}
    )
    assert extract_memory(traces([record()]), tmp_path / "out", cache=cache)["status"] == "blocked"


def test_top_level_response_and_schema_errors_block(tmp_path):
    def transport(provider, request):
        if provider == "jev":
            return {"answers": {k: {"noul": 0.95} for k in request["questions"]}}
        return {"output": [{"content": [{"type": "output_text", "text": '{"candidates":[{}]}'}]}]}

    result = extract_memory(
        traces([record()]),
        tmp_path / "out",
        cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport),
    )
    assert result["status"] == "blocked" and "schema validation" in result["error"]
    with pytest.raises(ValueError, match="must be an object"):
        ProviderCache(tmp_path / "other", allow_paid=True, transport=lambda p, r: []).call("jev", {})


def test_read_context_preserved_and_user_context_bounded(tmp_path):
    read = record("read", "tool")
    read.update(kind="tool_call")
    assistant = record("a", "assistant", text="A" * 10000)
    user = record("u", text="Manifests are okay.")
    requests = list(feature_requests([assistant, user]))
    assert len(requests[0][0]["state"]["r1"]["preceding_agent"]) < 2100
    cache = ProviderCache(
        tmp_path, allow_paid=True, transport=lambda p, r: {"answers": {k: {"noul": 0.01} for k in r["questions"]}}
    )
    output = screen_features(traces([read, user]), cache)
    assert output[0]["retained"] and output[0]["status"] == "read_context"


def test_native_http_transport_auth_and_replay(tmp_path, monkeypatch):
    import httpx2 as httpx

    from agent_hotwash.config import Config, SemanticConfig
    from agent_hotwash.semantic.client import SystemOneAsker

    calls = []

    def respond(request):
        calls.append(request)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test"
        body = json.loads(request.content)
        assert body["model"] == "jev-1.14.0"
        assert all(q["type"] == "noul" for q in body["questions"].values())
        return httpx.Response(
            200, json={"model": "jev-1.14.0", "answers": {k: {"type": "noul", "noul": 0.99} for k in body["questions"]}}
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential")
    config = Config(semantic=SemanticConfig(cache_dir=str(tmp_path / "native"), max_retries=0))
    asker = SystemOneAsker(
        "jev-1.14.0",
        tmp_path / "native",
        mode="live",
        transport_override=httpx.MockTransport(respond),
        strict_answers=True,
    )
    cache = ProviderCache(tmp_path, allow_paid=True, config=config, asker=asker)
    first = screen_features(traces([record()]), cache)
    assert screen_features(traces([record()]), ProviderCache(tmp_path)) == first
    assert len(calls) == 1

    def limited(request):
        return httpx.Response(429, json={"error": "rate limited"})

    limited_asker = SystemOneAsker(
        "jev-1.14.0",
        tmp_path / "failed-native",
        mode="live",
        max_retries=0,
        transport_override=httpx.MockTransport(limited),
        strict_answers=True,
    )
    with pytest.raises(RuntimeError, match="TypeSafeRateLimitError"):
        ProviderCache(tmp_path / "failed", allow_paid=True, asker=limited_asker).call(
            "jev", {"model": "jev-1.14.0", "state": {}, "questions": {"q": {"type": "noul", "instructions": "Test"}}}
        )
    assert not list((tmp_path / "failed").glob("*.json"))


def test_recurrence_requires_known_sessions_and_evidence_paths():
    a = record("r1", "assistant", "s1")
    b = record("r2", "assistant", "s2")
    response = candidate_response(category="repeated_reading", refs=["r1", "r2"])
    with pytest.raises(ValueError, match="independent sessions"):
        decode_candidates(response, [a, b])
    a["independent_session_known"] = b["independent_session_known"] = True
    with pytest.raises(ValueError, match="path read"):
        decode_candidates(response, [a, b])
    a["kind"] = b["kind"] = "tool_call"
    assert decode_candidates(response, [a, b])[0]["supporting_sessions"] == 2


def test_cli_prepare_success_and_extraction_blocked_exit(tmp_path):
    import subprocess
    import sys

    source = tmp_path / "trace.jsonl"
    source.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "cli-session", "cwd": "/repo"}})
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "No, inspect the existing guide."},
            }
        )
        + "\n"
    )
    command = [
        sys.executable,
        "-m",
        "agent_hotwash.cli",
        "analyze",
        str(source),
        "--analysis",
        "memory",
    ]
    prepared = subprocess.run([*command, "--semantic", "off"], capture_output=True, text=True, cwd=tmp_path)
    assert prepared.returncode == 0 and json.loads(prepared.stdout)["memory"]["status"] == "preprocessed"
    blocked = subprocess.run([*command, "--semantic", "cached"], capture_output=True, text=True, cwd=tmp_path)
    assert blocked.returncode == 1 and json.loads(blocked.stdout)["memory"]["status"] == "blocked"
    assert list((tmp_path / ".cache/agent-hotwash/memory").glob("*/report.md"))


def test_bounded_provider_repair_keeps_original_evidence_and_guards(tmp_path):
    from agent_hotwash.memory.extraction import propose_candidates

    calls = []

    def transport(provider, request):
        calls.append(request)
        response = candidate_response()
        value = json.loads(response["output"][0]["content"][0]["text"])
        if len(calls) == 1:
            value["candidates"][0]["paths"] = ["uncited.md"]
        else:
            repair = json.loads(request["input"])
            assert repair["evidence"] == {"records": [record()]}
            assert repair["repair"]["validation_error"].startswith("Memory proposal names an uncited path")
            assert "uncited.md" in repair["repair"]["validation_error"]
            value["candidates"][0]["paths"] = []
        response["output"][0]["content"][0]["text"] = json.dumps(value)
        return response

    cache = ProviderCache(tmp_path, allow_paid=True, transport=transport)
    candidates = propose_candidates(cache, "gpt-6.1-sol", {"records": [record()]}, [record()])
    assert len(candidates) == 1 and candidates[0]["paths"] == [] and len(calls) == 2
    assert len(cache.validation_events) == 1


def test_configurable_retention_boundary(tmp_path):
    cache = ProviderCache(
        tmp_path, allow_paid=True, transport=lambda p, r: {"answers": {k: {"noul": 0.49} for k in r["questions"]}}
    )
    assert not screen_features(traces([record()]), cache)[0]["retained"]
    assert screen_features(traces([record()]), cache, retain_threshold=0.3)[0]["retained"]
    with pytest.raises(ValueError, match="probability"):
        screen_features(traces([record()]), cache, retain_threshold=float("nan"))


def test_single_session_scope_and_context_are_separate_from_candidate():
    import copy

    from agent_hotwash.memory.extraction import assess_proposal_scope

    assistant = record("before", "assistant")
    user = record()
    candidate = decode_candidates(candidate_response(), [user])[0]
    original = copy.deepcopy(candidate)
    assessment = assess_proposal_scope([candidate], [assistant, user])[candidate["id"]]
    assert assessment["scope_status"] == "source_task_only"
    assert assessment["context_evidence_ids"] == ["before"]
    assert candidate == original and candidate["evidence_ids"] == ["r1"]


def test_duplicate_proposals_enter_bounded_repair(tmp_path):
    from agent_hotwash.memory.extraction import propose_candidates

    response = candidate_response()
    part = response["output"][0]["content"][0]
    candidate = json.loads(part["text"])["candidates"][0]
    part["text"] = json.dumps({"candidates": [candidate, candidate]})
    calls = []

    def transport(provider, request):
        calls.append(request)
        return response if len(calls) == 1 else candidate_response()

    cache = ProviderCache(tmp_path, allow_paid=True, transport=transport)
    result = propose_candidates(cache, "test", {"records": [record()]}, [record()])
    assert len(result) == 1
    assert len(calls) == 2
    assert "duplicate proposal" in cache.validation_events[0]["error"].lower()
