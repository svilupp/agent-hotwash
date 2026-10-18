"""Corpus consolidation stays bounded while preserving cited evidence and cache reuse."""

import json

import pytest

from agent_hotwash.config import Config, MemoryConfig
from agent_hotwash.memory import extraction as ex
from agent_hotwash.memory.pipeline import _synthesize


def records(count=6):
    return [
        {
            "id": f"source-{index}",
            "session_id": f"session-{index}",
            "independent_session_known": True,
            "role": "user",
            "kind": "message",
            "text": f"No, preserve schema {index}. " + "x" * 65000,
            "paths": [],
            "source_path": f"trace-{index}.jsonl",
            "line_number": index + 1,
        }
        for index in range(count)
    ]


def candidate(row, draft_size):
    return {
        "category": "user_corrections",
        "statement": "Preserve " + row["id"],
        "applicability": "The existing schema",
        "evidence_ids": [row["id"]],
        "paths": [],
        "intervention": "instruction",
        "draft": "Preserve the schema. " + "x" * draft_size,
        "reason": "Explicit correction",
    }


def cache_for(tmp_path, requests, *, draft_size=0):
    config = Config(memory=MemoryConfig(max_candidate_chars=200000))

    def transport(provider, request):
        assert provider == "openai"
        assert len(ex.canonical(request)) <= config.memory.synthesis_max_request_bytes
        data = json.loads(request["input"])
        requests.append(data)
        proposed = data.get("candidates")
        if proposed is None:
            proposed = [candidate(row, draft_size) for row in data["records"]]
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": proposed})}]}]}

    return ex.ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config)


def test_full_corpus_uses_bounded_groups_without_recursive_no_progress_and_replays(tmp_path):
    source = records()
    features = [{"record_id": row["id"], "retained": True} for row in source]
    requests = []
    cache = cache_for(tmp_path, requests)
    result = _synthesize({"records": source}, features, cache, cache.config.memory.model)
    final_requests = [data for data in requests if "sources" in data]
    assert len(requests) == 7
    assert len(final_requests) == 1
    assert len(result["candidates"]) == 6
    assert {ref for item in result["candidates"] for ref in item["evidence_ids"]} == {row["id"] for row in source}
    by_id = {row["id"]: row for row in source}
    for data in final_requests:
        assert {row["id"] for row in data["sources"]} == {
            ref for item in data["candidates"] for ref in item["evidence_ids"]
        }
        for row in data["sources"]:
            assert row["source_record_sha256"] == ex.digest(by_id[row["id"]])
            assert "text" not in row
            assert row["text_omitted"]
    summary = result["synthesis_evidence"]["consolidation"]
    assert summary["level_group_counts"] == [1]
    assert summary["input_candidate_count"] == 6
    assert summary["global"] is True
    assert summary["cross_group_recurrence_evaluated"] is True
    assert summary["source_text_compacted"] is True
    replay = ex.ProviderCache(tmp_path / "calls", config=cache.config)
    assert _synthesize({"records": source}, features, replay, cache.config.memory.model) == result
    assert len(replay.calls) == 7
    assert all(call["cache_hit"] for call in replay.calls)
    assert len(requests) == 7


def test_oversized_single_candidate_fails_explicitly_before_final_transport(tmp_path):
    source = records(2)
    requests = []
    cache = cache_for(tmp_path, requests, draft_size=110000)
    features = [{"record_id": row["id"], "retained": True} for row in source]
    with pytest.raises(ValueError, match="candidate and its citation metadata"):
        _synthesize({"records": source}, features, cache, cache.config.memory.model)
    assert len(requests) == 2
    assert all("records" in data for data in requests)


def test_small_consolidation_keeps_global_semantics_and_exact_dedup(tmp_path, monkeypatch):
    source = records(1)
    source[0]["text"] = "No, preserve the existing schema."
    monkeypatch.setattr(ex, "evidence_batches", lambda *args, **kwargs: [[source[0]], [source[0]]])
    requests = []
    cache = cache_for(tmp_path, requests)
    result = _synthesize({"records": source}, [], cache, cache.config.memory.model)
    assert len(result["candidates"]) == 1
    assert result["synthesis_evidence"]["consolidation"]["global"] is True
    assert result["synthesis_evidence"]["consolidation"]["level_group_counts"] == [1]
    assert requests[-1]["sources"] == source


def test_global_final_validates_recurrence_against_original_independence_groups(tmp_path):
    source = records()
    for index, row in enumerate(source):
        row.update(
            text="Investigated the shared schema. " + "x" * 30000,
            role="assistant",
            kind="reasoning",
            independence_confidence="known",
            independence_group_id=f"family-{index}",
        )
    requests = []
    config = Config()

    def transport(provider, request):
        data = json.loads(request["input"])
        requests.append(data)
        proposed = data.get("candidates")
        if proposed is None:
            proposed = [candidate(data["records"][0], 0)]
            proposed[0].update(category="repeated_discovery", evidence_ids=[row["id"] for row in data["records"]])
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": proposed})}]}]}

    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config)
    features = [{"record_id": row["id"], "retained": True} for row in source]
    result = _synthesize({"records": source}, features, cache, config.memory.model)
    assert len(result["candidates"]) == 2
    groups = [set(item["supporting_independence_groups"]) for item in result["candidates"]]
    assert groups == [{"family-0", "family-1", "family-2"}, {"family-3", "family-4", "family-5"}]
    assert result["synthesis_evidence"]["consolidation"]["global"] is True


def test_noncontracting_hierarchy_fails_without_losing_cached_drafts(tmp_path):
    source = records()
    features = [{"record_id": row["id"], "retained": True} for row in source]
    requests = []
    cache = cache_for(tmp_path, requests, draft_size=40000)
    with pytest.raises(ValueError, match="made no progress"):
        _synthesize({"records": source}, features, cache, cache.config.memory.model)
    assert len(requests) == 9
    assert len(list((tmp_path / "calls").glob("*.json"))) == 9
    assert sum("records" in data for data in requests) == 6


def test_hierarchy_combines_recurrence_across_original_partitions(tmp_path):
    source = records()
    for index, row in enumerate(source):
        row.update(
            role="assistant",
            kind="reasoning",
            independence_confidence="known",
            independence_group_id=f"family-{index}",
        )
    requests = []
    config = Config(memory=MemoryConfig(max_candidate_chars=200000))

    def transport(provider, request):
        assert len(ex.canonical(request)) <= config.memory.synthesis_max_request_bytes
        data = json.loads(request["input"])
        requests.append(data)
        if "records" in data:
            proposed = candidate(data["records"][0], 40000)
            proposed["evidence_ids"] = [row["id"] for row in data["records"]]
        else:
            proposed = candidate(data["sources"][0], 1000)
            proposed["evidence_ids"] = sorted({ref for item in data["candidates"] for ref in item["evidence_ids"]})
        proposed["category"] = "repeated_discovery"
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({"candidates": [proposed]})}]}]}

    cache = ex.ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport, config=config)
    features = [{"record_id": row["id"], "retained": True} for row in source]
    result = _synthesize({"records": source}, features, cache, config.memory.model)
    assert len(requests) == 10
    assert result["synthesis_evidence"]["consolidation"]["level_group_counts"] == [3, 1]
    assert len(result["candidates"]) == 1
    final = result["candidates"][0]
    assert set(final["evidence_ids"]) == {row["id"] for row in source}
    assert set(final["supporting_independence_groups"]) == {f"family-{index}" for index in range(6)}
    assert final["supporting_sessions"] == 6
