"""Complete synthesis envelopes include partial headers and bounded repair diagnostics."""

from collections import Counter

import pytest

from agent_hotwash.config import Config
from agent_hotwash.memory import extraction as ex


def record(index):
    return {
        "id": f"record-{index}",
        "session_id": f"session-{index}",
        "role": "user",
        "kind": "message",
        "authorship": "human",
        "text": "No, preserve the existing schema.",
        "paths": [],
    }


def near_boundary(rows, maximum):
    size = len(ex.canonical(ex.openai_request("gpt-6.1-sol", {"records": rows})))
    rows[0]["text"] += "x" * (maximum - size - 1)
    assert len(ex.canonical(ex.openai_request("gpt-6.1-sol", {"records": rows}))) == maximum - 1
    return rows


def test_partial_header_overflow_splits_only_affected_legacy_partition(monkeypatch, tmp_path):
    config = Config()
    maximum = config.memory.synthesis_max_request_bytes - config.memory.synthesis_repair_reserve_bytes
    overflowing = near_boundary([record(0), record(1)], maximum)
    healthy = [record(2)]
    legacy = [overflowing, healthy]
    monkeypatch.setattr(ex, "_evidence_batches", lambda *args, **kwargs: iter(legacy))
    batches = list(ex.evidence_batches([], [], policy=config.memory))
    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert Counter(row["id"] for batch in batches for row in batch) == Counter(
        row["id"] for batch in legacy for row in batch
    )
    assert batches[-1] is healthy
    cache = ex.ProviderCache(tmp_path, config=config)
    old = ex.openai_request(config.memory.model, {"records": healthy}, partial=True)
    new = ex.openai_request(config.memory.model, {"records": batches[-1]}, partial=True)
    assert cache.request_identity("openai", old) == cache.request_identity("openai", new)
    for batch in batches:
        request = ex.openai_request(config.memory.model, {"records": batch}, partial=True)
        assert len(ex.canonical(request)) <= maximum
    assert batches[0][0] == overflowing[0]
    assert batches[1][0] == overflowing[1]


def test_single_record_overflow_fails_explicitly_before_transport(monkeypatch):
    config = Config()
    maximum = config.memory.synthesis_max_request_bytes - config.memory.synthesis_repair_reserve_bytes
    oversized = near_boundary([record(0)], maximum)
    monkeypatch.setattr(ex, "_evidence_batches", lambda *args, **kwargs: iter([oversized, [record(1)]]))
    with pytest.raises(ValueError, match="record cannot fit the complete repair-reserved"):
        list(ex.evidence_batches([], [], policy=config.memory))


def test_repair_envelope_bounds_diagnostics_without_changing_cited_evidence():
    rows = near_boundary([record(0)], 108000)
    evidence = {"records": rows}
    base = ex.openai_request("gpt-6.1-sol", evidence)
    response = {"output": [{"type": "output_text", "text": "\x00" * 10000}]}
    repaired = ex.repair_request(base, evidence, response, "\x00" * 2000, "a" * 64)
    import json

    payload = json.loads(repaired["input"])
    assert len(ex.canonical(repaired)) <= 120000
    assert payload["evidence"] == evidence
    assert payload["repair"]["validation_error_truncated"] is True
    assert payload["repair"]["previous_output_truncated"] is True


def test_partial_header_builder_preserves_existing_valid_request_bytes():
    evidence = {"records": [record(0)]}
    legacy = ex.openai_request("gpt-6.1-sol", evidence, synthesis=True)
    legacy["instructions"] += (
        "\nPartial batch: provisional single-family lessons are permitted; "
        "final consolidation enforces independent recurrence."
    )
    assert ex.canonical(ex.openai_request("gpt-6.1-sol", evidence, synthesis=True, partial=True)) == ex.canonical(
        legacy
    )
