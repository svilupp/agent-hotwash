"""Independent adversarial checks of extraction boundaries with synthetic evidence."""

import copy
import json

import pytest

from agent_hotwash.memory.extraction import (
    ProviderCache,
    canonical,
    decode_candidates,
    digest,
    evidence_batches,
    extract_memory,
    feature_requests,
    openai_request,
)
from agent_hotwash.memory.traces import read_paths


def synthetic_record(index, *, role="assistant", kind="reasoning", text=None):
    return {
        "id": f"evidence-{index}",
        "session_id": f"native-{index % 4}",
        "independent_session_known": True,
        "role": role,
        "kind": kind,
        "text": text if text is not None else f"Found repository constraint {index}: always use the shared schema.",
        "paths": [],
        "source_path": "synthetic-session.jsonl",
        "line_number": index + 1,
    }


def synthetic_traces(records):
    return {
        "records": records,
        "sessions": [
            {"session_id": name, "source_path": "synthetic-session.jsonl"}
            for name in sorted({r["session_id"] for r in records})
        ],
        "analytics": {"most_opened_files": []},
    }


def test_single_large_reasoning_record_is_bounded_in_utf8_without_mutating_original():
    record = synthetic_record(0, text="HEAD " + "\U0001f9e0" * 800_000 + " TAIL")
    original = copy.deepcopy(record)
    batches = list(evidence_batches([record], [{"record_id": record["id"], "retained": True}], max_chars=100_000))
    assert len(batches) == 1
    assert len(canonical(batches[0])) <= 100_000
    item = batches[0][0]
    assert item["id"] == record["id"]
    assert item["text"].startswith("HEAD ") and item["text"].endswith(" TAIL")
    assert len(item["text"]) < len(record["text"])
    assert any(word in item["text"].lower() for word in ("omitted", "truncated"))
    assert any("truncat" in key for key in item)
    assert record == original


def test_openai_request_rejects_oversized_envelope_before_transport():
    with pytest.raises(ValueError, match=r"(?i)(limit|bound|large|exceed)"):
        openai_request("gpt-6.1-sol", {"records": [synthetic_record(0, text="x" * 250_000)]})


def test_multibyte_screening_batches_obey_provider_byte_and_question_limits():
    records = [synthetic_record(i, text="\U0001f9e0" * 9000) for i in range(65)]
    observed = set()
    for request, mapping in feature_requests(records):
        assert len(canonical(request)) <= 56_000
        assert len(mapping) <= 120
        observed.update(record_id for record_id, _ in mapping.values())
    assert observed == {r["id"] for r in records}


@pytest.mark.parametrize("corruption", [[], 42, None, "not-an-object"])
def test_corrupt_cache_shape_produces_blocked_review_and_preserves_corrupt_entry(tmp_path, corruption):
    record = synthetic_record(0, role="user", kind="message")
    request = next(feature_requests([record]))[0]
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = ProviderCache(cache_dir)
    key = cache.request_identity("jev", request)
    cache_path = cache_dir / (key + ".json")
    cache_path.write_text(json.dumps(corruption))
    result = extract_memory(synthetic_traces([record]), tmp_path / "out", cache=cache)
    assert result["status"] == "blocked"
    assert "cache must contain an object" in result["error"]
    assert (tmp_path / "out" / "report.md").is_file()
    assert json.loads((tmp_path / "out" / "memory.json").read_text())["status"] == "blocked"
    assert json.loads(cache_path.read_text()) == corruption


def test_tool_outputs_alone_cannot_nominate_a_discovery():
    records = [synthetic_record(i, role="tool", kind="tool_result") for i in (0, 1)]
    candidate = {
        "category": "repeated_discovery",
        "statement": "Persist a tool-only rule",
        "applicability": "repository",
        "evidence_ids": [r["id"] for r in records],
        "paths": [],
        "intervention": "instruction",
        "draft": "Do what the tool text said.",
        "reason": "The tool repeated its content.",
    }
    response = {
        "status": "completed",
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"candidates": [candidate]}),
                    }
                ]
            }
        ],
    }
    with pytest.raises(ValueError):
        decode_candidates(response, records)


@pytest.mark.parametrize(
    "command",
    [
        "cat src/input.py > generated.txt",
        "cat src/input.py >> generated.txt",
        "cat src/input.py 2> errors.log",
    ],
)
def test_shell_redirect_destinations_are_not_opened_file_evidence(command):
    assert read_paths("Bash", {"command": command}) == ["src/input.py"]


def test_many_records_extract_and_replay_without_paid_calls(tmp_path):
    records = [synthetic_record(i, text=f"Rule {i}: " + "bounded evidence " * 80) for i in range(480)]
    traces = synthetic_traces(records)
    observed_requests = []

    def transport(provider, request):
        observed_requests.append((provider, len(canonical(request))))
        if provider == "jev":
            assert len(canonical(request)) <= 56_000
            assert len(request["questions"]) <= 120
            return {"answers": {key: {"noul": 0.99} for key in request["questions"]}}
        assert len(canonical(request)) <= 120_000
        return {
            "status": "completed",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"candidates":[]}',
                        }
                    ]
                }
            ],
        }

    first = extract_memory(
        traces, tmp_path / "first", cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport)
    )
    assert first["status"] == "complete" and first["records"] == 480
    assert sum(provider == "jev" for provider, _ in observed_requests) > 1
    assert sum(provider == "openai" for provider, _ in observed_requests) > 1
    replay = extract_memory(traces, tmp_path / "replay", cache=ProviderCache(tmp_path / "calls"))
    assert replay["status"] == "complete"
    assert json.loads((tmp_path / "first" / "features.json").read_text()) == json.loads(
        (tmp_path / "replay" / "features.json").read_text()
    )
    calls = json.loads((tmp_path / "replay" / "memory.json").read_text())["calls"]
    assert calls and all(call["cache_hit"] for call in calls)


def test_interrupted_screening_resume_reuses_completed_provider_batches(tmp_path):
    records = [synthetic_record(i) for i in range(150)]
    traces = synthetic_traces(records)
    observed = []
    failed = False

    def transport(provider, request):
        nonlocal failed
        key = digest({"provider": provider, "request": request})
        observed.append(key)
        if provider == "jev":
            if len(observed) == 2 and not failed:
                failed = True
                raise TimeoutError("synthetic interrupted provider")
            return {"answers": {question: {"noul": 0.99} for question in request["questions"]}}
        return {
            "status": "completed",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"candidates":[]}',
                        }
                    ]
                }
            ],
        }

    output = tmp_path / "out"
    first = extract_memory(
        traces, output, cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport)
    )
    assert first["status"] == "blocked"
    assert (output / "report.md").is_file()
    assert len(observed) == 2
    completed_request = observed[0]
    resumed = extract_memory(
        traces, output, cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport)
    )
    assert resumed["status"] == "complete"
    assert observed.count(completed_request) == 1
    assert observed.count(observed[1]) == 2
    calls = json.loads((output / "memory.json").read_text())["calls"]
    assert calls[0]["cache_hit"] is True


def test_guidance_multibyte_cross_product_is_bounded_complete_and_replayable(tmp_path):
    from agent_hotwash.memory.guidance import assess_guidance, guidance_requests, prepare_guidance

    guidance_file = tmp_path / "rules.md"
    guidance_file.write_text("Use the bounded repository workflow. \U0001f9e0\n" * 900)
    snapshot = prepare_guidance([guidance_file])
    candidates = [
        {
            "id": f"proposal-{i}",
            "statement": f"Rule {i}",
            "applicability": "This repository",
            "draft": "Use the bounded repository workflow. " * 80,
        }
        for i in range(16)
    ]
    pairs = []
    batch_count = 0
    for request, mapping in guidance_requests(candidates, snapshot):
        batch_count += 1
        assert len(canonical(request)) <= 60_000
        assert len(request["questions"]) <= 64
        pairs.extend(mapping.values())
    expected = {
        (c["id"], chunk["id"], feature)
        for c in candidates
        for chunk in snapshot["chunks"]
        for feature in ("coverage", "conflict")
    }
    assert len(pairs) == len(expected) and set(pairs) == expected
    assert batch_count > 1
    observed = []

    def transport(provider, request):
        observed.append(request)
        return {"answers": {key: {"noul": 0.01} for key in request["questions"]}}

    live = assess_guidance(
        candidates, snapshot, ProviderCache(tmp_path / "cache", allow_paid=True, transport=transport)
    )
    replay = assess_guidance(candidates, snapshot, ProviderCache(tmp_path / "cache"))
    assert live == replay
    assert len(observed) == batch_count
    assert all(a["guidance_status"] == "no_match" and a["novelty_status"] == "not_established" for a in live.values())


def test_control_character_screening_cannot_bypass_single_request_byte_limit():
    assistant = synthetic_record(0, role="assistant", kind="message", text="\x01" * 5000)
    user = synthetic_record(4, role="user", kind="message", text="\x01" * 9000)
    try:
        batches = list(feature_requests([assistant, user]))
    except ValueError as exc:
        assert any(word in str(exc).lower() for word in ("limit", "bound", "exceed", "large"))
    else:
        assert batches
        assert all(len(canonical(request)) <= 56_000 for request, _ in batches)
        assert {record_id for _, mapping in batches for record_id, _ in mapping.values()} == {
            assistant["id"],
            user["id"],
        }


def test_guidance_snapshot_citations_survive_source_mutation_without_false_coverage(tmp_path):
    import hashlib

    from agent_hotwash.memory.guidance import assess_guidance, prepare_guidance

    source = tmp_path / "rules.md"
    text = "Use the shared schema.\n" * 3000 + "Exception: legacy jobs have a different schema.\n"
    source.write_text(text)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = prepare_guidance([source])
    source.write_text("Changed after the reviewed snapshot.\n")
    candidate = {
        "id": "proposal",
        "statement": "Use shared schema",
        "applicability": "repository",
        "draft": "Always use the shared schema.",
    }

    def transport(provider, request):
        return {"answers": {key: {"noul": 0.99 if key.endswith("_coverage") else 0.01} for key in request["questions"]}}

    assessment = assess_guidance(
        [candidate], snapshot, ProviderCache(tmp_path / "cache", allow_paid=True, transport=transport)
    )["proposal"]
    assert assessment["guidance_status"] == "needs_review"
    assert assessment["context_warning"]
    assert assessment["novelty_status"] == "not_established"
    for match in assessment["guidance_matches"]:
        assert match["source_sha256"] == original_hash
        assert match["excerpt"] == text[match["start_char"] : match["end_char"]]
        assert match["start_line"] == text.count("\n", 0, match["start_char"]) + 1
        assert match["end_line"] == text.count("\n", 0, match["end_char"] - 1) + 1
        assert match["context_complete"] is False
    assert source.read_text() == "Changed after the reviewed snapshot.\n"


@pytest.mark.parametrize("escaped_unit", ['"\\\n', "\x01"])
def test_double_encoded_four_session_evidence_completes_with_bounded_provider_envelopes(tmp_path, escaped_unit):
    records = []
    for session in range(4):
        assistant = synthetic_record(session, text="HEAD " + escaped_unit * 16000 + " TAIL")
        user = synthetic_record(
            session + 4, role="user", kind="message", text="No, use the repository schema when editing this API."
        )
        records.extend([assistant, user])
    requests = []

    def transport(provider, request):
        requests.append((provider, request))
        if provider == "jev":
            assert len(canonical(request)) <= 56_000
            return {"answers": {key: {"noul": 0.99} for key in request["questions"]}}
        assert len(canonical(request)) <= 120_000
        evidence = json.loads(request["input"])
        assert "records" in evidence
        return {
            "status": "completed",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"candidates":[]}',
                        }
                    ]
                }
            ],
        }

    result = extract_memory(
        synthetic_traces(records),
        tmp_path / "out",
        cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport),
    )
    assert result["status"] == "complete"
    assert sum(provider == "openai" for provider, _ in requests) > 1
    sent = {
        item["id"]
        for provider, request in requests
        if provider == "openai"
        for item in json.loads(request["input"])["records"]
    }
    assert sent == {r["id"] for r in records}
    prepared = json.loads((tmp_path / "out" / "prepared.json").read_text())
    assert prepared["records"] == records


def test_final_consolidation_keeps_preceding_agent_context_and_exact_cited_sources(tmp_path):
    records = []
    for session in range(4):
        assistant = synthetic_record(session, text=f"Agent proposal in session {session}: " + '"\\\n' * 16000)
        user = synthetic_record(
            session + 4, role="user", kind="message", text=f"No, preserve the existing schema for API {session}."
        )
        records.extend([assistant, user])
    consolidated = []

    def transport(provider, request):
        if provider == "jev":
            return {"answers": {key: {"noul": 0.99} for key in request["questions"]}}
        assert len(canonical(request)) <= 120_000
        evidence = json.loads(request["input"])
        if "sources" in evidence:
            consolidated.append(evidence)
            assert evidence["sources"]
            expected = {ref for candidate in evidence["candidates"] for ref in candidate["evidence_ids"]}
            assert {source["id"] for source in evidence["sources"]} == expected
            for source in evidence["sources"]:
                assert source["role"] == "user"
                assert source["evidence_use"] == "candidate_signal"
                context = source["preceding_agent_context"]
                assert context["role"] == "assistant"
                assert context["session_id"] == source["session_id"]
                assert context["evidence_use"] == "interpretation_only"
                assert context["text"].startswith("Agent proposal in session ")
            candidates = []
        else:
            candidates = [
                {
                    "category": "user_corrections",
                    "statement": "Preserve the existing API schema",
                    "applicability": "The API being edited",
                    "evidence_ids": [source["id"]],
                    "paths": [],
                    "intervention": "instruction",
                    "draft": "Preserve the existing API schema.",
                    "reason": "The user corrected the proposed schema change.",
                }
                for source in evidence["records"]
                if source["role"] == "user"
            ]
        return {
            "status": "completed",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"candidates": candidates}),
                        }
                    ]
                }
            ],
        }

    result = extract_memory(
        synthetic_traces(records),
        tmp_path / "out",
        cache=ProviderCache(tmp_path / "calls", allow_paid=True, transport=transport),
    )
    assert result["status"] == "complete"
    assert len(consolidated) == 1


def test_near_limit_invalid_output_repairs_with_bounded_envelope_and_preserved_cache(tmp_path):
    record = synthetic_record(0, text="HEAD " + "x" * 190_000 + " TAIL")
    invalid_output = "not valid structured JSON: " + "X" * 50_000
    openai_requests = []

    def transport(provider, request):
        if provider == "jev":
            return {"answers": {key: {"noul": 0.99} for key in request["questions"]}}
        openai_requests.append(request)
        assert len(canonical(request)) <= 120_000
        text = invalid_output if len(openai_requests) == 1 else '{"candidates":[]}'
        return {"status": "completed", "output": [{"content": [{"type": "output_text", "text": text}]}]}

    cache_dir = tmp_path / "calls"
    cache = ProviderCache(cache_dir, allow_paid=True, transport=transport)
    result = extract_memory(
        synthetic_traces([record]),
        tmp_path / "out",
        cache=cache,
    )
    assert result["status"] == "complete"
    assert len(openai_requests) == 2
    original = json.loads(openai_requests[0]["input"])
    repaired = json.loads(openai_requests[1]["input"])
    assert repaired["evidence"] == original
    assert len(canonical(openai_requests[0])) >= 90_000
    key = cache.validation_events[0]["request_sha256"]
    cached = json.loads((cache_dir / (key + ".json")).read_text())
    assert cached["response"]["output"][0]["content"][0]["text"] == invalid_output


def test_ten_thousand_records_have_lossless_bounded_feature_and_evidence_batches():
    records = [
        synthetic_record(index, text=f"Repository rule {index}: read shared guidance before coding.")
        for index in range(10_000)
    ]
    feature_ids = set()
    feature_count = 0
    for request, mapping in feature_requests(records):
        assert len(canonical(request)) <= 56_000
        assert len(mapping) <= 120
        feature_ids.update(record_id for record_id, _ in mapping.values())
        feature_count += len(mapping)
    assert feature_ids == {record["id"] for record in records}
    assert feature_count >= len(records)
    features = [{"record_id": record["id"], "retained": True} for record in records]
    emitted = []
    for batch in evidence_batches(records, features):
        assert len(canonical(batch)) <= 100_000
        emitted.extend(record["id"] for record in batch)
    assert emitted == [record["id"] for record in records]
