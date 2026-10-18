import json
from pathlib import Path

import pytest

from agent_hotwash.memory.evaluation import evaluate, lexical_baseline

FIXTURE = Path(__file__).parent / "fixtures" / "memory_evaluation.json"


def test_missing_predictions_are_measured_as_misses():
    examples = json.loads(FIXTURE.read_text())["examples"]
    result = evaluate(examples, [])
    assert result["missing_predictions"] == len(examples)
    for split in result["splits"].values():
        retained = split["features"]["retained"]
        assert retained["recall"] == 0
        assert retained["accuracy"] == 0
        assert retained["abstained"] == split["examples"]


def test_exact_predictions_recover_all_expected_labels():
    examples = json.loads(FIXTURE.read_text())["examples"]
    predictions = [{"record_id": example["record"]["id"], **example["labels"]} for example in examples]
    result = evaluate(examples, predictions)
    assert result["missing_predictions"] == 0
    for split in result["splits"].values():
        assert all(metric["accuracy"] == 1 for metric in split["features"].values())


def test_duplicate_and_unknown_predictions_fail_instead_of_hiding_errors():
    examples = json.loads(FIXTURE.read_text())["examples"]
    prediction = lexical_baseline(examples)[0]
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate(examples, [prediction, prediction])
    with pytest.raises(ValueError, match="unknown record"):
        evaluate(examples, [{"record_id": "unknown"}])


def test_fixture_is_synthetic_and_session_disjoint():
    examples = json.loads(FIXTURE.read_text())["examples"]
    sessions = {}
    for example in examples:
        assert example["origin"] == "synthetic"
        assert len(example["record"]["source_sha256"]) == 64
        sessions.setdefault(example["split"], set()).add(example["record"]["session_id"])
    assert len(sessions["development"]) == 2
    assert len(sessions["heldout"]) == 2
    assert sessions["development"].isdisjoint(sessions["heldout"])


def test_cli_scores_only_requested_split_from_full_trace_predictions(tmp_path, capsys):
    from agent_hotwash.memory.evaluation import main

    examples = json.loads(FIXTURE.read_text())["examples"]
    predictions = [{"record_id": example["record"]["id"], **example["labels"]} for example in examples]
    predictions.append({"record_id": "unlabeled-trace-record", "retained": True})
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(predictions))
    with pytest.raises(ValueError, match="unlabeled prediction IDs"):
        main(["--labels", str(FIXTURE), "--predictions", str(path)])
    main(["--labels", str(FIXTURE), "--predictions", str(path), "--full-trace-predictions", "--split", "development"])
    result = json.loads(capsys.readouterr().out)
    assert result["examples"] == sum(example["split"] == "development" for example in examples)
    assert set(result["splits"]) == {"development"}
    assert result["splits"]["development"]["features"]["retained"]["recall"] == 1


def test_canonical_mapping_preserves_annotation_scope_and_provenance():
    from agent_hotwash.memory.evaluation import map_examples_to_evidence

    sha = "a" * 64
    evidence = {"schema_version": "memory-evidence-v2", "records": []}
    examples: list[dict] = []
    cases = [
        ("Exact finding.", "Exact finding.", "exact_text"),
        ("Read [LOCAL_PATH] first.", "Read /local/repo/guide.md first.", "declared_local_path_sanitization"),
        ("x" * 2400, "x" * 2400 + " extra context", "declared_2400_character_excerpt"),
    ]
    for index, (annotation, full, _) in enumerate(cases):
        original = {
            "id": f"legacy-{index}",
            "session_id": "s",
            "role": "assistant",
            "kind": "message",
            "text": annotation,
            "source_sha256": sha,
        }
        examples.append({"record": original, "labels": {"discovery": True}, "split": "development"})
        evidence["records"].append(
            {
                **original,
                "id": f"canonical-{index}",
                "text": full,
                "canonical_event_identity": f"event-{index}",
                "source_refs": [{"source_sha256": sha, "line_number": index + 1}],
            }
        )
    mapping = map_examples_to_evidence(examples, evidence)
    assert not mapping["diagnostics"]
    assert mapping["summary"] == {
        "input_examples": 3,
        "mapped_examples": 3,
        "exact_text_matches": 1,
        "sanitized_matches": 1,
        "bounded_excerpt_matches": 1,
    }
    for index, (_, _, basis) in enumerate(cases):
        mapped = mapping["examples"][index]
        assert mapped["record"]["text"] == examples[index]["record"]["text"]
        assert mapped["record"]["id"] == f"canonical-{index}"
        assert mapped["mapping"]["basis"] == basis
        assert mapped["mapping"]["original_evidence_id"] == f"legacy-{index}"
    assert not mapping["examples"][-1]["mapping"]["annotation_covers_full_record"]


def test_canonical_mapping_never_guesses_on_missing_or_ambiguous_text():
    from agent_hotwash.memory.evaluation import map_examples_to_evidence

    original = {
        "id": "original",
        "session_id": "s",
        "role": "user",
        "kind": "message",
        "text": "Use shared tools.",
        "source_sha256": "a" * 64,
    }
    example = {"record": original}
    candidate = {
        **original,
        "id": "canonical",
        "canonical_event_identity": "event",
        "source_refs": [{"source_sha256": "a" * 64}],
    }
    bundle = {"schema_version": "memory-evidence-v2", "records": [{**candidate, "text": "Use the shared tools."}]}
    assert map_examples_to_evidence([example], bundle)["diagnostics"][0]["status"] == "missing"
    bundle["records"] = [candidate, {**candidate, "id": "other-event"}]
    assert map_examples_to_evidence([example], bundle)["diagnostics"][0]["status"] == "ambiguous"
