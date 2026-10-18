"""Compare screened evidence with labeled examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FEATURES = ("correction", "discovery", "reusable", "retained")


def evaluate(examples, predictions, threshold=0.5):
    """Missing and uncertain predictions count as errors, never disappear."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    by_id = {}
    gold_ids = {example["record"]["id"] for example in examples}
    if len(gold_ids) != len(examples):
        raise ValueError("duplicate labeled record IDs")
    for prediction in predictions:
        record_id = prediction["record_id"]
        if record_id in by_id:
            raise ValueError(f"duplicate prediction ID: {record_id}")
        by_id[record_id] = prediction
    if extra := by_id.keys() - gold_ids:
        raise ValueError(f"predictions contain {len(extra)} unknown record IDs")
    output = {
        "examples": len(examples),
        "missing_predictions": len(gold_ids - by_id.keys()),
        "feature_threshold": threshold,
        "splits": {},
    }
    for split in sorted({example["split"] for example in examples}):
        selected = [example for example in examples if example["split"] == split]
        metrics = {}
        for feature in FEATURES:
            tp = fp = fn = tn = abstained = count = 0
            for example in selected:
                gold = example["labels"].get(feature)
                if gold is None:
                    continue
                count += 1
                value = by_id.get(example["record"]["id"], {}).get(feature)
                if value is None:
                    abstained += 1
                    fn += bool(gold)
                    continue
                if feature == "retained":
                    if not isinstance(value, bool):
                        raise ValueError("retained predictions must be booleans")
                    predicted = value
                else:
                    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                        raise ValueError("feature predictions must be probabilities")
                    predicted = value >= threshold
                tp += bool(gold and predicted)
                fp += bool(not gold and predicted)
                fn += bool(gold and not predicted)
                tn += bool(not gold and not predicted)
            metrics[feature] = {
                "count": count,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "abstained": abstained,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "accuracy": (tp + tn) / count if count else None,
            }
        output["splits"][split] = {"examples": len(selected), "features": metrics}
    return output


def lexical_baseline(examples):
    """Simple reproducible baseline with no provider calls or learned thresholds."""
    predictions = []
    for example in examples:
        record = example["record"]
        text = record["text"].lower()
        user = record["role"] == "user"
        correction = user and bool(re.search(r"\b(no|too|dislike|instead|careful|meant|don.t)\b", text))
        discovery = record["role"] == "assistant" and bool(
            re.search(r"\b(found|fails|failure|bug|mistake|root cause|missing)\b", text)
        )
        reusable = bool(re.search(r"\b(always|never|checklist|preference|convention)\b", text))
        predictions.append(
            {
                "record_id": record["id"],
                "correction": float(correction) if user else None,
                "discovery": float(discovery) if not user else None,
                "reusable": float(reusable),
                "retained": user or discovery or reusable,
            }
        )
    return predictions


def map_examples_to_evidence(examples, bundle):
    """Map reviewed annotations by source identity and visible text, never fuzzily.

    Annotation text remains the annotated excerpt. Declared [LOCAL_PATH]
    sanitization and the original 2400-character cap are recorded separately
    from exact matches so a full-record prediction is not silently represented
    as an independently labeled full record.
    """
    mapped, diagnostics = [], []
    for example in examples:
        original = example["record"]
        matches = []
        for record in bundle["records"]:
            if any(record[key] != original[key] for key in ("session_id", "role", "kind")):
                continue
            if not any(ref.get("source_sha256") == original["source_sha256"] for ref in record.get("source_refs", [])):
                continue
            text, annotated = record["text"], original["text"]
            basis = None
            if text == annotated:
                basis = "exact_text"
            elif "[LOCAL_PATH]" in annotated and re.fullmatch(
                re.escape(annotated).replace(re.escape("[LOCAL_PATH]"), r"\S+"), text
            ):
                basis = "declared_local_path_sanitization"
            elif len(annotated) == 2400 and text.startswith(annotated):
                basis = "declared_2400_character_excerpt"
            if basis:
                matches.append((record, basis))
        if len(matches) != 1:
            diagnostics.append(
                {
                    "original_id": original["id"],
                    "status": "missing" if not matches else "ambiguous",
                    "candidate_count": len(matches),
                }
            )
            continue
        record, basis = matches[0]
        mapped.append(
            {
                **example,
                "record": {**original, "id": record["id"]},
                "mapping": {
                    "original_evidence_id": original["id"],
                    "canonical_evidence_id": record["id"],
                    "canonical_event_identity": record["canonical_event_identity"],
                    "source_refs": record["source_refs"],
                    "basis": basis,
                    "annotation_text_preserved": True,
                    "annotation_covers_full_record": basis != "declared_2400_character_excerpt",
                },
            }
        )
    return {
        "schema_version": "memory-learning-canonical-mapping-v1",
        "evidence_schema_version": bundle["schema_version"],
        "examples": mapped,
        "diagnostics": diagnostics,
        "summary": {
            "input_examples": len(examples),
            "mapped_examples": len(mapped),
            "exact_text_matches": sum(e["mapping"]["basis"] == "exact_text" for e in mapped),
            "sanitized_matches": sum(e["mapping"]["basis"] == "declared_local_path_sanitization" for e in mapped),
            "bounded_excerpt_matches": sum(e["mapping"]["basis"] == "declared_2400_character_excerpt" for e in mapped),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--map-prepared", type=Path, help="Map authentic labels to a canonical prepared bundle")
    parser.add_argument("--mapping-output", type=Path, help="Write the provenance-linked label mapping")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split", choices=("development", "heldout"))
    parser.add_argument(
        "--full-trace-predictions",
        action="store_true",
        help="Select labeled IDs from predictions covering a larger trace set",
    )
    args = parser.parse_args(argv)
    examples = json.loads(args.labels.read_text())["examples"]
    if args.map_prepared:
        from .extraction import write_json
        from .models import EvidenceBundle

        bundle = EvidenceBundle.validate_dict(json.loads(args.map_prepared.read_text()))
        mapping = map_examples_to_evidence(examples, bundle)
        if args.mapping_output:
            write_json(args.mapping_output, mapping)
        print(json.dumps({"summary": mapping["summary"], "diagnostics": mapping["diagnostics"]}, indent=2))
        return
    if args.mapping_output:
        parser.error("--mapping-output requires --map-prepared")
    predictions = json.loads(args.predictions.read_text()) if args.predictions else lexical_baseline(examples)
    if isinstance(predictions, dict):
        predictions = predictions["features"]
    if not args.full_trace_predictions:
        known_ids = {example["record"]["id"] for example in examples}
        if any(prediction["record_id"] not in known_ids for prediction in predictions):
            raise ValueError("unlabeled prediction IDs; use --full-trace-predictions for a full trace run")
    if args.split:
        examples = [example for example in examples if example["split"] == args.split]
    if args.split or args.full_trace_predictions:
        selected_ids = {example["record"]["id"] for example in examples}
        predictions = [prediction for prediction in predictions if prediction["record_id"] in selected_ids]
    print(json.dumps(evaluate(examples, predictions, args.threshold), indent=2))


if __name__ == "__main__":
    main()
