"""First-class memory audit stages with validated persistence and replay."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hotwash.semantic.redact import AUDIT_REDACTION_VERSION, REDACTION_VERSION

from . import extraction as ex
from .guidance import VERSION as GUIDANCE_VERSION
from .guidance import assess_guidance, prepare_guidance
from .persistence import atomic_bytes, atomic_text, audit_lock

MANIFEST_VERSION = "memory-stages-v1"
STAGES = ("preparation", "screening", "synthesis", "comparison", "rendering")
CONSOLIDATION_VERSION = "bounded-global-citations-v1"


class StageState(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: Literal["pending", "running", "complete", "blocked", "skipped"]
    input_sha256: str | None = None
    output_sha256: str | None = None


class StageManifest(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: Literal["memory-stages-v1"]
    run_id: str
    stages: dict[str, StageState]
    artifacts: dict[str, str] = Field(default_factory=dict)


def _validated_evidence(value):
    from .models import EvidenceBundle

    return EvidenceBundle.validate_dict(value)


def _validated_result(value):
    from .models import AuditResult

    return AuditResult.validate_dict(value)


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid saved artifact: {Path(path).name}") from exc


def load_saved(audit_dir):
    """Load current typed artifacts, rejecting incomplete/corrupt artifact sets."""
    directory = Path(audit_dir)
    manifest = _read(directory / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported memory stage manifest version")
    StageManifest.model_validate(manifest)
    raw_evidence = _read(directory / "prepared.json")
    raw_result = _read(directory / "memory.json")
    if not isinstance(raw_evidence, dict) or not isinstance(raw_result, dict):
        raise ValueError("Saved memory evidence and result must be JSON objects")
    if (
        raw_evidence.get("schema_version") != "memory-evidence-v2"
        or raw_result.get("schema_version") != "memory-audit-v2"
    ):
        raise ValueError("Unsupported memory artifact schema version")
    evidence = _validated_evidence(raw_evidence)
    result = _validated_result(raw_result)
    for name, value in (("prepared.json", evidence), ("memory.json", result)):
        expected = manifest.get("artifacts", {}).get(name)
        if expected != ex.digest(value):
            raise ValueError(f"Saved memory artifact hash mismatch: {name}")
    artifacts = manifest["artifacts"]
    supported = {"prepared.json", "memory.json", "features.json", "proposals.json", "comparison.json", "guidance.json"}
    for name, expected in artifacts.items():
        if name not in supported:
            raise ValueError("Unsupported memory artifact name in manifest")
        if ex.digest(_read(directory / name)) != expected:
            raise ValueError(f"Saved memory artifact hash mismatch: {name}")
    if "guidance.json" in artifacts:
        snapshot = _read(directory / "guidance.json")
        if snapshot.get("version") != GUIDANCE_VERSION:
            raise ValueError("Unsupported saved guidance snapshot version")
        chunks = {chunk["id"]: chunk for chunk in snapshot["chunks"]}
        for assessment in result.get("guidance_assessments", {}).values():
            for match in assessment.get("guidance_matches", []):
                chunk = chunks.get(match["chunk_id"])
                if chunk is None or match.get("excerpt") != chunk["text"]:
                    raise ValueError("Saved guidance citation does not resolve against its snapshot")
    dataset_names = {
        "command-hooks.json",
        "command-calls.jsonl",
        "command-calls.csv",
        "command-matches.jsonl",
        "command-matches.csv",
    }
    datasets = manifest.get("dataset_artifacts", {})
    if result.get("command_hooks") and set(datasets) != dataset_names:
        raise ValueError("Saved memory command dataset manifest is incomplete")
    for name, expected in datasets.items():
        if name not in dataset_names:
            raise ValueError("Unsupported memory dataset name in manifest")
        try:
            data = (directory / name).read_bytes()
        except OSError as exc:
            raise ValueError(f"Saved memory dataset is unavailable: {name}") from exc
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Saved memory dataset hash mismatch: {name}")
    return evidence, result


def _export_commands(output, records, hooks):
    from .command_hooks import collect_command_hooks

    dataset = collect_command_hooks(records, hooks)
    payloads = {"command-hooks.json": json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False)}
    for name, fields in (
        (
            "calls",
            [
                "id",
                "evidence_id",
                "session_id",
                "tool_name",
                "command",
                "shell_command",
                "command_index",
                "parse_status",
                "source_path",
                "line_number",
                "source_refs",
            ],
        ),
        ("matches", ["call_id", "hook", "values", "captures"]),
    ):
        stem = "command-" + name
        rows = dataset[name]
        payloads[stem + ".jsonl"] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
        payloads[stem + ".csv"] = stream.getvalue()
    hashes = {}
    for name, text in payloads.items():
        data = text.encode("utf-8")
        atomic_bytes(output / name, data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {
        "call_count": len(dataset["calls"]),
        "match_count": len(dataset["matches"]),
        "summaries": dataset["summaries"],
        "diagnostics": dataset.get("diagnostics", []),
        "dataset": "command-hooks.json",
    }, hashes


def _summary(evidence, result, output):
    return {
        "status": result["status"],
        "sessions": len(evidence["sessions"]),
        "records": len(evidence["records"]),
        "candidates": len(result["candidates"]),
        "output": str(Path(output).resolve()),
        **({"error": result["error"]} if result.get("error") else {}),
    }


def _consolidate_candidates(candidates, batches, originals_by_id, cache, model, *, on_consolidation_level=None):
    """Globally consolidate validated drafts using the selected evidence batches.

    Kept separate from initial batch extraction so an immutable, validated
    first-pass draft artifact can resume directly at global consolidation.
    """
    consolidation_levels = []
    compact_sources = False
    input_candidate_count = len(candidates)
    if len(batches) > 1 and candidates:
        candidates = list({candidate["id"]: candidate for candidate in candidates}.values())
        sources = {item["id"]: item for batch in batches for item in batch}
        policy = cache.config.memory
        maximum = policy.synthesis_max_request_bytes - policy.synthesis_repair_reserve_bytes

        def payload(group):
            ids = {ref for candidate in group for ref in candidate["evidence_ids"]}
            return {
                "candidates": [{key: candidate[key] for key in ex.CANDIDATE_PROPERTIES} for candidate in group],
                "sources": [sources[ref] for ref in sorted(ids)],
            }

        def fits(group, *, partial=False):
            try:
                request = ex.openai_request(model, payload(group), synthesis=True, partial=partial, max_bytes=maximum)
            except ValueError:
                return False
            return len(ex.canonical(request)) <= maximum

        def consolidate(group, *, partial=False, worker=cache):
            consolidation = payload(group)
            originals = [originals_by_id[row["id"]] for row in consolidation["sources"]]
            return ex.propose_candidates(worker, model, consolidation, originals, partial=partial, synthesis=True)

        if not fits(candidates):
            compact_sources = True
            sources = {
                ref: {
                    key: value
                    for key, value in row.items()
                    if key not in ("text", "preceding_agent_context", "synthesis_truncation")
                }
                | {"text_omitted": True, "source_record_sha256": ex.digest(originals_by_id[ref])}
                for ref, row in sources.items()
            }
        while candidates and not fits(candidates):
            before = len(ex.canonical(payload(candidates)))
            candidate_count_before = len(candidates)
            groups, group = [], []
            for candidate in candidates:
                if group and not fits([*group, candidate], partial=True):
                    groups.append(group)
                    group = []
                if not fits([candidate], partial=True):
                    raise ValueError("A memory candidate and its citation metadata exceed the consolidation envelope")
                group.append(candidate)
            if group:
                groups.append(group)
            consolidation_levels.append(len(groups))
            merged = [
                candidate
                for proposed in ex.cached_operations(
                    groups, cache, lambda worker, group: consolidate(group, partial=True, worker=worker)
                )
                for candidate in proposed
            ]
            candidates = list({candidate["id"]: candidate for candidate in merged}.values())
            after = len(ex.canonical(payload(candidates))) if candidates else 0
            level = {
                "level": len(consolidation_levels),
                "groups": len(groups),
                "input_candidates": candidate_count_before,
                "output_candidates": len(candidates),
                "input_payload_bytes": before,
                "output_payload_bytes": after,
            }
            if on_consolidation_level is not None:
                on_consolidation_level(level)
            if candidates and not fits(candidates):
                if after >= before:
                    raise ValueError(
                        "Global memory consolidation made no progress; cached provisional drafts are preserved"
                    )
                if len(consolidation_levels) >= 32:
                    raise ValueError(
                        "Global memory consolidation exceeds 32 levels; cached provisional drafts are preserved"
                    )
        if candidates:
            input_count = len(candidates)
            input_bytes = len(ex.canonical(payload(candidates)))
            candidates = consolidate(candidates)
            consolidation_levels.append(1)
            output_bytes = len(ex.canonical(payload(candidates))) if candidates else 0
            level = {
                "level": len(consolidation_levels),
                "groups": 1,
                "input_candidates": input_count,
                "output_candidates": len(candidates),
                "input_payload_bytes": input_bytes,
                "output_payload_bytes": output_bytes,
                "final": True,
            }
            if on_consolidation_level is not None:
                on_consolidation_level(level)
    omitted = [item for batch in batches for item in batch if item.get("synthesis_truncation")]
    return (
        candidates,
        {
            "version": CONSOLIDATION_VERSION,
            "input_candidate_count": input_candidate_count,
            "level_group_counts": consolidation_levels,
            "global": len(batches) > 1,
            "cross_group_recurrence_evaluated": len(batches) > 1,
            "source_text_compacted": compact_sources,
            "validation_records": "canonical_originals",
        },
        omitted,
    )


def _synthesize(evidence, features, cache, model, *, on_consolidation_level=None):
    batches = list(ex.evidence_batches(evidence["records"], features, policy=cache.config.memory))
    originals_by_id = {row["id"]: row for row in evidence["records"]}
    candidates = []

    def propose_batch(worker, batch):
        originals = [originals_by_id[item["id"]] for item in batch]
        return ex.propose_candidates(worker, model, {"records": batch}, originals, partial=len(batches) > 1)

    for proposed in ex.cached_operations(batches, cache, propose_batch):
        candidates.extend(proposed)
    candidates, consolidation, omitted = _consolidate_candidates(
        candidates,
        batches,
        originals_by_id,
        cache,
        model,
        on_consolidation_level=on_consolidation_level,
    )
    return {
        "candidates": candidates,
        "synthesis_evidence": {
            "consolidation": consolidation,
            "truncated_record_ids": [item["id"] for item in omitted],
            "truncated_reasoning_records": sum(item["kind"] == "reasoning" for item in omitted),
            "full_text_artifact": "prepared.json",
        },
    }


def _execute(evidence, output, *, config, extract, allow_paid, snapshot, cache, saved=None):
    output = Path(output)
    policy = config.memory
    patterns = config.lexicons.secret
    evidence = ex.redact_tree(evidence, patterns)
    for record in evidence["records"]:
        record["retained_chars"] = len(record["text"])
    if saved is not None:
        saved = ex.redact_tree(saved, patterns)
    previous_manifest = output / "manifest.json"
    old = {}
    if previous_manifest.exists():
        raw_manifest = previous_manifest.read_bytes()
        try:
            old = json.loads(raw_manifest.decode("utf-8"))
            StageManifest.model_validate(old)
        except (ValueError, UnicodeError):
            quarantine = output / (".manifest-corrupt-" + uuid.uuid4().hex + ".json")
            atomic_bytes(quarantine, raw_manifest)
            failure = ValueError("Invalid existing stage manifest; preserved for inspection as " + quarantine.name)
            _preparation_failed(output, config, failure)
            raise failure from None
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "run_id": uuid.uuid4().hex,
        "config": ex.redact_tree(config.model_dump(mode="json"), config.lexicons.secret),
        "redaction_version": REDACTION_VERSION,
        "audit_redaction_version": AUDIT_REDACTION_VERSION,
        "secret_patterns_sha256": ex.digest(list(patterns)),
        "versions": {
            "evidence": evidence["schema_version"],
            "features": ex.FEATURE_VERSION,
            "synthesis": ex.SYNTHESIS_VERSION,
            "guidance": GUIDANCE_VERSION,
        },
        "source_hashes": {
            source["source_id"]: source.get("sha256", source.get("source_sha256"))
            for source in evidence.get("sources", [])
        },
        "guidance_snapshot_sha256": snapshot["snapshot_sha256"] if snapshot else None,
        "stages": {stage: {"status": "pending"} for stage in STAGES},
        "artifacts": {},
    }
    cache = cache or ex.ProviderCache(output / "calls", allow_paid=allow_paid, config=config)
    cache.config = config
    call_offset = len(cache.calls)
    validation_offset = len(cache.validation_events)
    result: dict[str, Any] = {
        "schema_version": "memory-audit-v2",
        "version": ex.VERSION,
        "feature_version": ex.FEATURE_VERSION,
        "synthesis_version": ex.SYNTHESIS_VERSION,
        "model": policy.model,
        "jev_model": policy.jev_model,
        "status": "preprocessed" if not extract else "complete",
        "run_id": manifest["run_id"],
        "features": [],
        "candidates": [],
        "calls": [],
        "source_sessions": evidence["sessions"],
        "thresholds": {"retain_at_or_above": policy.retain_threshold, "calibrated": False},
    }

    def persist_manifest():
        StageManifest.model_validate(manifest)
        ex._write_sanitized_json(output / "manifest.json", manifest)

    def stage(name, inputs, artifact, operation):
        identity = ex.digest(
            {
                "redaction_version": REDACTION_VERSION,
                "audit_redaction_version": AUDIT_REDACTION_VERSION,
                "secret_patterns_sha256": ex.digest(list(patterns)),
                **inputs,
            }
        )
        row = manifest["stages"][name]
        row.update(status="running", input_sha256=identity)
        persist_manifest()
        previous = old.get("stages", {}).get(name, {})
        path = output / artifact
        if previous.get("status") == "complete" and previous.get("input_sha256") == identity and path.is_file():
            value = _read(path)
            if ex.digest(value) != previous.get("output_sha256"):
                raise ValueError(f"Corrupt saved {name} stage; preserve for inspection")
            row["resumed"] = True
        else:
            offset = len(cache.calls)
            try:
                value = operation()
            finally:
                calls = cache.calls[offset:]
                row["request_references"] = [call["request_sha256"] for call in calls]
                row["audit_cache_hits"] = sum(call["cache_hit"] for call in calls)
                row["audit_cache_misses"] = sum(not call["cache_hit"] for call in calls)
                row["native_requests"] = sum((call.get("native_usage") or {}).get("requests", 0) for call in calls)
                row["provider_usage"] = {
                    "jev_requests": row["native_requests"],
                    "openai_requests": ex.openai_usage(calls)["requests"],
                    "openai_retries": ex.openai_usage(calls)["retries"],
                    "reported_usage": [call["usage"] for call in calls if call.get("usage") is not None],
                }
            ex._write_sanitized_json(path, value)
            row["resumed"] = False
        row.update(status="complete", output_sha256=ex.digest(value), artifact=artifact)
        manifest["artifacts"][artifact] = ex.digest(value)
        persist_manifest()
        return value

    # Publish a current in-progress result before work; failures cannot retain stale success.
    result["status"] = "running"
    ex._write_sanitized_json(output / "memory.json", result)
    for report in ("report.md", "report.html"):
        atomic_text(output / report, "Memory audit is running; see manifest.json for stage status.\n")
    persist_manifest()
    active = "preparation"
    try:
        stage("preparation", {"evidence": evidence, "redaction": REDACTION_VERSION}, "prepared.json", lambda: evidence)
        result["command_hooks"], manifest["dataset_artifacts"] = _export_commands(
            output, evidence["records"], policy.command_hooks
        )
        persist_manifest()
        persisted_snapshot = snapshot or {"version": GUIDANCE_VERSION, "files": [], "chunks": []}
        ex._write_sanitized_json(output / "guidance.json", persisted_snapshot)
        manifest["artifacts"]["guidance.json"] = ex.digest(persisted_snapshot)
        if extract:
            active = "screening"
            if saved is not None:
                result["features"] = saved.get("features", [])
                manifest["stages"]["screening"] = {"status": "skipped", "reason": "saved_proposals"}
            else:
                result["features"] = stage(
                    "screening",
                    {
                        "evidence": ex.digest(evidence),
                        "version": ex.FEATURE_VERSION,
                        "bank_hash": ex.digest(
                            {key: feature.question for key, feature in ex._native_memory_features().items()}
                        ),
                        "model": policy.jev_model,
                        "threshold": policy.retain_threshold,
                        "limits": [policy.screening_max_questions, policy.screening_max_request_bytes],
                    },
                    "features.json",
                    lambda: ex.screen_features(evidence, cache, retain_threshold=policy.retain_threshold),
                )
            active = "synthesis"
            if saved is not None:
                synthesized = {
                    "candidates": saved["candidates"],
                    "synthesis_evidence": saved.get("synthesis_evidence", {}),
                }
                ex._write_sanitized_json(output / "proposals.json", synthesized)
                manifest["stages"]["synthesis"] = {"status": "skipped", "reason": "saved_proposals"}
            else:
                synthesized = stage(
                    "synthesis",
                    {
                        "evidence": ex.digest(evidence),
                        "features": result["features"],
                        "version": ex.SYNTHESIS_VERSION,
                        "consolidation_version": CONSOLIDATION_VERSION,
                        "model": policy.model,
                        "limits": [
                            policy.synthesis_max_request_bytes,
                            policy.synthesis_repair_reserve_bytes,
                            policy.synthesis_max_repairs,
                            policy.max_candidates,
                            policy.max_candidate_chars,
                        ],
                    },
                    "proposals.json",
                    lambda: _synthesize(evidence, result["features"], cache, policy.model),
                )
            result.update(synthesized)
            active = "comparison"
            if snapshot and snapshot.get("chunks"):
                # Threshold changes rerun deterministic assessment while response identities remain unchanged.
                result["guidance_assessments"] = stage(
                    "comparison",
                    {
                        "candidates": result["candidates"],
                        "snapshot": snapshot,
                        "version": GUIDANCE_VERSION,
                        "model": policy.jev_model,
                        "limits": [
                            policy.max_pairs,
                            policy.max_candidates,
                            policy.max_candidate_chars,
                            policy.comparison_max_questions,
                            policy.comparison_max_request_bytes,
                        ],
                        "thresholds": [
                            policy.coverage_threshold,
                            policy.conflict_threshold,
                            policy.uncertain_threshold,
                        ],
                    },
                    "comparison.json",
                    lambda: assess_guidance(result["candidates"], snapshot, cache),
                )
            else:
                manifest["stages"]["comparison"] = {"status": "skipped", "reason": "no_usable_selected_guidance"}
            result["status"] = "complete"
        else:
            for name in ("screening", "synthesis", "comparison"):
                manifest["stages"][name] = {"status": "skipped", "reason": "preparation_only"}
            result["status"] = "preprocessed"
    except Exception as exc:
        result.update(status="blocked", error=ex.redact(str(exc), patterns), failing_stage=active)
        manifest["stages"][active].update(
            status="blocked", error=ex.redact(str(exc), patterns), error_type=type(exc).__name__
        )
    result["proposal_assessments"] = ex.assess_proposal_scope(result["candidates"], evidence["records"])
    for candidate in result["candidates"]:
        result["proposal_assessments"][candidate["id"]].update(
            result.get("guidance_assessments", {}).get(
                candidate["id"],
                {
                    "guidance_status": "unchecked",
                    "comparison_complete": False,
                    "comparison_scope": "selected_files_only",
                    "novelty_status": "not_established",
                    "installation_status": "human_review_required",
                },
            )
        )
    result["guidance_comparison"] = {
        "status": "blocked"
        if result.get("failing_stage") == "comparison"
        else "selected_files_only"
        if snapshot
        else "not_requested",
        "file_count": len(snapshot["files"]) if snapshot else 0,
    }
    jev_calls = [call for call in cache.calls[call_offset:] if call["provider"] == "jev"]
    native_keys = {key for call in jev_calls for key in (call.get("native_usage") or {})} | {"requests", "cache_hits"}
    result["provider_usage"] = {
        "jev": {key: sum((call.get("native_usage") or {}).get(key, 0) for call in jev_calls) for key in native_keys},
        "openai": ex.openai_usage(cache.calls[call_offset:]),
    }
    manifest["provider_usage"] = result["provider_usage"]
    result["calls"] = cache.calls[call_offset:]
    result["validation_events"] = cache.validation_events[validation_offset:]
    openai_calls = [call for call in result["calls"] if call.get("provider") == "openai"]
    model_rows: dict[str, dict[str, Any]] = {}
    for call in openai_calls:
        model = call.get("model", "unspecified")
        row = model_rows.setdefault(
            model,
            {
                "requests": 0,
                "cache_hits": 0,
                "new_responses": 0,
                "reasoning_efforts": {},
            },
        )
        row["requests"] += 1
        row["cache_hits"] += int(bool(call.get("cache_hit")))
        row["new_responses"] += int(not call.get("cache_hit") and not call.get("error_type"))
        effort = call.get("reasoning_effort") or "unspecified"
        row["reasoning_efforts"][effort] = row["reasoning_efforts"].get(effort, 0) + 1
    result["synthesis_provenance"] = {
        "openai_requests_by_model": model_rows,
        "request_references_include_each_actual_model_and_effort": True,
    }
    manifest["request_references"] = [call["request_sha256"] for call in result["calls"]]
    result = _validated_result(result)
    ex._write_sanitized_json(output / "memory.json", result)
    manifest["artifacts"]["memory.json"] = ex.digest(result)
    from agent_hotwash.report.memory import render_memory_html, render_memory_markdown

    active = "rendering"
    manifest["stages"][active] = {"status": "running", "input_sha256": ex.digest(result)}
    persist_manifest()
    try:
        markdown = render_memory_markdown(evidence, result)
        html = render_memory_html(evidence, result)
        atomic_text(output / "report.md", markdown)
        atomic_text(output / "report.html", html)
        manifest["stages"][active].update(
            status="complete", output_sha256=ex.digest({"markdown": markdown, "html": html})
        )
    except Exception as exc:
        manifest["stages"][active].update(status="blocked", error=ex.redact(str(exc), patterns))
        result.update(status="blocked", error=ex.redact(str(exc), patterns), failing_stage=active)
        ex._write_sanitized_json(output / "memory.json", result)
        manifest["artifacts"]["memory.json"] = ex.digest(result)
    persist_manifest()
    return _summary(evidence, result, output)


def _preparation_failed(output, config, exc):
    output = Path(output)
    error = ex.redact(str(exc), config.lexicons.secret)
    result = {
        "schema_version": "memory-audit-v2",
        "status": "blocked",
        "candidates": [],
        "features": [],
        "error": error,
        "failing_stage": "preparation",
    }
    ex._write_sanitized_json(output / "memory.json", _validated_result(result))
    ex._write_sanitized_json(
        output / "manifest.json",
        {
            "schema_version": MANIFEST_VERSION,
            "run_id": uuid.uuid4().hex,
            "config": ex.redact_tree(config.model_dump(mode="json"), config.lexicons.secret),
            "artifacts": {},
            "stages": {
                stage: {
                    "status": "blocked" if stage == "preparation" else "pending",
                    **({"error": error} if stage == "preparation" else {}),
                }
                for stage in STAGES
            },
        },
    )
    from agent_hotwash.report.memory import render_memory_html, render_memory_markdown

    evidence = {"sessions": [], "records": [], "analytics": {}}
    atomic_text(output / "report.md", render_memory_markdown(evidence, result))
    atomic_text(output / "report.html", render_memory_html(evidence, result))


def run_audit(
    paths, output, *, config, extract=False, allow_paid=False, guidance_paths=(), cache=None, run_options=None
):
    from .inputs import prepare_inputs

    with audit_lock(output):
        try:
            evidence = _validated_evidence(
                prepare_inputs(
                    paths, memory_config=config.memory, run_options=run_options, secret_patterns=config.lexicons.secret
                )
            )
            snapshot = (
                prepare_guidance(guidance_paths, policy=config.memory, secret_patterns=config.lexicons.secret)
                if guidance_paths
                else None
            )
        except Exception as exc:
            _preparation_failed(output, config, exc)
            raise
        return _execute(
            evidence, output, config=config, extract=extract, allow_paid=allow_paid, snapshot=snapshot, cache=cache
        )


def compare_saved(audit_dir, output, *, config, guidance_paths=(), allow_paid=False, cache=None):
    evidence, saved = load_saved(audit_dir)
    if saved.get("status") not in ("complete", "blocked") or "candidates" not in saved:
        raise ValueError("Saved audit has no completed proposals to compare")
    synthesis = _read(Path(audit_dir) / "manifest.json").get("stages", {}).get("synthesis", {})
    if synthesis.get("status") not in ("complete", "skipped"):
        raise ValueError("Saved audit synthesis did not complete")
    with audit_lock(output):
        try:
            snapshot = (
                prepare_guidance(guidance_paths, policy=config.memory, secret_patterns=config.lexicons.secret)
                if guidance_paths
                else None
            )
        except Exception as exc:
            _preparation_failed(output, config, exc)
            raise
        return _execute(
            evidence,
            output,
            config=config,
            extract=True,
            allow_paid=allow_paid,
            snapshot=snapshot,
            cache=cache or ex.ProviderCache(Path(audit_dir) / "calls", allow_paid=allow_paid, config=config),
            saved=saved,
        )
