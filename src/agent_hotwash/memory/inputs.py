"""Canonical evidence projection; all harness interpretation lives in sources."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from agent_hotwash.events import ArtifactOp, EventKind
from agent_hotwash.semantic.redact import redact_audit_text as redact
from agent_hotwash.semantic.redact import redact_audit_value
from agent_hotwash.sources.detect import discover_many, load_unit

from .models import EVIDENCE_VERSION, EvidenceBundle


class EmptyTraceInputError(ValueError):
    """No supported evidence was found in an explicitly selected corpus."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _session_key(session):
    if not session.native_session_id_known:
        source = session.source_artifacts[0]["source_sha256"] if session.source_artifacts else _hash(session.session_id)
        return f"{session.agent}:source:{source[:20]}"
    return f"{session.agent}:{redact(session.session_id)}"


def _families(traces):
    """Count native root families; missing or ambiguous linkage stays unknown."""
    sessions = {_session_key(s): s for trace in traces for s in (trace.root, *trace.subagents)}
    parent = dict.fromkeys(sessions)
    for key in parent:
        parent[key] = key

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        left, right = find(a), find(b)
        if left != right:
            parent[max(left, right)] = min(left, right)

    unknown, hashes = set(), {}
    for trace in traces:
        root = _session_key(trace.root)
        for session in (trace.root, *trace.subagents):
            key = _session_key(session)
            union(root, key)
            if (
                not session.native_session_id_known
                or "thread_linkage" in session.degraded
                or "usage_estimated" in session.degraded
            ):
                unknown.add(key)
            if session.parent_session_id:
                ancestor = f"{session.agent}:{redact(session.parent_session_id)}"
                if ancestor in sessions:
                    union(key, ancestor)
                else:
                    unknown.add(key)
            for artifact in session.source_artifacts:
                sha = artifact["source_sha256"]
                if sha in hashes:
                    union(key, hashes[sha])
                else:
                    hashes[sha] = key
        if trace.provenance.thread_linkage == "partial":
            unknown.add(root)
        for link in trace.links:
            child, ancestor = f"{trace.agent}:{redact(link.child_id)}", f"{trace.agent}:{redact(link.parent_id)}"
            if child in sessions and ancestor in sessions:
                union(child, ancestor)
            elif child in sessions:
                unknown.add(child)
    bad = {find(key) for key in unknown}
    return {key: ("family-" + _hash(find(key))[:20], "unknown" if find(key) in bad else "known") for key in sessions}


def _analytics(records, sessions, sources, duplicates):
    counts, file_sessions, file_groups, evidence = Counter(), defaultdict(set), defaultdict(set), defaultdict(list)
    tools, roles = Counter(), Counter()
    for record in records:
        roles[record["role"]] += 1
        if record["kind"] != "tool_call":
            continue
        tools[record["tool_name"] or "unknown"] += 1
        for path in record["paths"]:
            counts[path] += 1
            file_sessions[path].add(record["session_id"])
            if record["independence_confidence"] == "known":
                file_groups[path].add(record["independence_group_id"])
            evidence[path].append(record["id"])
    files = [
        {
            "path": path,
            "read_count": count,
            "session_count": len(file_sessions[path]),
            "independent_group_count": len(file_groups[path]),
            "repeated_within_sessions": count - len(file_sessions[path]),
            "evidence_ids": evidence[path],
        }
        for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "session_count": len(sessions),
        "record_count": len(records),
        "role_counts": dict(roles),
        "tool_calls": dict(tools),
        "most_opened_files": files,
        "independent_group_count": len(
            {s["independence_group_id"] for s in sessions if s["independence_confidence"] == "known"}
        ),
        "tool_results_truncated": sum(r["truncated"] for r in records),
        "original_tool_result_chars": sum(r["original_chars"] for r in records if r["kind"] == "tool_result"),
        "retained_tool_result_chars": sum(len(r["text"]) for r in records if r["kind"] == "tool_result"),
        "file_count_basis": "canonical_read_artifacts",
        "source_artifact_count": len(sources),
        "overlapping_snapshot_events_skipped": duplicates,
        "native_session_count": sum(s["native_session_id_known"] for s in sessions),
    }


def prepare_inputs(paths, *, memory_config=None, run_options=None, secret_patterns=()):
    """Discover a corpus once and project only public canonical evidence fields."""
    edge = memory_config.tool_edge_chars if memory_config is not None else 1000
    if not 1 <= edge <= 1000:
        raise ValueError("Canonical tool capture supports between 1 and 1000 characters per edge")
    paths = [Path(path).expanduser().resolve() for path in paths]
    diagnostics = []
    try:
        units = discover_many(paths, diagnostics=diagnostics)
    except ValueError as exc:
        raise EmptyTraceInputError(redact(str(exc))) from None
    if not units:
        raise EmptyTraceInputError("No recognized trace sources in the selected inputs")
    traces = []
    for unit in units:
        try:
            traces.extend(load_unit(unit))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"Recognized source decoder failed for {redact(unit.label)} ({type(exc).__name__})"
            ) from exc
    if run_options is not None:
        from agent_hotwash.runner import trace_matches_options

        traces = [trace for trace in traces if trace_matches_options(trace, run_options)]
    families = _families(traces)
    records, sessions, sources = {}, {}, {}
    duplicate_events = 0
    mapping = {
        EventKind.user_msg: ("user", "message"),
        EventKind.assistant_msg: ("assistant", "message"),
        EventKind.thinking: ("assistant", "reasoning"),
        EventKind.tool_call: ("assistant", "tool_call"),
        EventKind.tool_result: ("tool", "tool_result"),
    }
    for trace in traces:
        for session in (trace.root, *trace.subagents):
            key = _session_key(session)
            group, confidence = families[key]
            artifacts = redact_audit_value(session.source_artifacts)
            for artifact in artifacts:
                source_id = "source-" + _hash([artifact["source_sha256"], artifact["source_path"]])[:20]
                sources[source_id] = {"source_id": source_id, **artifact}
            diagnostics.extend(redact_audit_value(session.source_diagnostics))
            metadata = sessions.setdefault(
                key,
                {
                    "session_id": key,
                    "source_path": artifacts[0]["source_path"] if artifacts else None,
                    "source_sha256": artifacts[0]["source_sha256"] if artifacts else None,
                    "bytes": artifacts[0]["bytes"] if artifacts else 0,
                    "source_artifacts": artifacts,
                    "session_identity_basis": session.identity_basis,
                    "native_session_id_known": session.native_session_id_known,
                    "independence_group_id": group,
                    "independence_confidence": confidence,
                    "independent_session_known": confidence == "known",
                    "capabilities": session.capabilities.model_dump(mode="json"),
                    "degraded": redact_audit_value(session.degraded),
                    "source_format": trace.provenance.source_format,
                    "reasoning_visibility": "none_recorded",
                    "plaintext_reasoning_records": 0,
                    "encrypted_reasoning_items": 0,
                },
            )
            for artifact in artifacts:
                if artifact not in metadata["source_artifacts"]:
                    metadata["source_artifacts"].append(artifact)
            antecedent = None
            for event in session.events:
                if event.kind not in mapping:
                    continue
                role, kind = mapping[event.kind]
                encrypted = kind == "reasoning" and event.reasoning_encrypted
                metadata["encrypted_reasoning_items"] += int(encrypted)
                arguments = redact_audit_value(event.tool_args)
                text = (
                    json.dumps(arguments, ensure_ascii=False)
                    if kind == "tool_call"
                    else (event.output_excerpt if event.output_excerpt is not None else event.output or "")
                    if kind == "tool_result"
                    else event.text or ""
                )
                text = redact(text)
                if kind == "tool_result" and len(text) > 2 * edge:
                    text = text[:edge] + "\n[... tool result truncated ...]\n" + text[-edge:]
                if not text and kind != "tool_call":
                    continue
                duplicate_events += len(event.alternate_sources)
                refs = [
                    redact_audit_value(ref.model_dump(mode="json"))
                    for ref in ([event.source] if event.source else []) + event.alternate_sources
                ]
                source = refs[0] if refs else {}
                native = event.source.item_id if event.source else None
                native = native or event.call_id or event.span_id
                local = (
                    [native, event.source.block_ordinal, kind]
                    if native and event.source
                    else [
                        event.source.record_index if event.source else event.idx,
                        event.source.block_ordinal if event.source else None,
                        kind,
                    ]
                )
                identity = _hash([key, local])
                evidence_id = "evidence-" + identity[:24]
                if evidence_id in records:
                    duplicate_events += 1
                    existing = records[evidence_id]
                    if existing["text"] != text or existing["arguments"] != arguments:
                        raise ValueError(
                            "Conflicting canonical exports for one event identity; select a consistent source"
                        )
                    for ref in refs:
                        if ref not in existing["source_refs"]:
                            existing["source_refs"].append(ref)
                    continue
                authorship = event.authorship if role == "user" else "unknown"
                if (
                    role == "user"
                    and (session.parent_session_id or session is not trace.root)
                    and authorship == "human"
                ):
                    authorship = "delegated"
                record = {
                    "id": evidence_id,
                    "session_id": key,
                    "role": role,
                    "kind": kind,
                    "text": text,
                    "authorship": authorship,
                    "is_meta": role == "user" and authorship != "human",
                    "user_source": authorship,
                    "arguments": arguments,
                    "tool_name": event.tool_name,
                    "paths": sorted({redact(a.path) for a in event.artifacts if a.op == ArtifactOp.read})
                    if kind == "tool_call"
                    else [],
                    "source_path": source.get("path"),
                    "source_sha256": source.get("source_sha256"),
                    "line_number": source.get("line_number"),
                    "source_refs": refs,
                    "source_coordinates": source or None,
                    "coordinate_basis": "physical_jsonl_line" if source.get("line_number") else "canonical_event_only",
                    "canonical_event_identity": identity,
                    "canonical_event_index": event.idx,
                    "native_id": redact(event.call_id or native) if event.call_id or native else None,
                    "independence_group_id": group,
                    "independence_confidence": confidence,
                    "independent_session_known": confidence == "known",
                    "original_chars": event.output_chars_original or len(text) if kind == "tool_result" else len(text),
                    "retained_chars": len(text),
                    "truncated": kind == "tool_result" and (event.output_truncated or len(text) > 2 * edge),
                    "output_sha256": event.output_sha256 if kind == "tool_result" else None,
                    "output_chars_original": event.output_chars_original if kind == "tool_result" else None,
                    "output_lines_original": event.output_lines_original if kind == "tool_result" else None,
                    "output_truncated": event.output_truncated if kind == "tool_result" else False,
                    "truncation_basis": "canonical_captured_head_tail" if kind == "tool_result" else "none",
                    "correction_antecedent_ids": [antecedent] if role == "user" and antecedent else [],
                }
                records[evidence_id] = record
                if role == "assistant" and kind in ("message", "reasoning"):
                    antecedent = evidence_id
                if kind == "reasoning":
                    metadata["plaintext_reasoning_records"] += 1
            metadata["reasoning_visibility"] = (
                ("plaintext_with_encrypted_items" if metadata["encrypted_reasoning_items"] else "plaintext_recorded")
                if metadata["plaintext_reasoning_records"]
                else ("encrypted_only_unavailable" if metadata["encrypted_reasoning_items"] else "none_recorded")
            )
    session_list, record_list, source_list = list(sessions.values()), list(records.values()), list(sources.values())
    if not record_list:
        raise EmptyTraceInputError("No analyzable trace records found in recognized sources")
    bundle = {
        "schema_version": EVIDENCE_VERSION,
        "sources": source_list,
        "sessions": session_list,
        "records": record_list,
        "input_coverage": [
            {
                "input": redact(str(path)),
                "parser": "canonical_sources",
                "work_units": len(units),
                "scope": "shared_corpus",
            }
            for path in paths
        ],
        "diagnostics": list({_hash(item): item for item in diagnostics}.values()),
        "analytics": _analytics(record_list, session_list, source_list, duplicate_events),
    }
    bundle = redact_audit_value(bundle, secret_patterns)
    for record in bundle["records"]:
        record["retained_chars"] = len(record["text"])
    return EvidenceBundle.validate_dict(bundle)
