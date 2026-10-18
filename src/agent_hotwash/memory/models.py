"""Versioned contracts at memory preparation, persistence and report boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EVIDENCE_VERSION = "memory-evidence-v2"
AUDIT_VERSION = "memory-audit-v2"


class Contract(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def validate_dict(cls, value: Any) -> dict[str, Any]:
        return cls.model_validate(value).model_dump(mode="json")


class SourceArtifact(Contract):
    source_id: str
    source_path: str
    source_sha256: str
    bytes: int = Field(ge=0)


class EvidenceSession(Contract):
    session_id: str
    source_path: str | None = None
    source_sha256: str | None = None
    native_session_id_known: bool = False
    session_identity_basis: str = "unknown"
    independence_group_id: str | None = None
    independence_confidence: Literal["known", "unknown"] = "unknown"
    independent_session_known: bool = False
    reasoning_visibility: str = "none_recorded"
    source_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)


class EvidenceRecord(Contract):
    id: str
    session_id: str
    role: Literal["user", "assistant", "tool"]
    authorship: Literal["human", "delegated", "injected", "unknown"] = "unknown"
    kind: Literal["message", "reasoning", "tool_call", "tool_result"]
    text: str
    paths: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    line_number: int | None = Field(None, ge=1)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    canonical_event_identity: str
    canonical_event_index: int = Field(ge=0)
    independence_group_id: str | None = None
    independence_confidence: Literal["known", "unknown"] = "unknown"
    independent_session_known: bool = False
    original_chars: int = Field(ge=0)
    retained_chars: int = Field(ge=0)
    truncated: bool = False
    output_sha256: str | None = None
    output_chars_original: int | None = None
    output_lines_original: int | None = None
    output_truncated: bool = False
    correction_antecedent_ids: list[str] = Field(default_factory=list)


class EvidenceBundle(Contract):
    schema_version: Literal["memory-evidence-v2"] = EVIDENCE_VERSION
    sources: list[SourceArtifact] = Field(default_factory=list)
    sessions: list[EvidenceSession]
    records: list[EvidenceRecord]
    input_coverage: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    analytics: dict[str, Any]


AuditEvidenceBundle = EvidenceBundle


class AuditProposal(Contract):
    id: str
    category: Literal["user_corrections", "repeated_discovery", "repeated_reading"]
    statement: str
    applicability: str
    intervention: Literal["instruction", "skill", "documentation", "navigation_pointer", "tooling"]
    evidence_ids: list[str]
    paths: list[str]
    draft: str
    reason: str
    supporting_independence_groups: list[str] = Field(default_factory=list)
    installation_status: Literal["review_draft"] = "review_draft"


class AuditResult(Contract):
    schema_version: Literal["memory-audit-v2"] = AUDIT_VERSION
    status: Literal["running", "preprocessed", "complete", "blocked"]
    candidates: list[AuditProposal] = Field(default_factory=list)
    features: list[dict[str, Any]] = Field(default_factory=list)
    proposal_assessments: dict[str, Any] = Field(default_factory=dict)
    guidance_assessments: dict[str, Any] = Field(default_factory=dict)
    calls: list[dict[str, Any]] = Field(default_factory=list)
