"""评测上下文的服务端准入、累计额度与单向撤销契约。"""

from typing import Literal

from pydantic import Field, field_serializer

from harness.contracts.specs import Contract, Identifier


class EvaluationLimits(Contract):
    max_sessions: int = Field(default=64, ge=1, le=1000, strict=True)
    max_runs: int = Field(default=128, ge=1, le=1000, strict=True)


class EvaluationAdmission(Contract):
    version: Identifier = "evaluation-admission@1"
    test_resource_scopes: frozenset[Identifier] = frozenset({"synthetic"})
    candidate_release_ids: frozenset[Identifier] | None = None
    max_valid_contexts_per_subject: int = Field(default=4, ge=1, le=100, strict=True)
    max_lifetime_seconds: int = Field(default=86400, ge=60, le=86400, strict=True)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)

    @field_serializer("test_resource_scopes", "candidate_release_ids")
    def ordered_sets(self, value: frozenset[str] | None) -> list[str] | None:
        return sorted(value) if value is not None else None


class EvaluationRevocationRequest(Contract):
    request_id: Identifier
    expected_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: Literal["operator_requested", "security", "dataset_invalid"] = "operator_requested"
