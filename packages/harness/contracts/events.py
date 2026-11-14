"""公开事件、回答草稿与授权分页读取的数据契约。"""

from typing import Literal, Self

from pydantic import AwareDatetime, ConfigDict, Field, JsonValue, model_validator

from harness.contracts.evaluation_versions import ExecutionVersion
from harness.contracts.specs import Contract, Identifier


class EventPayload(Contract):
    # 新增可选字段可以随原始事件保留，类型视图只暴露已知字段。
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class RunAcceptedPayload(EventPayload):
    release_id: Identifier
    manifest_hash: str
    environment_profile_hash: str


class RunStartedPayload(EventPayload):
    observed_manifest_hash: str
    execution_identity: Identifier
    version_evidence_seq: int | None = Field(default=None, ge=1)
    observed_environment_profile_hash: str | None = None


class AnswerStartedPayload(EventPayload):
    draft_id: Identifier
    producer: str
    model_call_id: Identifier | None = None
    model_attempt_id: Identifier | None = None

    @model_validator(mode="after")
    def require_model_binding(self) -> Self:
        if self.producer == "model" and (self.model_call_id is None or self.model_attempt_id is None):
            raise ValueError("模型草稿必须包含调用与尝试身份")
        return self


class AnswerDeltaPayload(EventPayload):
    draft_id: Identifier
    model_attempt_id: Identifier
    delta: str


class AnswerInvalidatedPayload(EventPayload):
    draft_id: Identifier
    reason: str


class RunCompletedPayload(EventPayload):
    committed_draft_id: Identifier
    commit_id: Identifier
    result_hash: str
    version_evidence_seq: int | None = Field(default=None, ge=1)
    observed_manifest_hash: str | None = None
    observed_environment_profile_hash: str | None = None


class RunTerminatedPayload(EventPayload):
    code: str
    version_evidence_seq: int | None = Field(default=None, ge=1)
    observed_manifest_hash: str | None = None
    observed_environment_profile_hash: str | None = None


class RunVersionObservedPayload(EventPayload):
    phase: Literal["start", "terminal"]
    execution_identity: Identifier
    observation: ExecutionVersion | None
    error_code: Identifier | None


class ApprovalRequiredPayload(EventPayload):
    approval_id: Identifier
    operation_id: Identifier
    binding_hash: str


class ApprovalDecidedPayload(EventPayload):
    approval_id: Identifier
    decision: str


class ArtifactSavedPayload(EventPayload):
    artifact_ref: Identifier
    session_id: Identifier
    content_hash: str
    source_operation_id: Identifier


PAYLOAD_TYPES: dict[str, type[EventPayload]] = {
    "run.accepted": RunAcceptedPayload,
    "run.started": RunStartedPayload,
    "run.version_observed": RunVersionObservedPayload,
    "answer.started": AnswerStartedPayload,
    "answer.delta": AnswerDeltaPayload,
    "answer.invalidated": AnswerInvalidatedPayload,
    "run.completed": RunCompletedPayload,
    "run.failed": RunTerminatedPayload,
    "run.cancelled": RunTerminatedPayload,
    "run.expired": RunTerminatedPayload,
    "artifact.saved": ArtifactSavedPayload,
    "approval.required": ApprovalRequiredPayload,
    "approval.decided": ApprovalDecidedPayload,
}


class RuntimeEvent(Contract):
    schema_version: Literal[1] = 1
    run_id: Identifier
    seq: int = Field(ge=1, strict=True)
    type: str
    timestamp: AwareDatetime
    payload: dict[str, JsonValue]

    def typed_payload(self) -> EventPayload | None:
        schema = PAYLOAD_TYPES.get(self.type)
        return schema.model_validate(self.payload) if schema is not None else None

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        required: dict[str, tuple[str, ...]] = {
            "run.accepted": ("release_id", "manifest_hash", "environment_profile_hash"),
            "run.started": ("observed_manifest_hash", "execution_identity"),
            "answer.started": ("draft_id", "producer"),
            "answer.delta": ("draft_id", "model_attempt_id", "delta"),
            "answer.invalidated": ("draft_id", "reason"),
            "run.completed": ("committed_draft_id", "commit_id", "result_hash"),
            "artifact.saved": ("artifact_ref", "session_id", "content_hash", "source_operation_id"),
        }
        for key in required.get(self.type, ()):
            if self.payload.get(key) is None:
                raise ValueError(f"事件 {self.type} 缺少 {key}")
        if self.type == "answer.started" and self.payload.get("producer") == "model":
            if not self.payload.get("model_call_id") or not self.payload.get("model_attempt_id"):
                raise ValueError("模型草稿必须包含调用与尝试身份")
        return self


class ArtifactPage(Contract):
    """M0同步探针页；保留旧字段，M4正式异步端口的v2页在contracts.artifacts。"""

    ref: Identifier
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_ref: Identifier
    offset: int = Field(ge=0, strict=True)
    next_offset: int = Field(ge=0, strict=True)
    has_more: bool
    content: str
    truncated: bool

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.next_offset != self.offset + len(self.content):
            raise ValueError("分页游标必须按实际 Unicode 字符数推进")
        return self
