"""确定性门禁输入与证据契约；不提供发布授权。"""

from typing import Annotated, Literal, Protocol, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.retrieval_models import RetrievalLimits, RetrievalScope, RetrievalUsage
from harness.contracts.runtime import RunView
from harness.contracts.specs import Contract, Identifier

JudgeScore = Annotated[float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)]


class DispatchEvidence(Contract):
    attempt_id: Identifier
    ordinal: int = Field(ge=1, strict=True)
    state: str
    dispatched_at: AwareDatetime


class OperationEvidence(Contract):
    operation_id: Identifier
    tool_name: Identifier
    tool_hash: str
    effect: Literal["read", "write"]
    state: str
    resource_scopes: tuple[str, ...]
    args_hash: str
    receipt_args_hash: str | None = None
    action_key: str | None = None
    attempts: tuple[DispatchEvidence, ...] = ()
    recorded_attempts: int = Field(ge=0, strict=True)
    max_attempts: int | None = Field(default=None, ge=1, strict=True)
    approval_id: str | None = None
    binding_hash: str | None = None
    approval_binding_hash: str | None = None
    approval_state: str | None = None
    approval_expires_at: AwareDatetime | None = None
    approver: str | None = None
    external_reference_hash: str | None = None


class ModelCallEvidence(Contract):
    call_id: Identifier
    state: str
    provider: Identifier | None = None
    model: Identifier | None = None
    price_version_ref: Identifier | None = None
    input_tokens: int = Field(ge=0, strict=True)
    reserved_output_tokens: int = Field(ge=0, strict=True)
    reported_input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    response_content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    remaining_total_tokens: int = Field(ge=0, strict=True)


class RetrievalModelCallEvidence(Contract):
    """不含检索输入正文、向量或供应商原文的子调用记录。"""

    call_id: Identifier
    attempt_id: Identifier
    request_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    binding_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_type: Literal["embedding", "rerank"]
    input_bytes: int = Field(gt=0, strict=True)
    state: Literal["reserved", "succeeded", "failed"]
    result_call_id: Identifier | None = None
    result_attempt_id: Identifier | None = None
    result_fingerprint: str | None = None
    result_task_type: str | None = None
    usage: RetrievalUsage | None = None
    dispatch_status: Literal["not_sent", "unknown", "response_received"] | None = None
    billing_status: Literal["not_billable", "unknown"] | None = None


class RetrievalChildEvidence(Contract):
    retrieval_id: Identifier
    knowledge_base_id: Identifier
    state: Literal["queued", "running", "succeeded", "failed"]
    ticket_scope: RetrievalScope
    ticket_binding_valid: bool
    profile_limits: RetrievalLimits
    parent_reserved_tokens: int = Field(gt=0, strict=True)
    context_scope: RetrievalScope | None = None
    context_binding_valid: bool | None = None
    context_limits: RetrievalLimits | None = None
    calls_used: int | None = Field(default=None, ge=0, strict=True)
    input_bytes_used: int | None = Field(default=None, ge=0, strict=True)
    calls: tuple[RetrievalModelCallEvidence, ...] = Field(default=(), max_length=128)


class RunGateEvidence(Contract):
    version: Literal["pg-evaluation-evidence@1"] = "pg-evaluation-evidence@1"
    run_id: Identifier
    session_id: Identifier
    context_id: Identifier
    dataset_hash: str
    gate_policy_version: Identifier
    request_hash: str
    manifest_hash: str
    environment_profile_hash: str
    snapshot_at: AwareDatetime
    status: str
    requester_id: Identifier
    event_seq: int = Field(ge=0, strict=True)
    result_hash: str | None = None
    run_scopes: tuple[str, ...]
    context_scopes: tuple[str, ...]
    operations: tuple[OperationEvidence, ...] = Field(max_length=1000)
    model_calls: tuple[ModelCallEvidence, ...] = Field(max_length=1000)
    model_steps: int = Field(ge=0, strict=True)
    reserved_tokens: int = Field(ge=0, strict=True)
    tool_dispatches: int = Field(ge=0, strict=True)
    execution_used_ms: int = Field(ge=0, strict=True)
    max_input_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_total_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_execution_ms: int | None = Field(default=None, gt=0, strict=True)
    reserved_extension_tokens: int | None = Field(default=None, ge=0, strict=True)
    validation_checks: dict[str, bool] | None = None
    validation_verdict: str | None = None
    events: tuple[RuntimeEvent, ...] = Field(max_length=10000)
    event_hashes: tuple[str, ...] = Field(max_length=10000)
    missing: tuple[str, ...] = ()
    retrieval_children: tuple[RetrievalChildEvidence, ...] | None = Field(default=None, max_length=4)


class ExternalObservation(Contract):
    """由独立观测适配器生成；不能由PG操作行推断外部写入数。"""

    run_id: Identifier
    freeze_hash: str
    manifest_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    result_hash: str | None
    source_ref: Identifier
    event_seq: int = Field(ge=1, strict=True)
    unauthorized_deliveries: int = Field(ge=0, strict=True)
    credential_disclosures: int | None = Field(default=None, ge=0, strict=True)
    # 每个写operation的实际副作用数；不能把全租户累计计数当单operation计数。
    side_effect_counts: dict[str, Annotated[int, Field(ge=0, strict=True)]] = Field(default_factory=dict)
    coverage: tuple[Literal["authorization_delivery", "external_writes", "credential_exposure"], ...]


class JudgeInput(Contract):
    """固定 Judge 输入；材料来源仍由受信部署方和运行时权限验证。"""

    version: Literal["evaluation-judge-input@1"] = "evaluation-judge-input@1"
    case_id: Identifier
    target_run_id: Identifier
    target_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    freeze_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_result_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_seq: int = Field(ge=1, strict=True)
    message: str = Field(min_length=1, max_length=16000)
    attachment_versions: tuple[str, ...] = ()
    business_options: dict[str, JsonValue] = Field(default_factory=dict)
    expected: dict[str, JsonValue] = Field(min_length=1)
    dimensions: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    target_result: JsonValue
    material_source_ref: Identifier
    reference_material: JsonValue


class JudgeExecutionEvidence(Contract):
    """独立 Judge 运行的清单、事件、实际结果及持久账本快照。"""

    version: Literal["evaluation-judge-execution@1"] = "evaluation-judge-execution@1"
    manifest: ReleaseManifest
    context: EvaluationContext
    input: JudgeInput
    accepted: RunView
    final: RunView
    events: tuple[RuntimeEvent, ...] = Field(min_length=1, max_length=10000)
    ledger: RunGateEvidence
    model_response: str | None = Field(default=None, max_length=1048576, repr=False)


class JudgeOutput(Contract):
    version: Literal["evaluation-judge-output@1"] = "evaluation-judge-output@1"
    case_id: Identifier
    target_run_id: Identifier
    target_result_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    freeze_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scores: dict[str, JudgeScore]


class JudgeModelEnvelope(Contract):
    answer: str = Field(min_length=1, max_length=64000)
    answer_kind: Literal["answer"]
    task_outcome: Literal["achieved"]
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=40)


class JudgeModelResponse(Contract):
    run_id: Identifier
    model_call_id: Identifier
    content: str = Field(min_length=1, max_length=1048576, repr=False)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        if canonical_hash(self.content) != self.content_hash:
            raise ValueError("Judge 模型响应 hash 不匹配")
        return self


class JudgeSourceMaterial(Contract):
    """经当前授权读取的目标运行参考材料。"""

    version: Literal["evaluation-judge-material@1"] = "evaluation-judge-material@1"
    run_id: Identifier
    context_id: Identifier
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_seq: int = Field(ge=1, strict=True)
    source_ref: Identifier
    material: JsonValue = Field(repr=False)
    material_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_material_hash(self) -> Self:
        if canonical_hash(self.material) != self.material_hash:
            raise ValueError("Judge 参考材料 hash 不匹配")
        return self


class JudgeReceipt(Contract):
    version: Literal["evaluation-judge-receipt@1"] = "evaluation-judge-receipt@1"
    run_id: Identifier
    freeze_hash: str
    result_hash: str
    policy_version: Identifier
    implementation: Identifier
    prompt_sha256: str
    model_binding_ref: Identifier
    model_call_id: Identifier
    receipt_ref: Identifier
    state: Literal["succeeded", "failed", "timeout", "invalid"]
    scores: dict[str, JudgeScore] = Field(default_factory=dict)


class GateCheck(Contract):
    gate: Identifier
    outcome: Literal["passed", "incomplete", "failed"]
    reasons: tuple[Identifier, ...] = ()


class CaseGateReport(Contract):
    case_id: Identifier
    outcome: Literal["not_run", "incomplete", "failed", "passed"]
    checks: tuple[GateCheck, ...]


class EvaluationEvidenceAPI(Protocol):
    async def evaluation_evidence(self, run_id: str) -> RunGateEvidence: ...
