"""M2 精确批准、稳定写入身份与三态核对；不读取环境或调用外部系统。"""

from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from harness.contracts.runtime import ClaimedRun, Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier, PositiveLimit, ToolSpec
from harness.contracts.state import ReconciliationConclusion, ToolReceipt


class WritePolicy(Contract):
    version: Identifier = "exact-write@1"
    approval_ttl_seconds: PositiveLimit = 600
    max_reconciliations_per_operation: PositiveLimit = 5
    reconciliation_timeout_seconds: PositiveLimit = 10
    max_result_bytes: PositiveLimit = 1048576


class WriteProposal(Contract):
    run_id: Identifier
    logical_call_id: Identifier
    tool: JsonSnapshot
    arguments: JsonSnapshot
    action_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_snapshots(self) -> Self:
        ToolSpec.model_validate(self.tool.as_object())
        self.arguments.as_object()
        return self


class WriteOperation(Contract):
    operation_id: Identifier
    run_id: Identifier
    logical_call_id: Identifier
    principal: Principal
    tool: JsonSnapshot
    arguments: JsonSnapshot
    action_key: str
    plan_hash: str
    binding_hash: str
    approval_id: Identifier
    state: Literal["prepared", "dispatched", "unknown", "succeeded", "not_executed", "rejected"]
    receipt: ToolReceipt | None = None
    output_error: str | None = None


class ApprovalDecisionRequest(Contract):
    request_id: Identifier
    binding_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_revision: int = Field(ge=0, strict=True)
    decision: Literal["approve", "reject"]


class ReconcileRequest(Contract):
    request_id: Identifier
    operation_id: Identifier


class DispatchPermit(Contract):
    operation: WriteOperation
    attempt_id: Identifier
    deadline_at: AwareDatetime


class ExternalWriteResult(Contract):
    """供应商已确认副作用；即使data不合Schema也必须保留这个事实。"""

    external_reference: str = Field(min_length=1, max_length=256)
    data: JsonValue


class ReconciliationResult(Contract):
    operation_id: Identifier
    args_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    conclusion: ReconciliationConclusion
    evidence_ref: str = Field(min_length=1, max_length=256)
    # authoritative_not_executed只能由可信供应商适配器提供，不能接受客户端结论。
    authoritative_not_executed: bool = False
    result: ExternalWriteResult | None = None
    output_error: Literal["TOOL_OUTPUT_INVALID", "TOOL_RESULT_TOO_LARGE"] | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.conclusion == ReconciliationConclusion.SUCCEEDED and self.result is None:
            raise ValueError("确认成功必须有外部回执")
        if (
            self.conclusion == ReconciliationConclusion.NOT_EXECUTED
            and not self.authoritative_not_executed
        ):
            raise ValueError("未执行必须有权威证明，单次未查到不足以证明")
        if self.conclusion != ReconciliationConclusion.SUCCEEDED and self.result is not None:
            raise ValueError("非成功核对不得附加成功回执")
        return self


class ReconciliationPermit(Contract):
    token: Identifier
    operation: WriteOperation
    deadline_at: AwareDatetime
    recorded_result: ReconciliationResult | None = None


class SuspensionProof(Contract):
    """可信引擎查验实际checkpoint后给出；不作为HTTP输入。"""

    run_id: Identifier
    operation_id: Identifier
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoint_ref: Identifier
    interrupt_ref: Identifier


class ExecutionSuspended(Exception):
    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__("WRITE_EXECUTION_SUSPENDED")


class WriteHandler(Protocol):
    async def invoke(self, permit: DispatchPermit) -> ExternalWriteResult: ...

    async def reconcile(self, permit: ReconciliationPermit) -> ReconciliationResult: ...


class SuspendingEngine(Protocol):
    async def suspension_proof(self, run: ClaimedRun, operation_id: str) -> SuspensionProof: ...


class WriteLedger(Protocol):
    async def prepare_write(
        self, actor: Principal, owner: str, proposal: WriteProposal
    ) -> WriteOperation: ...

    async def begin_dispatch(
        self, actor: Principal, owner: str, operation_id: str
    ) -> DispatchPermit: ...

    async def record_write_result(
        self, permit: DispatchPermit, result: ExternalWriteResult, output_error: str | None
    ) -> ToolReceipt: ...

    async def mark_write_unknown(self, permit: DispatchPermit) -> None: ...

    async def decide_approval(
        self, actor: Principal, approval_id: str, request: ApprovalDecisionRequest
    ) -> dict[str, JsonValue]: ...

    async def list_approvals(
        self, actor: Principal, after: str, limit: int
    ) -> list[dict[str, JsonValue]]: ...

    async def begin_reconciliation(
        self, actor: Principal, run_id: str, request: ReconcileRequest
    ) -> ReconciliationPermit: ...

    async def finish_reconciliation(
        self, actor: Principal, permit: ReconciliationPermit, result: ReconciliationResult
    ) -> WriteOperation: ...

    async def park_write(self, owner: str, proof: SuspensionProof) -> None: ...

    async def write_receipts(self, actor: Principal, run_id: str) -> tuple[ToolReceipt, ...]: ...
