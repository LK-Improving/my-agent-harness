"""运行状态、回执、任务进度与完成验证的纯契约。"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, JsonValue

from harness.contracts.specs import Contract, Identifier


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    RECONCILING = "reconciling"
    COMMITTING = "committing"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATES = frozenset(
    {RunState.SUCCEEDED, RunState.PARTIAL, RunState.FAILED, RunState.CANCELLED, RunState.EXPIRED}
)

_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.QUEUED: frozenset({RunState.RUNNING}),
    RunState.RUNNING: frozenset(
        {RunState.WAITING_APPROVAL, RunState.RECONCILING, RunState.COMMITTING}
    ),
    RunState.WAITING_APPROVAL: frozenset({RunState.QUEUED}),
    RunState.RECONCILING: frozenset({RunState.QUEUED}),
    RunState.COMMITTING: frozenset({RunState.SUCCEEDED, RunState.PARTIAL}),
}


def validate_transition(current: RunState, target: RunState, *, has_unknown: bool = False) -> None:
    """校验纯状态边界；权限、CAS、审批和取消条件仍由协调器负责。"""
    if current in TERMINAL_STATES:
        if current != target:
            raise ValueError("终态不可重新执行")
        return
    if has_unknown and target not in {RunState.RECONCILING}:
        raise ValueError("存在未知副作用时必须先核对")
    if target in {RunState.FAILED, RunState.CANCELLED, RunState.EXPIRED}:
        return
    if has_unknown and target == RunState.RECONCILING:
        return
    if target not in _TRANSITIONS.get(current, frozenset()):
        raise ValueError(f"非法状态迁移：{current} → {target}")


class ReceiptState(StrEnum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ReconciliationConclusion(StrEnum):
    SUCCEEDED = "confirmed_succeeded"
    NOT_EXECUTED = "confirmed_not_executed"
    UNKNOWN = "still_unknown"


class ToolReceipt(Contract):
    operation_id: Identifier
    attempt_id: Identifier
    status: ReceiptState
    args_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    external_reference: str | None = None
    artifact_ref: Identifier | None = None
    data: JsonValue = None
    error_code: str | None = None
    cost_status: Literal["known", "unknown", "not_billable"] = "unknown"


class Constraint(Contract):
    key: str
    value: JsonValue
    source_turn_id: Identifier


class ValidationReport(Contract):
    validator_ref: Identifier
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: Literal["pass", "partial", "needs_clarification", "fail"]
    checks: dict[str, bool]
    failed_constraints: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    checked_at: AwareDatetime


class TaskState(Contract):
    objective: str
    source_turn_id: Identifier
    constraints: tuple[Constraint, ...] = ()
    completed_items: tuple[str, ...] = ()
    pending_items: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    validation: ValidationReport | None = None


class RunDeadlines(Contract):
    queue_deadline_at: AwareDatetime
    execution_budget_ms: int = Field(gt=0, strict=True)
    execution_used_ms: int = Field(ge=0, strict=True)
    run_expires_at: AwareDatetime
    approval_expires_at: AwareDatetime | None = None
    reconciliation_due_at: AwareDatetime | None = None

    def execution_remaining_ms(self, now: datetime) -> int:
        """以调用方提供的权威时间计算剩余额度，不读取系统时钟。"""
        if now.tzinfo is None:
            raise ValueError("时间必须包含时区")
        return max(
            0,
            min(
                self.execution_budget_ms - self.execution_used_ms,
                int((self.run_expires_at - now).total_seconds() * 1000),
            ),
        )
