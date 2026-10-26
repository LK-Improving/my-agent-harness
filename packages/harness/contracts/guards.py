"""单调工具守卫契约：只允许拒绝或不表态，不能生成越权放行。"""

from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ExecutionContext, Identifier


class GuardVerdict(StrEnum):
    ABSTAIN = "abstain"
    DENY = "deny"


class GuardedToolCall(Contract):
    context: ExecutionContext
    operation_id: Identifier
    logical_call_id: Identifier
    tool_name: Identifier
    tool_version: Identifier
    arguments: JsonSnapshot

    @model_validator(mode="after")
    def validate_arguments(self) -> Self:
        self.arguments.as_object()
        return self

    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class GuardDecision(Contract):
    guard_ref: Identifier
    execution_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: GuardVerdict
    reason_code: Identifier | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> Self:
        if self.verdict == GuardVerdict.DENY and not self.reason_code:
            raise ValueError("拒绝决定必须有可审计原因码")
        return self


class ToolGuard(Protocol):
    def evaluate(self, execution: GuardedToolCall) -> GuardDecision: ...


class GuardSummary(Contract):
    execution_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    decisions: tuple[GuardDecision, ...]

    @model_validator(mode="after")
    def validate_decisions(self) -> Self:
        refs = [decision.guard_ref for decision in self.decisions]
        if len(set(refs)) != len(refs):
            raise ValueError("同一守卫不能重复提交决定")
        if any(
            decision.execution_fingerprint != self.execution_fingerprint
            for decision in self.decisions
        ):
            raise ValueError("守卫决定不属于同一固定工具调用")
        return self

    @property
    def denied(self) -> bool:
        return any(decision.verdict == GuardVerdict.DENY for decision in self.decisions)


def compose_guard_decisions(
    execution: GuardedToolCall,
    decisions: tuple[GuardDecision, ...],
) -> GuardSummary:
    """保留所有拒绝；全不表态也不代表审批、权限和预算已经通过。"""
    return GuardSummary(execution_fingerprint=execution.fingerprint(), decisions=decisions)
