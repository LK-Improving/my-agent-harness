"""Agent、工具与执行身份的版本化配置契约。"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[\w.@:/-]+$")]
PositiveLimit = Annotated[int, Field(strict=True, gt=0)]


class Contract(BaseModel):
    """拒绝未知字段并冻结字段赋值；嵌套 JSON 边界另用 JsonSnapshot。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionContext(Contract):
    tenant_id: Identifier
    subject_id: Identifier
    application_id: Identifier
    roles: frozenset[str] = frozenset()
    resource_scopes: frozenset[str] = frozenset()
    run_id: Identifier
    release_id: Identifier
    trace_id: Identifier

    @field_serializer("roles", "resource_scopes")
    def serialize_scopes(self, value: frozenset[str]) -> list[str]:
        return sorted(value)


class Effect(StrEnum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"


class RetryPolicy(Contract):
    max_attempts_per_operation: PositiveLimit = 1
    strategy: Literal["never", "bounded_read", "same_key_after_reconcile"] = "never"

    @model_validator(mode="after")
    def validate_attempts(self) -> Self:
        if self.strategy == "never" and self.max_attempts_per_operation != 1:
            raise ValueError("禁止重试策略只能配置一次尝试")
        return self


class IdempotencySpec(Contract):
    provider_supported: bool
    provider_key_source: Literal["operation_id"] = "operation_id"
    retention_seconds: PositiveLimit | None = None

    @model_validator(mode="after")
    def validate_retention(self) -> Self:
        if self.provider_supported and self.retention_seconds is None:
            raise ValueError("供应商幂等必须明确有效期")
        return self


class CostSpec(Contract):
    billable: bool
    upper_bound: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)] | None = None
    currency: Literal["CNY", "USD"] = "CNY"
    price_version: Identifier | None = None

    @model_validator(mode="after")
    def validate_cost(self) -> Self:
        if not self.billable and self.upper_bound != Decimal("0"):
            raise ValueError("免费工具必须显式声明费用上界为零")
        if self.billable and self.upper_bound is not None and self.price_version is None:
            raise ValueError("已知费用必须绑定价格版本")
        return self


class ToolSpec(Contract):
    schema_version: Literal[1] = 1
    name: Identifier
    version: Identifier
    handler_ref: Identifier
    effect: Effect
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    authorization_policy_ref: Identifier
    approval_policy_ref: Identifier | None = None
    timeout_ms: PositiveLimit
    max_concurrency: PositiveLimit = 1
    retry: RetryPolicy = RetryPolicy()
    idempotency: IdempotencySpec | None = None
    business_action_key_resolver_ref: Identifier | None = None
    reconcile_handler_ref: Identifier | None = None
    cost: CostSpec
    sensitive_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_write(self) -> Self:
        if self.effect == Effect.WRITE:
            if not all(
                (
                    self.approval_policy_ref,
                    self.business_action_key_resolver_ref,
                    self.reconcile_handler_ref,
                )
            ):
                raise ValueError("基础版写工具必须声明审批、业务动作身份与核对入口")
            if self.max_concurrency != 1:
                raise ValueError("基础版写工具仅支持顺序派发")
            if self.retry.strategy == "bounded_read":
                raise ValueError("写工具不能使用普通只读重试")
        if self.retry.strategy == "same_key_after_reconcile":
            if not self.idempotency or not self.idempotency.provider_supported:
                raise ValueError("同 key 重试要求明确供应商幂等保证")
            if not self.reconcile_handler_ref:
                raise ValueError("核对后重试必须有核对入口")
        return self


class ToolBinding(Contract):
    name: Identifier
    version: Identifier
    max_operations_per_run: PositiveLimit


class KnowledgeBinding(Contract):
    knowledge_base_id: Identifier
    snapshot_id: Identifier
    retrieval_profile_ref: Identifier


class ExecutionLimits(Contract):
    max_model_steps: PositiveLimit = 8
    max_tool_dispatches: PositiveLimit = 12
    read_concurrency: PositiveLimit = 4


class AgentSpec(Contract):
    schema_version: Literal[1] = 1
    agent_id: Identifier
    release_id: Identifier
    graph_template: Literal["bounded_tool_agent", "rag_chat"]
    graph_version: Identifier
    prompt_ref: Identifier
    model_profile_ref: Identifier
    knowledge_bindings: tuple[KnowledgeBinding, ...] = ()
    allowed_tools: tuple[ToolBinding, ...] = ()
    policy_profile_ref: Identifier
    budget_profile_ref: Identifier
    memory_profile_ref: Identifier
    evaluation_profile_ref: Identifier
    completion_validator_ref: Identifier
    timeout_profile_ref: Identifier
    stream_profile_ref: Identifier
    execution_limits: ExecutionLimits = ExecutionLimits()

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        names = [tool.name for tool in self.allowed_tools]
        if len(names) != len(set(names)):
            raise ValueError("同一 Agent 不允许重复注册工具名称")
        if self.graph_template == "rag_chat" and not self.knowledge_bindings:
            raise ValueError("知识模板必须绑定知识快照")
        return self
