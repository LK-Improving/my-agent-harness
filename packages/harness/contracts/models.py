"""模型供应商端口与用量，不包含供应商连接或重试实现。"""

from datetime import datetime
from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, Field, field_serializer, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ExecutionContext, Identifier, PositiveLimit

ModelCapability = Literal["tool_calling", "structured_output", "vision", "streaming"]


class ResolvedModelRoute(Contract):
    provider: Identifier
    model: Identifier
    adapter_ref: Identifier
    configuration_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    capabilities: tuple[ModelCapability, ...] = ()
    capabilities_verified: bool = False
    context_window_tokens: PositiveLimit | None = None
    max_output_tokens: PositiveLimit
    price_version_ref: Identifier | None = None

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("模型能力不能重复声明")
        return self

    @field_serializer("capabilities")
    def serialize_capabilities(self, value: tuple[ModelCapability, ...]) -> list[str]:
        return sorted(value)


class ModelParameters(Contract):
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    stop: tuple[str, ...] | None = Field(default=None, max_length=4)
    seed: int | None = Field(default=None, strict=True)


class ModelBudgetQuote(Contract):
    """只描述固定输入的计数与额度快照；实际原子消费由 M1 协调器负责。"""

    quote_id: Identifier
    policy_ref: Identifier
    measured_input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_tokens: int = Field(ge=0, strict=True)
    output_reserved_tokens: PositiveLimit
    safety_margin_tokens: int = Field(default=0, ge=0, strict=True)
    remaining_total_tokens: PositiveLimit
    expires_at: AwareDatetime


def model_input_hash(
    route: ResolvedModelRoute,
    messages: JsonSnapshot,
    tools: JsonSnapshot,
    parameters: ModelParameters,
    max_output_tokens: int,
) -> str:
    """计数对象包括模型可见消息、工具、参数和实际模型能力/配置身份。"""
    return canonical_hash(
        {
            "route": route.model_dump(mode="json"),
            "messages": messages.materialize(),
            "tools": tools.materialize(),
            "parameters": parameters.model_dump(mode="json", exclude_none=True),
            "max_output_tokens": max_output_tokens,
        }
    )


class PreparedModelRequest(Contract):
    schema_version: Literal[1] = 1
    context: ExecutionContext
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    environment_profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_call_id: Identifier
    model_attempt_id: Identifier
    draft_id: Identifier
    route: ResolvedModelRoute
    messages: JsonSnapshot = Field(repr=False)
    tools: JsonSnapshot = Field(default_factory=lambda: JsonSnapshot.capture([]), repr=False)
    parameters: ModelParameters = ModelParameters()
    max_output_tokens: PositiveLimit
    budget: ModelBudgetQuote
    stream_policy: Literal["no_retry_after_publish"] = "no_retry_after_publish"

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        return self.validate_binding()

    def validate_binding(self) -> Self:
        messages = self.messages.as_object_array()
        tools = self.tools.as_object_array()
        if not messages:
            raise ValueError("模型请求至少包含一条消息")
        if tools and "tool_calling" not in self.route.capabilities:
            raise ValueError("工具请求必须绑定工具调用能力")
        if self.max_output_tokens > self.route.max_output_tokens:
            raise ValueError("输出上限超出已解析模型能力")
        if self.max_output_tokens != self.budget.output_reserved_tokens:
            raise ValueError("实际输出上限必须与预算预留一致")
        if self.budget.input_tokens + self.max_output_tokens > self.budget.remaining_total_tokens:
            raise ValueError("输入和输出预留超过整轮剩余额度")
        if self.budget.measured_input_hash != model_input_hash(
            self.route,
            self.messages,
            self.tools,
            self.parameters,
            self.max_output_tokens,
        ):
            raise ValueError("输入计数依据与固定请求不一致，必须重新准备")
        return self

    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def require_dispatchable(self, now: datetime) -> None:
        """不读取全局时钟，不扣减预算，也不替代服务端执行权/取消检查。"""
        self.validate_binding()
        if now.tzinfo is None:
            raise ValueError("派发时间必须包含时区")
        if now >= self.budget.expires_at:
            raise ValueError("请求输入预算已过期，必须重新准备")
        if not self.route.capabilities_verified or self.route.context_window_tokens is None:
            raise ValueError("实际模型能力和窗口尚未确认")
        required = (
            self.budget.input_tokens + self.max_output_tokens + self.budget.safety_margin_tokens
        )
        if required > self.route.context_window_tokens:
            raise ValueError("完整输入、输出预留和安全余量超出模型窗口")


class ModelUsage(Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ModelResult(Contract):
    model: str
    content: str | None
    tool_calls: JsonSnapshot = Field(default_factory=lambda: JsonSnapshot.capture([]))
    finish_reason: str | None
    request_id: str | None
    usage: ModelUsage
    model_call_id: Identifier
    model_attempt_id: Identifier
    request_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    draft_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_tool_calls(self) -> Self:
        self.tool_calls.as_object_array()
        return self


class ModelGateway(Protocol):
    async def complete(self, request: PreparedModelRequest) -> ModelResult: ...
