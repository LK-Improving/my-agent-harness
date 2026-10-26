"""固定扩展契约；无放行/重试/权限或回执覆盖能力。"""

from collections.abc import Awaitable, Callable
from typing import Literal, Self

from pydantic import Field, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import ClaimedRun
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ExecutionContext, Identifier

ExtensionStage = Literal[
    "before_run",
    "before_model",
    "wrap_model_call",
    "after_model",
    "before_tool",
    "wrap_tool_call",
    "after_tool",
    "validate_completion",
    "before_commit",
]
STAGE_ORDER: tuple[ExtensionStage, ...] = (
    "before_run",
    "before_model",
    "wrap_model_call",
    "after_model",
    "before_tool",
    "wrap_tool_call",
    "after_tool",
    "validate_completion",
    "before_commit",
)


class ExtensionSpec(Contract):
    extension_id: Identifier
    version: Identifier
    handler_ref: Identifier
    stage: ExtensionStage
    order: int = Field(ge=0, le=1000, strict=True)
    mode: Literal["policy", "observer"] = "policy"
    timeout_ms: int = Field(default=500, ge=1, le=1000, strict=True)
    configuration: JsonSnapshot = Field(
        default_factory=lambda: JsonSnapshot.capture({}), repr=False
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        self.configuration.as_object()
        if len(self.configuration.canonical_text.encode("utf-8")) > 16384:
            raise ValueError("扩展非秘密配置不能超过16KiB")
        if self.mode == "observer" and self.stage not in {"after_model", "after_tool"}:
            raise ValueError("观察扩展只允许调用后阶段")
        return self


class ExtensionInvocation(Contract):
    context: ExecutionContext
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    stage: ExtensionStage
    payload: JsonSnapshot = Field(repr=False)
    configuration: JsonSnapshot = Field(repr=False)

    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ExtensionDecision(Contract):
    extension_id: Identifier
    version: Identifier
    invocation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: Literal["abstain", "deny"] = "abstain"
    reason_code: Identifier | None = None
    context_prefix: JsonSnapshot | None = Field(default=None, repr=False)
    tool_arguments: JsonSnapshot | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.verdict == "deny" and (
            not self.reason_code
            or self.context_prefix is not None
            or self.tool_arguments is not None
        ):
            raise ValueError("拒绝必须提供原因码且不得同时改参")
        if self.context_prefix is not None and self.tool_arguments is not None:
            raise ValueError("一次决定不能同时改变模型和工具输入")
        if self.tool_arguments is not None:
            self.tool_arguments.as_object()
        if self.context_prefix is not None:
            self.context_prefix.as_object_array()
        return self


DecisionRecorder = Callable[
    [ClaimedRun, ExtensionSpec, ExtensionInvocation, ExtensionDecision], Awaitable[None]
]
