"""发布身份与候选评测绑定，不提供实际激活或认证实现。"""

from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from harness.contracts.hashing import canonical_hash
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec, Contract, Identifier, PositiveLimit


class BudgetProfile(Contract):
    version: Identifier
    mode: Literal["bounded_estimate"] = "bounded_estimate"
    max_input_tokens: PositiveLimit
    max_output_tokens: PositiveLimit
    max_total_tokens: PositiveLimit
    execution_budget_ms: PositiveLimit
    allow_unknown_billable_tools: Literal[False] = False

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.max_input_tokens + self.max_output_tokens > self.max_total_tokens:
            raise ValueError("单步输入输出预留不能超过整轮 Token 上限")
        return self


class CapabilityProfile(Contract):
    environment: Literal["development", "test", "production"]
    auth_mode: Literal["development", "oidc"]
    enabled_capabilities: frozenset[Literal["core", "rag"]] = frozenset({"core"})
    execution_worker_replicas: Literal[1] = 1
    max_active_runs_per_worker: Literal[1] = 1
    answer_stream_retry_policy: Literal["no_retry_after_publish"] = "no_retry_after_publish"

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.auth_mode == "development" and self.environment != "development":
            raise ValueError("开发身份只能用于显式开发环境")
        if "core" not in self.enabled_capabilities:
            raise ValueError("必须启用 core")
        return self


class ReleaseManifest(Contract):
    schema_version: Literal[1] = 1
    spec: AgentSpec
    source_revision: str | None
    source_tree_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    dependency_lock_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool_schema_hashes: JsonSnapshot
    model_profile: JsonSnapshot
    profiles: JsonSnapshot
    environment_profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator(
        "tool_schema_hashes",
        "model_profile",
        "profiles",
        mode="before",
        json_schema_input_type=dict[str, JsonValue],
    )
    @classmethod
    def capture_mapping(cls, value: object) -> JsonSnapshot:
        if isinstance(value, JsonSnapshot):
            value.as_object()
            return value
        if not isinstance(value, dict):
            raise ValueError("发布映射必须为 JSON 对象")
        return JsonSnapshot.capture(value)

    @field_serializer("tool_schema_hashes", "model_profile", "profiles")
    def serialize_mapping(self, value: JsonSnapshot) -> dict[str, JsonValue]:
        # 线协议保持原对象形状，但每次导出均返回独立副本。
        return value.as_object()

    @model_validator(mode="after")
    def validate_tool_identity(self) -> Self:
        expected = {f"{tool.name}@{tool.version}" for tool in self.spec.allowed_tools}
        hashes = self.tool_schema_hashes.as_object()
        if expected != hashes.keys():
            raise ValueError("manifest 工具目录与 Agent 白名单不一致")
        for value in hashes.values():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("工具 Schema hash 格式错误")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def snapshot(self) -> JsonSnapshot:
        """提供可持久保存的发布内容快照，不把内部可变视图交给调用方。"""
        return JsonSnapshot.capture(self.model_dump(mode="json"))


class EvaluationContext(Contract):
    context_id: Identifier
    tenant_id: Identifier
    subject_id: Identifier
    agent_id: Identifier
    candidate_release_id: Identifier
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    gate_policy_version: Identifier
    environment_profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    resource_scopes: frozenset[str]
    expires_at: AwareDatetime
    revoked_at: AwareDatetime | None = None

    def require_access(self, tenant: str, subject: str, now: AwareDatetime) -> None:
        if tenant != self.tenant_id or subject != self.subject_id:
            raise PermissionError("评测上下文主体范围不匹配")
        if now.tzinfo is None:
            raise ValueError("评测检查时间必须包含时区")
        if self.revoked_at is not None or now >= self.expires_at:
            raise PermissionError("评测上下文已过期或撤销")
