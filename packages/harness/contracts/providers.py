"""提供商配置与能力证据纯契约；不读取凭据，不访问环境或网络。"""

from datetime import datetime
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import AwareDatetime, Field, field_validator, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.models import ModelCapability, ResolvedModelRoute
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier, PositiveLimit

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
ModelTask = Literal["chat", "embedding", "rerank"]


def endpoint_identity(value: str) -> str:
    """只接受明确HTTPS根地址；提供商允许列表在解析时再次约束。"""
    try:
        parsed = urlsplit(value)
        valid = (
            value == value.strip()
            and parsed.scheme == "https"
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and "?" not in value
            and "#" not in value
            and parsed.port in {None, 443}
            and not any(character.isspace() for character in value)
            and "\\" not in value
            and "%" not in value
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("模型连接必须为无认证信息和查询参数的明确HTTPS地址")
    return value.rstrip("/")


class ProviderProtocol(Contract):
    task_type: ModelTask
    protocol_ref: Identifier
    adapter_refs: tuple[Identifier, ...] = Field(min_length=1, max_length=16)

    @field_validator("adapter_refs")
    @classmethod
    def unique_adapters(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("协议实现不能重复")
        return tuple(sorted(value))


class ProviderDefinition(Contract):
    provider_id: Identifier
    version: Identifier
    display_name: str = Field(min_length=1, max_length=128)
    endpoint_policy_ref: Identifier
    allowed_base_urls: tuple[str, ...] = Field(min_length=1, max_length=16)
    compatibility_ref: Identifier
    protocols: tuple[ProviderProtocol, ...] = Field(min_length=1, max_length=16)

    @field_validator("allowed_base_urls")
    @classmethod
    def validated_endpoints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(endpoint_identity(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("提供商地址不能重复")
        return tuple(sorted(normalized))

    @field_validator("protocols")
    @classmethod
    def unique_protocols(cls, values: tuple[ProviderProtocol, ...]) -> tuple[ProviderProtocol, ...]:
        keys = [(value.task_type, value.protocol_ref) for value in values]
        if len(set(keys)) != len(keys):
            raise ValueError("提供商任务与协议组合不能重复")
        return tuple(sorted(values, key=lambda value: (value.task_type, value.protocol_ref)))

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ModelDescriptor(Contract):
    provider_id: Identifier
    model_id: Identifier
    task_type: ModelTask
    declared_capabilities: tuple[ModelCapability, ...] = ()
    context_window_tokens: PositiveLimit | None = None
    max_output_tokens: PositiveLimit | None = None
    price_version_ref: Identifier | None = None

    @field_validator("declared_capabilities")
    @classmethod
    def unique_capabilities(cls, value: tuple[ModelCapability, ...]) -> tuple[ModelCapability, ...]:
        if len(set(value)) != len(value):
            raise ValueError("模型能力声明不能重复")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        if self.task_type != "chat" and (
            self.declared_capabilities or self.max_output_tokens is not None
        ):
            raise ValueError("非聊天模型不能声明聊天能力和输出Token上限")
        return self


class ModelCatalog(Contract):
    version: Identifier
    provider_id: Identifier
    models: tuple[ModelDescriptor, ...] = Field(min_length=1, max_length=256)

    @field_validator("models")
    @classmethod
    def sorted_models(cls, values: tuple[ModelDescriptor, ...]) -> tuple[ModelDescriptor, ...]:
        return tuple(sorted(values, key=lambda value: (value.task_type, value.model_id)))

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        keys = [(model.task_type, model.model_id) for model in self.models]
        if len(set(keys)) != len(keys):
            raise ValueError("模型目录不能重复登记同任务/模型")
        if any(model.provider_id != self.provider_id for model in self.models):
            raise ValueError("模型不能登记在其他提供商目录下")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ProviderConnection(Contract):
    ref: Identifier
    provider_id: Identifier
    protocol_ref: Identifier
    base_url: str = Field(min_length=1, max_length=512)
    credential_profile_ref: Identifier
    timeout_seconds: float = Field(default=30, gt=0, le=120, allow_inf_nan=False)
    transport_retries: Literal[0] = 0
    follow_redirects: Literal[False] = False

    @field_validator("base_url")
    @classmethod
    def normalize_endpoint(cls, value: str) -> str:
        return endpoint_identity(value)

    def content_hash(self) -> str:
        return canonical_hash(
            {"identity_version": "provider-connection@1", **self.model_dump(mode="json")}
        )


class ModelProfile(Contract):
    ref: Identifier
    connection_ref: Identifier
    model_id: Identifier
    task_type: ModelTask = "chat"
    required_capabilities: tuple[ModelCapability, ...] = ()
    capability_evidence_ref: Identifier | None = None
    max_output_tokens: PositiveLimit

    @field_validator("required_capabilities")
    @classmethod
    def unique_capabilities(cls, value: tuple[ModelCapability, ...]) -> tuple[ModelCapability, ...]:
        if len(set(value)) != len(value):
            raise ValueError("所需模型能力不能重复")
        return tuple(sorted(value))


class ModelProviderConfiguration(Contract):
    version: Literal["model-providers@1"] = "model-providers@1"
    connections: tuple[ProviderConnection, ...] = Field(min_length=1, max_length=64)
    profiles: tuple[ModelProfile, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def unique_refs(self) -> Self:
        for values in (self.connections, self.profiles):
            refs = [value.ref for value in values]
            if len(set(refs)) != len(refs):
                raise ValueError("连接或模型配置引用不能重复")
        return self


class InstalledModelProtocol(Contract):
    """由服务端注入的已安装实现身份，不代表实现或能力已经实测。"""

    protocol_ref: Identifier
    adapter_ref: Identifier
    task_type: ModelTask
    counter_ref: Identifier
    envelope_ref: Identifier


class ModelCapabilityEvidence(Contract):
    """必须由可信部署方提供；解析器只校验绑定，不认证报告真实性。"""

    ref: Identifier
    provider_id: Identifier
    model_id: Identifier
    task_type: ModelTask
    definition_hash: Digest
    catalog_hash: Digest
    connection_hash: Digest
    implementation: InstalledModelProtocol
    environment_profile_hash: Digest
    verified_capabilities: tuple[ModelCapability, ...] = ()
    context_window_tokens: PositiveLimit
    max_output_tokens: PositiveLimit
    report_hash: Digest
    verified_at: AwareDatetime
    expires_at: AwareDatetime

    @field_validator("verified_capabilities")
    @classmethod
    def unique_capabilities(cls, value: tuple[ModelCapability, ...]) -> tuple[ModelCapability, ...]:
        if len(set(value)) != len(value):
            raise ValueError("能力证据不能重复")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.expires_at <= self.verified_at:
            raise ValueError("能力证据有效期必须晚于验证时间")
        if self.implementation.task_type != self.task_type:
            raise ValueError("能力证据的任务与实现不匹配")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("能力证据必须保留输入窗口")
        return self

    def valid_at(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("能力证据校验时间必须带时区")
        return self.verified_at <= now < self.expires_at


class ResolvedModelSelection(Contract):
    """路由绑定独立深快照，未来由组合根写入新发布，不改旧路由hash。"""

    route: ResolvedModelRoute
    snapshot: JsonSnapshot

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.route.configuration_hash != self.snapshot.content_hash:
            raise ValueError("解析路由与配置快照不匹配")
        data = self.snapshot.as_object()
        if data.get("identity_version") != "resolved-provider-chat@1":
            raise ValueError("解析快照版本未知")
        connection = ProviderConnection.model_validate(data.get("connection"))
        profile = ModelProfile.model_validate(data.get("profile"))
        implementation = InstalledModelProtocol.model_validate(data.get("implementation"))
        evidence = ModelCapabilityEvidence.model_validate(data.get("evidence"))
        if (
            self.route.provider != connection.provider_id
            or self.route.model != profile.model_id
            or self.route.adapter_ref != implementation.adapter_ref
            or self.route.capabilities != evidence.verified_capabilities
            or not self.route.capabilities_verified
            or self.route.context_window_tokens != evidence.context_window_tokens
            or self.route.max_output_tokens != profile.max_output_tokens
        ):
            raise ValueError("路由字段与解析快照不一致")
        return self
