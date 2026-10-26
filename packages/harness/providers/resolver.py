"""解析可信部署配置为固定聊天路由；不读取环境、密钥或供应商目录。"""

from datetime import datetime

from harness.contracts.models import ResolvedModelRoute
from harness.contracts.providers import (
    InstalledModelProtocol,
    ModelCapabilityEvidence,
    ModelProviderConfiguration,
    ResolvedModelSelection,
)
from harness.contracts.release import BudgetProfile
from harness.contracts.snapshots import JsonSnapshot
from harness.providers.registry import ProviderRegistry, ProviderResolutionError


def resolve_model_profile(
    registry: ProviderRegistry,
    configuration: ModelProviderConfiguration,
    profile_ref: str,
    *,
    installed: tuple[InstalledModelProtocol, ...],
    evidence: tuple[ModelCapabilityEvidence, ...],
    budget: BudgetProfile,
    environment_profile_hash: str,
    now: datetime,
    safety_margin_tokens: int,
    revoked_evidence_refs: frozenset[str] = frozenset(),
) -> ResolvedModelSelection:
    """只解析选中profile；证据/实现必须来自服务端，不能接收API自报能力。"""
    configuration = ModelProviderConfiguration.model_validate_json(configuration.model_dump_json())
    budget = BudgetProfile.model_validate_json(budget.model_dump_json())
    if type(safety_margin_tokens) is not int or not 0 <= safety_margin_tokens <= 100000:
        raise ProviderResolutionError("MODEL_SAFETY_MARGIN_INVALID")
    profiles = {profile.ref: profile for profile in configuration.profiles}
    connections = {connection.ref: connection for connection in configuration.connections}
    if profile_ref not in profiles:
        raise ProviderResolutionError("MODEL_PROFILE_NOT_REGISTERED")
    profile = profiles[profile_ref]
    if profile.task_type != "chat":
        raise ProviderResolutionError("MODEL_TASK_NOT_SUPPORTED")
    if profile.connection_ref not in connections:
        raise ProviderResolutionError("PROVIDER_CONNECTION_NOT_REGISTERED")
    connection = connections[profile.connection_ref]
    definition = registry.get(connection.provider_id)
    if connection.base_url not in definition.allowed_base_urls:
        raise ProviderResolutionError("PROVIDER_ENDPOINT_NOT_ALLOWED")
    model = registry.model(connection.provider_id, profile.model_id, profile.task_type)
    binding = next(
        (
            binding
            for binding in definition.protocols
            if binding.task_type == profile.task_type
            and binding.protocol_ref == connection.protocol_ref
        ),
        None,
    )
    if binding is None:
        raise ProviderResolutionError("PROTOCOL_NOT_SUPPORTED")
    installed = tuple(
        InstalledModelProtocol.model_validate_json(item.model_dump_json()) for item in installed
    )
    keys = [(item.task_type, item.protocol_ref, item.adapter_ref) for item in installed]
    if len(set(keys)) != len(keys):
        raise ProviderResolutionError("PROTOCOL_IMPLEMENTATION_DUPLICATE")
    evidence = tuple(
        ModelCapabilityEvidence.model_validate_json(item.model_dump_json()) for item in evidence
    )
    by_ref = {item.ref: item for item in evidence}
    if len(by_ref) != len(evidence):
        raise ProviderResolutionError("MODEL_EVIDENCE_DUPLICATE")
    proof = by_ref.get(profile.capability_evidence_ref or "")
    if proof is None or proof.ref in revoked_evidence_refs:
        raise ProviderResolutionError("MODEL_CAPABILITY_UNVERIFIED")
    if not proof.valid_at(now):
        raise ProviderResolutionError("MODEL_EVIDENCE_EXPIRED_OR_FUTURE")
    if (
        proof.provider_id != connection.provider_id
        or proof.model_id != model.model_id
        or proof.task_type != profile.task_type
        or proof.definition_hash != definition.content_hash()
        or proof.catalog_hash != registry.catalog(connection.provider_id).content_hash()
        or proof.connection_hash != connection.content_hash()
        or proof.environment_profile_hash != environment_profile_hash
        or proof.implementation.protocol_ref != connection.protocol_ref
        or proof.implementation.adapter_ref not in binding.adapter_refs
    ):
        raise ProviderResolutionError("MODEL_EVIDENCE_BINDING_MISMATCH")
    if proof.implementation not in installed:
        raise ProviderResolutionError("PROTOCOL_NOT_INSTALLED")
    # 本阶段仅声明已有Completions计数边界；其他封装须后续独立实现/验证。
    if (
        proof.implementation.counter_ref != "utf8-envelope@1"
        or proof.implementation.envelope_ref != "openai-chat-buffered@1"
        or proof.implementation.protocol_ref != "openai-chat-completions@1"
    ):
        raise ProviderResolutionError("MODEL_ENVELOPE_NOT_SUPPORTED")
    verified = set(proof.verified_capabilities)
    if not verified.issubset(model.declared_capabilities) or not set(
        profile.required_capabilities
    ).issubset(verified):
        raise ProviderResolutionError("MODEL_CAPABILITY_UNVERIFIED")
    if (
        profile.max_output_tokens != budget.max_output_tokens
        or profile.max_output_tokens > proof.max_output_tokens
        or (
            model.max_output_tokens is not None
            and proof.max_output_tokens > model.max_output_tokens
        )
        or (
            model.context_window_tokens is not None
            and proof.context_window_tokens > model.context_window_tokens
        )
    ):
        raise ProviderResolutionError("MODEL_LIMITS_MISMATCH")
    # 安全余量作为固定解析输入，不从模型声明推断；与后续执行profile保持一致。
    snapshot = JsonSnapshot.capture(
        {
            "identity_version": "resolved-provider-chat@1",
            "definition": definition.model_dump(mode="json"),
            "catalog": registry.catalog(connection.provider_id).model_dump(mode="json"),
            "connection": connection.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "evidence": proof.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
            "safety_margin_tokens": safety_margin_tokens,
            "implementation": proof.implementation.model_dump(mode="json"),
        }
    )
    if (
        budget.max_input_tokens + budget.max_output_tokens + safety_margin_tokens
        > proof.context_window_tokens
    ):
        raise ProviderResolutionError("MODEL_CONTEXT_BUDGET_EXCEEDED")
    return ResolvedModelSelection(
        route=ResolvedModelRoute(
            provider=connection.provider_id,
            model=model.model_id,
            adapter_ref=proof.implementation.adapter_ref,
            configuration_hash=snapshot.content_hash,
            capabilities=proof.verified_capabilities,
            capabilities_verified=True,
            context_window_tokens=proof.context_window_tokens,
            max_output_tokens=profile.max_output_tokens,
            price_version_ref=model.price_version_ref,
        ),
        snapshot=snapshot,
    )
