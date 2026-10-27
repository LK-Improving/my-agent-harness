"""检索专用有限目录与纯解析；复用注册/连接规则，不改聊天目录身份。"""

from datetime import datetime

from harness.contracts.providers import (
    InstalledModelProtocol,
    ModelCatalog,
    ModelDescriptor,
    ProviderDefinition,
    ProviderProtocol,
)
from harness.contracts.retrieval_models import (
    RetrievalCapabilityEvidence,
    RetrievalModelBinding,
    RetrievalModelConfiguration,
    RetrievalTask,
)
from harness.contracts.snapshots import JsonSnapshot
from harness.providers.registry import ProviderRegistry, ProviderResolutionError

EMBEDDING_PROTOCOL = "openai-embeddings@1"
RERANK_PROTOCOL = "dashscope-text-rerank@1"
EMBEDDING_ADAPTER = "openai-embeddings-http@1"
RERANK_ADAPTER = "dashscope-rerank-http@1"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


def installed_retrieval_protocol(task: RetrievalTask) -> InstalledModelProtocol:
    return InstalledModelProtocol(
        task_type=task,
        protocol_ref=EMBEDDING_PROTOCOL if task == "embedding" else RERANK_PROTOCOL,
        adapter_ref=EMBEDDING_ADAPTER if task == "embedding" else RERANK_ADAPTER,
        counter_ref="canonical-json-utf8-bytes@1",
        envelope_ref="retrieval-json@1",
    )


def retrieval_registry() -> ProviderRegistry:
    tasks: tuple[RetrievalTask, ...] = ("embedding", "rerank")
    definition = ProviderDefinition(
        provider_id="dashscope", version="dashscope-retrieval-definition@1",
        display_name="通义检索模型", endpoint_policy_ref="dashscope-retrieval-official@1",
        allowed_base_urls=(EMBEDDING_URL, RERANK_URL), compatibility_ref="dashscope-retrieval@1",
        protocols=tuple(
            ProviderProtocol(task_type=task, protocol_ref=installed_retrieval_protocol(task).protocol_ref,
                             adapter_refs=(installed_retrieval_protocol(task).adapter_ref,))
            for task in tasks
        ),
    )
    catalog = ModelCatalog(
        version="dashscope-retrieval-catalog@1", provider_id="dashscope",
        models=(ModelDescriptor(provider_id="dashscope", model_id="text-embedding-v4", task_type="embedding"),
                ModelDescriptor(provider_id="dashscope", model_id="gte-rerank-v2", task_type="rerank")),
    )
    return ProviderRegistry((definition,), (catalog,))


def resolve_retrieval_profile(
    registry: ProviderRegistry, configuration: RetrievalModelConfiguration, profile_ref: str,
    *, task: RetrievalTask, evidence: tuple[RetrievalCapabilityEvidence, ...],
    revoked_refs: frozenset[str], environment_profile_hash: str, now: datetime,
) -> RetrievalModelBinding:
    """精确版本匹配；声明目录不等于实测，证据无效时不fallback。"""
    # 重验冻结边界，拒绝绕过校验的model_copy更新。
    configuration = RetrievalModelConfiguration.model_validate_json(configuration.model_dump_json())
    profiles = {item.ref: item for item in configuration.profiles}
    connections = {item.ref: item for item in configuration.connections}
    profile = profiles.get(profile_ref)
    if profile is None or profile.task_type != task:
        raise ProviderResolutionError("RETRIEVAL_PROFILE_TASK_MISMATCH")
    connection = connections.get(profile.connection_ref)
    if connection is None:
        raise ProviderResolutionError("RETRIEVAL_CONNECTION_MISSING")
    definition = registry.get(connection.provider_id)
    catalog = registry.catalog(connection.provider_id)
    registry.model(connection.provider_id, profile.model_id, task)
    implementation = installed_retrieval_protocol(task)
    expected_url = EMBEDDING_URL if task == "embedding" else RERANK_URL
    if (
        connection.base_url != expected_url or connection.base_url not in definition.allowed_base_urls
        or connection.protocol_ref != implementation.protocol_ref
        or not any(p.task_type == task and p.protocol_ref == implementation.protocol_ref
                   and implementation.adapter_ref in p.adapter_refs for p in definition.protocols)
    ):
        raise ProviderResolutionError("RETRIEVAL_PROTOCOL_OR_ENDPOINT_MISMATCH")
    proofs = [RetrievalCapabilityEvidence.model_validate_json(item.model_dump_json())
              for item in evidence if item.ref == profile.evidence_ref]
    if len(proofs) != 1:
        raise ProviderResolutionError("RETRIEVAL_EVIDENCE_MISSING_OR_DUPLICATE")
    proof = proofs[0]
    if (
        proof.ref in revoked_refs or not proof.valid_at(now)
        or proof.provider_id != connection.provider_id or proof.model_id != profile.model_id
        or proof.task_type != task or proof.implementation != implementation
        or proof.definition_hash != definition.content_hash()
        or proof.catalog_hash != catalog.content_hash()
        or proof.connection_hash != connection.content_hash()
        or proof.environment_profile_hash != environment_profile_hash
        or proof.dimensions != profile.dimensions
        or proof.max_items < profile.max_items or proof.max_characters < profile.max_characters
    ):
        raise ProviderResolutionError("RETRIEVAL_EVIDENCE_INVALID")
    snapshot = JsonSnapshot.capture({
        "identity_version": "resolved-retrieval-model@1", "definition": definition.model_dump(mode="json"),
        "catalog_hash": catalog.content_hash(), "connection": connection.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"), "evidence": proof.model_dump(mode="json"),
        "implementation": implementation.model_dump(mode="json"),
    })
    return RetrievalModelBinding(connection=connection, profile=profile,
                                 implementation=implementation, evidence=proof, snapshot=snapshot)
