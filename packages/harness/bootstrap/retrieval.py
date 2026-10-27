"""M3-03可选检索组合根；显式启用且给出实际执行存储/身份源才装配。"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, SecretStr

from harness.contracts.retrieval import RetrievalProfile, RetrievalRelease, SnapshotPin
from harness.contracts.runtime import Principal
from harness.contracts.specs import Contract

if TYPE_CHECKING:
    from harness.adapters.execution_postgres import PostgresExecutionStore
    from harness.retrieval.pipeline import RetrievalService
    from harness.adapters.retrieval_postgres import PostgresRetrievalRepository


class RetrievalPipelineConfiguration(Contract):
    profiles: tuple[RetrievalProfile, ...] = Field(min_length=1, max_length=8)
    snapshots: tuple[SnapshotPin, ...] = Field(min_length=1, max_length=4)


@dataclass(frozen=True, slots=True)
class RetrievalRuntime:
    service: "RetrievalService"
    release: RetrievalRelease
    repository: "PostgresRetrievalRepository"
    configuration_guard: Callable[[], None]


def load_retrieval(
    root: Path, *, enabled: bool, execution: "PostgresExecutionStore | None" = None,
    current_identity: Callable[[Principal], Principal] | None = None,
) -> RetrievalRuntime | None:
    if not enabled:
        return None
    from harness.adapters.execution_postgres import PostgresExecutionStore
    from harness.adapters.milvus_index import MilvusDualIndex
    from harness.adapters.milvus_recall import MilvusRecall
    from harness.adapters.retrieval_models_postgres import PostgresRetrievalLedger
    from harness.adapters.retrieval_postgres import PostgresRetrievalRepository
    from harness.bootstrap.indexing import IndexConfiguration
    from harness.bootstrap.providers import ModelConfigurationError, model_values
    from harness.bootstrap.retrieval_models import load_retrieval_runtime
    from harness.retrieval.pipeline import RetrievalService
    from harness.runtime.retrieval_models import GovernedRetrievalGateway

    if not isinstance(execution, PostgresExecutionStore) or current_identity is None:
        raise ModelConfigurationError("RETRIEVAL_EXECUTION_GOVERNANCE_REQUIRED")
    values = model_values(root)
    if values.get("HARNESS_RAG_ENABLED", "false").lower() != "true":
        raise ModelConfigurationError("RAG_DISABLED")
    dsn = values.get("HARNESS_DATABASE_DSN", "")
    index_ref = values.get("HARNESS_INDEX_CONFIG_FILE", "")
    pipeline_ref = values.get("HARNESS_RETRIEVAL_PIPELINE_FILE", "")
    if not dsn or not index_ref or not pipeline_ref:
        raise ModelConfigurationError("RETRIEVAL_PIPELINE_CONFIGURATION_REQUIRED")
    try:
        if (root / index_ref).stat().st_size > 65536 or (root / pipeline_ref).stat().st_size > 65536:
            raise ValueError("配置超过上限")
        index = IndexConfiguration.model_validate_json((root / index_ref).read_text("utf-8"))
        pipeline = RetrievalPipelineConfiguration.model_validate_json((root / pipeline_ref).read_text("utf-8"))
        repository = PostgresRetrievalRepository(dsn, execution, current_identity)
        ledger = PostgresRetrievalLedger(dsn)
        models = load_retrieval_runtime(root, enabled=True, environment_profile_hash=index.environment_profile_hash,
                                        ledger=ledger, authority=repository)
        if models is None:
            raise ModelConfigurationError("RETRIEVAL_MODELS_REQUIRED")
        release = RetrievalRelease(models=models.snapshot, backend_hash=index.backend_hash(),
                                    profiles=pipeline.profiles, snapshots=pipeline.snapshots)
        # 此gateway专属正式查询，不修改M3-07运行对象或索引构建所用账本上限。
        original = models.gateway
        gateway = GovernedRetrievalGateway(
            ledger=ledger, authority=repository, embedding=original.embedding, reranking=original.reranking,
            embedding_binding=models.embedding_binding, rerank_binding=models.rerank_binding,
            limits=release.profiles[0].model_limits, configuration_guard=models.configuration_guard,
        )
        backend = MilvusDualIndex(index.milvus_uri, SecretStr(values.get(index.token_env_var, "")),
                                  index.database, index.backend_hash(), rpc_timeout_seconds=index.rpc_timeout_seconds)
        service = RetrievalService(repository, MilvusRecall(backend), gateway, release)

        def guard() -> None:
            current = model_values(root)
            if (current.get("HARNESS_RAG_ENABLED", "false").lower() != "true"
                or current.get("HARNESS_INDEX_CONFIG_FILE") != index_ref
                or current.get("HARNESS_RETRIEVAL_PIPELINE_FILE") != pipeline_ref
                or current.get("HARNESS_DATABASE_DSN") != dsn
                or IndexConfiguration.model_validate_json((root / index_ref).read_text("utf-8")) != index
                or RetrievalPipelineConfiguration.model_validate_json((root / pipeline_ref).read_text("utf-8")) != pipeline):
                raise ModelConfigurationError("RETRIEVAL_DEPLOYMENT_CHANGED")
            models.configuration_guard()

        return RetrievalRuntime(service, release, repository, guard)
    except ModelConfigurationError:
        raise
    except (OSError, ValueError, ImportError):
        raise ModelConfigurationError("RETRIEVAL_PIPELINE_CONFIGURATION_INVALID") from None
