"""检索可选组合根；显式启用后才读取本项目配置，不改core启动与聊天目录。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator

from harness.adapters.models.dashscope_retrieval import DashScopeRetrievalClient
from harness.bootstrap.providers import (
    ModelConfigurationError, _binding, _json_file, model_values,
)
from harness.contracts.retrieval_models import (
    RetrievalAuthority, RetrievalCapabilityEvidence, RetrievalLedger, RetrievalModelBinding,
    RetrievalModelConfiguration, RetrievalTask,
)
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier
from harness.providers.registry import ProviderResolutionError
from harness.providers.retrieval import resolve_retrieval_profile, retrieval_registry
from harness.runtime.retrieval_models import GovernedRetrievalGateway


class RetrievalEvidenceBundle(Contract):
    version: Literal["retrieval-evidence@1"] = "retrieval-evidence@1"
    evidence: tuple[RetrievalCapabilityEvidence, ...] = Field(default=(), max_length=128)
    revoked_refs: tuple[Identifier, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def unique_refs(self) -> Self:
        if (len({item.ref for item in self.evidence}) != len(self.evidence)
            or len(set(self.revoked_refs)) != len(self.revoked_refs)):
            raise ValueError("检索证据或撤销引用不能重复")
        return self


@dataclass(frozen=True, slots=True)
class RetrievalModelRuntime:
    gateway: GovernedRetrievalGateway = field(repr=False)
    embedding_binding: RetrievalModelBinding
    rerank_binding: RetrievalModelBinding
    snapshot: JsonSnapshot
    configuration_guard: Callable[[], None] = field(repr=False)

    def ready(self) -> bool:
        """仅检查配置/凭据，不联网；不能替代PG迁移或真实运行验收。"""
        try:
            self.configuration_guard()
            return True
        except (ValueError, OSError):
            return False


def load_retrieval_runtime(
    root: Path, *, enabled: bool, environment_profile_hash: str = "",
    ledger: RetrievalLedger | None = None, authority: RetrievalAuthority | None = None,
) -> RetrievalModelRuntime | None:
    """接入方必须提供实际owner/知识授权实现；没有默认放行或自动内存账本。"""
    if not enabled:
        return None
    if ledger is None or authority is None:
        raise ModelConfigurationError("RETRIEVAL_GOVERNANCE_REQUIRED")
    try:
        values = model_values(root)
        config_ref = values.get("HARNESS_RETRIEVAL_MODELS_FILE", "")
        evidence_ref = values.get("HARNESS_RETRIEVAL_EVIDENCE_FILE", "")
        if not config_ref or not evidence_ref:
            raise ModelConfigurationError("RETRIEVAL_CONFIGURATION_REQUIRED")
        configuration = RetrievalModelConfiguration.model_validate(_json_file(root, config_ref))
        original = JsonSnapshot.capture(configuration.model_dump(mode="json"))

        def select(task: RetrievalTask) -> RetrievalModelBinding:
            current = model_values(root)
            if (current.get("HARNESS_RETRIEVAL_MODELS_FILE") != config_ref
                or current.get("HARNESS_RETRIEVAL_EVIDENCE_FILE") != evidence_ref):
                raise ModelConfigurationError("RETRIEVAL_CONFIGURATION_CHANGED")
            active = RetrievalModelConfiguration.model_validate(_json_file(root, config_ref))
            if JsonSnapshot.capture(active.model_dump(mode="json")) != original:
                raise ModelConfigurationError("RETRIEVAL_CONFIGURATION_CHANGED")
            bundle = RetrievalEvidenceBundle.model_validate(_json_file(root, evidence_ref))
            ref = active.embedding_profile_ref if task == "embedding" else active.rerank_profile_ref
            selected = resolve_retrieval_profile(
                retrieval_registry(), active, ref, task=task, evidence=bundle.evidence,
                revoked_refs=frozenset(bundle.revoked_refs),
                environment_profile_hash=environment_profile_hash, now=datetime.now(UTC),
            )
            account = _binding(root, current, selected.connection)
            snapshot = selected.snapshot.as_object()
            snapshot["credential_binding"] = account.model_dump(mode="json")
            return RetrievalModelBinding(
                connection=selected.connection, profile=selected.profile,
                implementation=selected.implementation, evidence=selected.evidence,
                snapshot=JsonSnapshot.capture(snapshot),
            )

        embedding_binding, rerank_binding = select("embedding"), select("rerank")

        def credential(binding: RetrievalModelBinding) -> SecretStr:
            # 每次读取精确选中配置，允许同一账户引用更新Key，不允许静默换账户。
            actual = select(binding.profile.task_type)
            if actual.snapshot != binding.snapshot:
                raise ModelConfigurationError("RETRIEVAL_BINDING_CHANGED")
            current = model_values(root)
            account = _binding(root, current, binding.connection)
            value = current.get(account.env_var, "")
            if not value:
                raise ModelConfigurationError("RETRIEVAL_CREDENTIAL_MISSING")
            return SecretStr(value)

        def guard() -> None:
            try:
                credential(embedding_binding)
                credential(rerank_binding)
            except ModelConfigurationError:
                raise
            except ProviderResolutionError as exc:
                raise ModelConfigurationError(exc.code) from None
            except (ValueError, OSError, TypeError, KeyError):
                raise ModelConfigurationError("RETRIEVAL_CONFIGURATION_INVALID") from None

        snapshot = JsonSnapshot.capture({
            "identity_version": "retrieval-model-runtime@1",
            "embedding": embedding_binding.snapshot.materialize(),
            "rerank": rerank_binding.snapshot.materialize(),
            "limits": configuration.limits.model_dump(mode="json"),
            "environment_profile_hash": environment_profile_hash,
        })
        gateway = GovernedRetrievalGateway(
            ledger=ledger, authority=authority,
            embedding=DashScopeRetrievalClient(embedding_binding, lambda: credential(embedding_binding)),
            reranking=DashScopeRetrievalClient(rerank_binding, lambda: credential(rerank_binding)),
            embedding_binding=embedding_binding, rerank_binding=rerank_binding,
            limits=configuration.limits, configuration_guard=guard,
        )
        return RetrievalModelRuntime(gateway, embedding_binding, rerank_binding, snapshot, guard)
    except ModelConfigurationError:
        raise
    except ProviderResolutionError as exc:
        raise ModelConfigurationError(exc.code) from None
    except (ValueError, OSError, TypeError, KeyError):
        raise ModelConfigurationError("RETRIEVAL_CONFIGURATION_INVALID") from None
