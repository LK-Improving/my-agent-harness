"""M3-07共用合成夹具；显式控制证据与传输，不连接真实供应商。"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from harness.contracts.hashing import canonical_hash
from harness.contracts.providers import ProviderConnection
from harness.contracts.retrieval_models import (
    EmbeddingResult, RetrievalCapabilityEvidence, RetrievalLimits, RetrievalModelConfiguration,
    RetrievalModelProfile, RetrievalReservation, RetrievalScope,
)
from harness.contracts.runtime import RuntimeFault
from harness.providers.retrieval import (
    EMBEDDING_PROTOCOL, EMBEDDING_URL, RERANK_PROTOCOL, RERANK_URL,
    installed_retrieval_protocol, resolve_retrieval_profile, retrieval_registry,
)

ENVIRONMENT = "a" * 64


def retrieval_values():
    registry = retrieval_registry()
    connections = tuple(ProviderConnection(
        ref=task + "-connection@1", provider_id="dashscope",
        protocol_ref=EMBEDDING_PROTOCOL if task == "embedding" else RERANK_PROTOCOL,
        base_url=EMBEDDING_URL if task == "embedding" else RERANK_URL,
        credential_profile_ref="dashscope-env@1",
    ) for task in ("embedding", "rerank"))
    profiles = tuple(RetrievalModelProfile(
        ref=task + "-profile@1", connection_ref=connection.ref, task_type=task,
        model_id="text-embedding-v4" if task == "embedding" else "gte-rerank-v2",
        evidence_ref=task + "-controlled-proof@1", max_items=10 if task == "embedding" else 40,
        dimensions=2 if task == "embedding" else None,
    ) for task, connection in zip(("embedding", "rerank"), connections, strict=True))
    config = RetrievalModelConfiguration(
        connections=connections, profiles=profiles,
        embedding_profile_ref=profiles[0].ref, rerank_profile_ref=profiles[1].ref,
        limits=RetrievalLimits(max_calls=2),
    )
    now = datetime.now(UTC)
    proofs = tuple(RetrievalCapabilityEvidence(
        ref=profile.evidence_ref, provider_id="dashscope", model_id=profile.model_id,
        task_type=profile.task_type, definition_hash=registry.get("dashscope").content_hash(),
        catalog_hash=registry.catalog("dashscope").content_hash(),
        connection_hash=connection.content_hash(),
        implementation=installed_retrieval_protocol(profile.task_type),
        environment_profile_hash=ENVIRONMENT, report_hash=canonical_hash("CONTROLLED_TEST_ONLY"),
        verified_at=now - timedelta(minutes=1), expires_at=now + timedelta(hours=1),
        dimensions=profile.dimensions, max_items=profile.max_items,
        max_characters=profile.max_characters,
    ) for profile, connection in zip(profiles, connections, strict=True))
    bindings = tuple(resolve_retrieval_profile(
        registry, config, profile.ref, task=profile.task_type, evidence=proofs,
        revoked_refs=frozenset(), environment_profile_hash=ENVIRONMENT, now=now,
    ) for profile in profiles)
    scope = RetrievalScope(
        context_id=uuid4().hex, tenant_id="test-tenant", subject_id="test-user",
        application_id="test-app", owner_kind="ingestion_job", owner_id=uuid4().hex,
        knowledge_base_id="test-kb", manifest_hash="b" * 64,
        embedding_binding_hash=bindings[0].snapshot.content_hash,
        rerank_binding_hash=bindings[1].snapshot.content_hash,
        deadline=now + timedelta(minutes=5),
    )
    return config, proofs, bindings, scope


def write_deployment(root: Path, monkeypatch):
    config, proofs, _, _ = retrieval_values()
    (root / "models.json").write_text(config.model_dump_json(), "utf-8")
    (root / "proofs.json").write_text(json.dumps({
        "version": "retrieval-evidence@1", "evidence": [p.model_dump(mode="json") for p in proofs],
        "revoked_refs": [],
    }), "utf-8")
    # 不读取宿主环境中的真实Key；替换组合根的值加载器。
    values = {"HARNESS_RETRIEVAL_MODELS_FILE": "models.json",
              "HARNESS_RETRIEVAL_EVIDENCE_FILE": "proofs.json", "DASHSCOPE_API_KEY": "controlled-only"}
    monkeypatch.setattr("harness.bootstrap.retrieval_models.model_values", lambda root: dict(values))
    return config, proofs, values


class Authority:
    """仅测试：真实接入必须查询实际owner、证据版本及实时权限。"""

    def __init__(self):
        self.denied = False
        self.calls = 0

    async def authorize(self, scope, payload):
        self.calls += 1
        if self.denied:
            raise RuntimeFault("FORBIDDEN", 403)


class MemoryLedger:
    """纯测试替身，不提供正式装配默认值；真实事务另有PG集成定义。"""

    def __init__(self):
        self.entries = {}
        self.calls = 0
        self.failure = None

    async def reserve(self, request, limits):
        existing = self.entries.get(request.call_id)
        if existing:
            old, result = existing
            if old.fingerprint() != request.fingerprint():
                raise RuntimeFault("RETRIEVAL_CALL_ID_CONFLICT", 409)
            if result is None:
                raise RuntimeFault("RETRIEVAL_ATTEMPT_NOT_REPLAYABLE", 409)
            return RetrievalReservation(request=old, cached_result=result)
        if self.calls >= limits.max_calls:
            raise RuntimeFault("RETRIEVAL_BUDGET_EXCEEDED", 429)
        self.calls += 1
        self.entries[request.call_id] = (request, None)
        return RetrievalReservation(request=request)

    async def succeed(self, request, result):
        self.entries[request.call_id] = (request, result)

    async def fail(self, request, failure):
        self.failure = failure.as_object()


class EmbeddingTransport:
    def __init__(self):
        self.calls = 0
        self.cancel = False
        self.bad_binding = False

    async def embed(self, request):
        self.calls += 1
        if self.cancel:
            raise asyncio.CancelledError()
        return EmbeddingResult(
            call_id=request.call_id, attempt_id=request.attempt_id,
            request_fingerprint="f" * 64 if self.bad_binding else request.fingerprint(),
            vectors=tuple((0.5, 0.5) for _ in request.payload.texts), usage=None,
        )


class RerankTransport:
    async def rerank(self, request):
        raise AssertionError("本用例不应发起重排")
