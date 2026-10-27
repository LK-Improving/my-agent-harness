"""固定快照检索、授权证据及受治理端口的纯契约。"""

from typing import Literal, Protocol, Self

from pydantic import Field, FiniteFloat, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import IndexBinding
from harness.contracts.providers import Digest
from harness.contracts.retrieval_models import (
    EmbeddingRequest, EmbeddingResult, RerankRequest, RerankResult, RetrievalLimits, RetrievalScope, Vector,
)
from harness.contracts.runtime import Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier


class RetrievalProfile(Contract):
    ref: Identifier = "hybrid-bounded@1"
    recall_limit: int = Field(default=12, ge=1, le=40, strict=True)
    fusion_limit: int = Field(default=12, ge=1, le=40, strict=True)
    top_n: int = Field(default=5, ge=1, le=10, strict=True)
    rrf_k: int = Field(default=60, ge=1, le=1000, strict=True)
    query_characters: int = Field(default=1000, ge=1, le=8000, strict=True)
    rerank_characters: int = Field(default=4000, ge=1, le=16000, strict=True)
    evidence_characters: int = Field(default=4000, ge=1, le=12000, strict=True)
    timeout_seconds: int = Field(default=60, ge=1, le=120, strict=True)
    branch_timeout_seconds: int = Field(default=15, ge=1, le=30, strict=True)
    model_limits: RetrievalLimits = RetrievalLimits(max_calls=2, max_input_bytes=16384)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if (self.top_n > self.fusion_limit or self.fusion_limit > self.recall_limit * 2
            or self.branch_timeout_seconds >= self.timeout_seconds or self.model_limits.max_calls != 2):
            raise ValueError("检索阶段上限不一致；首版固定一次Embedding和至多一次Rerank")
        return self

    def parent_token_reservation(self) -> int:
        """UTF-8请求字节上界加每次封装余量，保守预留，不代表实际tokens或费用。"""
        return self.model_limits.max_input_bytes + 512 * self.model_limits.max_calls


class SnapshotPin(Contract):
    knowledge_base_id: Identifier
    snapshot_id: Identifier
    snapshot_hash: Digest
    generation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    index_binding_hash: Digest


class RetrievalRelease(Contract):
    version: Literal["retrieval-release@1"] = "retrieval-release@1"
    models: JsonSnapshot
    backend_hash: Digest
    profiles: tuple[RetrievalProfile, ...] = Field(min_length=1, max_length=8)
    snapshots: tuple[SnapshotPin, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique(self) -> Self:
        if (len({p.ref for p in self.profiles}) != len(self.profiles)
            or len({p.knowledge_base_id for p in self.snapshots}) != len(self.snapshots)):
            raise ValueError("检索profile或KB pin不能重复")
        if any(p.model_limits != self.profiles[0].model_limits for p in self.profiles):
            raise ValueError("同一检索发布使用统一子模型额度，不允许运行中切换账本限制")
        return self


class RetrievalQuery(Contract):
    request_id: Identifier
    query: str = Field(min_length=1, max_length=8000, repr=False)


class RetrievalTicket(Contract):
    retrieval_id: Identifier
    actor: Principal
    owner: Identifier
    scope: RetrievalScope
    pin: SnapshotPin
    index_binding: IndexBinding
    profile: RetrievalProfile
    query: RetrievalQuery = Field(repr=False)

    @model_validator(mode="after")
    def bound(self) -> Self:
        if (self.scope.owner_kind != "run" or self.scope.knowledge_base_id != self.pin.knowledge_base_id
            or self.scope.embedding_binding_hash != self.index_binding.embedding_binding_hash
            or self.scope.rerank_binding_hash != self.index_binding.rerank_binding_hash
            or self.index_binding.content_hash() != self.pin.index_binding_hash
            or (self.actor.tenant_id, self.actor.subject_id, self.actor.application_id) != (
                self.scope.tenant_id, self.scope.subject_id, self.scope.application_id)):
            raise ValueError("检索ticket范围/模型/代际绑定不一致")
        return self


class RecallHit(Contract):
    chunk_id: Digest
    score: FiniteFloat


class Evidence(Contract):
    chunk_id: Digest
    version_id: Identifier
    document_id: Identifier
    snapshot_id: Identifier
    generation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    body_hash: Digest
    body: str = Field(min_length=1, max_length=8000, repr=False)

    @model_validator(mode="after")
    def valid_body(self) -> Self:
        if canonical_hash(self.body) != self.body_hash:
            raise ValueError("授权证据正文hash不一致")
        return self


class RetrievalResult(Contract):
    retrieval_id: Identifier
    outcome: Literal["evidence", "no_evidence"]
    evidence: tuple[Evidence, ...] = Field(max_length=10)
    degraded: tuple[Literal["dense_unavailable", "sparse_unavailable", "rerank_unavailable"], ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.outcome == "evidence") != bool(self.evidence):
            raise ValueError("无证据不能标记为有证据")
        if len({e.chunk_id for e in self.evidence}) != len(self.evidence):
            raise ValueError("证据不能重复")
        return self


class RetrievalReservation(Contract):
    ticket: RetrievalTicket
    state: Literal["queued", "running", "succeeded", "failed"]
    result: RetrievalResult | None = None


class RecallUnavailable(Exception):
    """仅代表明确可降级的依赖可用性错误，不承载权限/取消/绑定失败。"""


class RecallPort(Protocol):
    async def dense(self, ticket: RetrievalTicket, vector: Vector) -> tuple[RecallHit, ...]: ...
    async def sparse(self, ticket: RetrievalTicket) -> tuple[RecallHit, ...]: ...


class RetrievalModels(Protocol):
    async def embed(self, scope: RetrievalScope, call_id: str, request: EmbeddingRequest) -> EmbeddingResult: ...
    async def rerank(self, scope: RetrievalScope, call_id: str, request: RerankRequest) -> RerankResult: ...


class RetrievalRepository(Protocol):
    async def open(self, actor: Principal, owner: str, run_id: str, kb_id: str,
                   query: RetrievalQuery, configuration: RetrievalRelease) -> RetrievalReservation: ...
    async def claim(self, ticket: RetrievalTicket) -> None: ...
    async def check(self, ticket: RetrievalTicket) -> None: ...
    async def hydrate(self, ticket: RetrievalTicket, ids: tuple[str, ...]) -> tuple[Evidence, ...]: ...
    async def stage(self, ticket: RetrievalTicket, name: str, data: JsonSnapshot) -> None: ...
    async def finish(self, ticket: RetrievalTicket, result: RetrievalResult) -> None: ...
    async def deliver(self, ticket: RetrievalTicket, result: RetrievalResult) -> RetrievalResult: ...
    async def fail(self, ticket: RetrievalTicket, code: str) -> None: ...
    async def authorize_result(self, actor: Principal, owner: str, run_id: str,
                               result: RetrievalResult) -> RetrievalResult: ...


class KnowledgeRetrieval(Protocol):
    async def retrieve(self, actor: Principal, owner: str, run_id: str, kb_id: str,
                       query: RetrievalQuery) -> RetrievalResult: ...
    async def authorize_result(self, actor: Principal, owner: str, run_id: str,
                               result: RetrievalResult) -> RetrievalResult: ...
