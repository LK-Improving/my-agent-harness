"""检索模型独立契约；不包含聊天消息、不读取环境或持久化实现。"""

from datetime import datetime
from typing import Annotated, Literal, Protocol, Self

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.providers import Digest, InstalledModelProtocol, ProviderConnection
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier

RetrievalTask = Literal["embedding", "rerank"]
Text = Annotated[str, Field(min_length=1, max_length=8000)]


class RetrievalUsage(Contract):
    """未返回的计量为未知，不根据另一个字段推导或当作零费用。"""

    total_tokens: int | None = Field(default=None, ge=0, strict=True)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)


class EmbeddingRequest(Contract):
    task_type: Literal["embedding"] = "embedding"
    texts: tuple[Text, ...] = Field(min_length=1, max_length=10)


class RerankEvidence(Contract):
    """权威回表后的证据身份；Authority须校验版本、hash与实时访问权。"""

    chunk_id: Identifier
    version_id: Identifier
    body_hash: Digest


class RerankRequest(Contract):
    task_type: Literal["rerank"] = "rerank"
    query: Text
    documents: tuple[Text, ...] = Field(min_length=1, max_length=40)
    evidence: tuple[RerankEvidence, ...] = Field(min_length=1, max_length=40)
    top_n: int = Field(default=5, ge=1, le=40, strict=True)

    @model_validator(mode="after")
    def bounded_top_n(self) -> Self:
        if self.top_n > len(self.documents):
            raise ValueError("重排TopN不能超过文档数")
        if len(self.evidence) != len(self.documents):
            raise ValueError("重排正文与证据数量不一致")
        identities = [(item.chunk_id, item.version_id) for item in self.evidence]
        if len(set(identities)) != len(identities):
            raise ValueError("重排证据不能重复")
        if any(item.body_hash != canonical_hash(body)
               for item, body in zip(self.evidence, self.documents, strict=True)):
            raise ValueError("重排正文与证据hash不一致")
        return self


RetrievalPayload = Annotated[EmbeddingRequest | RerankRequest, Field(discriminator="task_type")]


class RetrievalModelProfile(Contract):
    ref: Identifier
    connection_ref: Identifier
    task_type: RetrievalTask
    model_id: Identifier
    evidence_ref: Identifier
    dimensions: int | None = Field(default=None, ge=1, le=2048, strict=True)
    max_items: int = Field(ge=1, le=40, strict=True)
    max_characters: int = Field(default=8000, ge=1, le=8000, strict=True)
    max_input_bytes: int = Field(default=64000, ge=1, le=1048576, strict=True)

    @model_validator(mode="after")
    def task_limits(self) -> Self:
        if self.task_type == "embedding" and (self.dimensions is None or self.max_items > 10):
            raise ValueError("Embedding必须固定维度且每批不超过10条")
        if self.task_type == "rerank" and self.dimensions is not None:
            raise ValueError("Rerank不能配置向量维度")
        return self


class RetrievalLimits(Contract):
    ref: Identifier = "retrieval-bounded@1"
    max_calls: int = Field(default=32, ge=1, le=128, strict=True)
    max_input_bytes: int = Field(default=1048576, ge=1, le=16777216, strict=True)


class RetrievalModelConfiguration(Contract):
    version: Literal["retrieval-models@1"] = "retrieval-models@1"
    connections: tuple[ProviderConnection, ...] = Field(min_length=1, max_length=32)
    profiles: tuple[RetrievalModelProfile, ...] = Field(min_length=1, max_length=64)
    embedding_profile_ref: Identifier
    rerank_profile_ref: Identifier
    limits: RetrievalLimits = RetrievalLimits()

    @model_validator(mode="after")
    def unique_refs(self) -> Self:
        for entries in (self.connections, self.profiles):
            if len({entry.ref for entry in entries}) != len(entries):
                raise ValueError("检索连接或profile引用不能重复")
        return self


class RetrievalCapabilityEvidence(Contract):
    ref: Identifier
    provider_id: Identifier
    model_id: Identifier
    task_type: RetrievalTask
    definition_hash: Digest
    catalog_hash: Digest
    connection_hash: Digest
    implementation: InstalledModelProtocol
    dimensions: int | None = Field(default=None, ge=1, le=2048, strict=True)
    max_items: int = Field(ge=1, le=40, strict=True)
    max_characters: int = Field(ge=1, le=8000, strict=True)
    environment_profile_hash: Digest
    report_hash: Digest
    verified_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def task_and_time(self) -> Self:
        if self.task_type != self.implementation.task_type or self.expires_at <= self.verified_at:
            raise ValueError("检索证据的任务或有效期不合法")
        if self.task_type == "embedding" and (self.dimensions is None or self.max_items > 10):
            raise ValueError("Embedding证据必须固定维度和实际批次")
        if self.task_type == "rerank" and self.dimensions is not None:
            raise ValueError("Rerank证据不能声明向量维度")
        return self

    def valid_at(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("时间必须带时区")
        return self.verified_at <= now < self.expires_at


class RetrievalModelBinding(Contract):
    connection: ProviderConnection
    profile: RetrievalModelProfile
    implementation: InstalledModelProtocol
    evidence: RetrievalCapabilityEvidence
    snapshot: JsonSnapshot

    @model_validator(mode="after")
    def exact_snapshot(self) -> Self:
        actual = self.snapshot.as_object()
        for name in ("connection", "profile", "implementation", "evidence"):
            if actual.get(name) != getattr(self, name).model_dump(mode="json"):
                raise ValueError("检索模型绑定与深快照不一致")
        return self


class RetrievalScope(Contract):
    """由服务端装配创建；Authority必须校验实际owner、知识范围及实时权限。"""

    context_id: Identifier
    tenant_id: Identifier
    subject_id: Identifier
    application_id: Identifier
    owner_kind: Literal["run", "ingestion_job"]
    owner_id: Identifier
    knowledge_base_id: Identifier
    manifest_hash: Digest
    embedding_binding_hash: Digest
    rerank_binding_hash: Digest
    deadline: AwareDatetime


class PreparedRetrievalCall(Contract):
    scope: RetrievalScope
    call_id: Identifier
    attempt_id: Identifier
    binding_hash: Digest
    payload: RetrievalPayload = Field(repr=False)

    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"attempt_id"}))


def validate_retrieval_call(
    request: PreparedRetrievalCall, binding: RetrievalModelBinding, now: datetime
) -> None:
    """共同的发送边界；模型身份、调用范围、单批大小均不可由payload覆盖。"""
    profile, payload = binding.profile, request.payload
    scope_hash = (request.scope.embedding_binding_hash if profile.task_type == "embedding"
                  else request.scope.rerank_binding_hash)
    if (request.binding_hash != binding.snapshot.content_hash or scope_hash != request.binding_hash
        or payload.task_type != profile.task_type or now >= request.scope.deadline):
        raise ValueError("检索请求绑定或时限不匹配")
    texts = payload.texts if isinstance(payload, EmbeddingRequest) else (payload.query, *payload.documents)
    count = len(payload.texts) if isinstance(payload, EmbeddingRequest) else len(payload.documents)
    input_bytes = len(canonical_json(payload.model_dump(mode="json")).encode("utf-8"))
    if (count > profile.max_items or any(len(text) > profile.max_characters for text in texts)
        or input_bytes > profile.max_input_bytes):
        raise ValueError("检索模型输入超过固定profile上限")


class RetrievalResultIdentity(Contract):
    call_id: Identifier
    attempt_id: Identifier
    request_fingerprint: Digest
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    usage: RetrievalUsage | None = None


Vector = Annotated[tuple[FiniteFloat, ...], Field(min_length=1, max_length=2048)]


class EmbeddingResult(RetrievalResultIdentity):
    task_type: Literal["embedding"] = "embedding"
    vectors: tuple[Vector, ...] = Field(min_length=1, max_length=10, repr=False)


class RerankItem(Contract):
    index: int = Field(ge=0, le=39, strict=True)
    relevance_score: FiniteFloat


class RerankResult(RetrievalResultIdentity):
    task_type: Literal["rerank"] = "rerank"
    items: tuple[RerankItem, ...] = Field(min_length=1, max_length=40)


RetrievalResult = Annotated[EmbeddingResult | RerankResult, Field(discriminator="task_type")]


class RetrievalReservation(Contract):
    request: PreparedRetrievalCall
    cached_result: RetrievalResult | None = None


class EmbeddingPort(Protocol):
    async def embed(self, request: PreparedRetrievalCall) -> EmbeddingResult: ...


class RerankPort(Protocol):
    async def rerank(self, request: PreparedRetrievalCall) -> RerankResult: ...


class RetrievalAuthority(Protocol):
    async def authorize(self, scope: RetrievalScope, payload: RetrievalPayload) -> None: ...


class RetrievalLedger(Protocol):
    async def reserve(
        self, request: PreparedRetrievalCall, limits: RetrievalLimits
    ) -> RetrievalReservation: ...

    async def succeed(self, request: PreparedRetrievalCall, result: RetrievalResult) -> None: ...

    async def fail(self, request: PreparedRetrievalCall, failure: JsonSnapshot) -> None: ...
