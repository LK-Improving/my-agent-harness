"""可选索引代际与快照契约；模型、成员与后端身份不可原地替换。"""

from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.providers import Digest
from harness.contracts.retrieval_models import EmbeddingRequest, EmbeddingResult, RetrievalScope, Vector
from harness.contracts.runtime import Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier


class IndexProfile(Contract):
    version: Literal["milvus-dual@1"] = "milvus-dual@1"
    dimensions: int = Field(default=1024, ge=1, le=2048, strict=True)
    dense_index: Literal["FLAT"] = "FLAT"
    dense_metric: Literal["COSINE"] = "COSINE"
    sparse_index: Literal["SPARSE_INVERTED_INDEX"] = "SPARSE_INVERTED_INDEX"
    sparse_metric: Literal["BM25"] = "BM25"
    tokenizer: Literal["jieba"] = "jieba"
    consistency: Literal["Strong"] = "Strong"
    protocol: Literal["milvus-single-rpc@1"] = "milvus-single-rpc@1"
    sdk_version: Literal["2.6.17"] = "2.6.17"
    max_chunks: int = Field(default=1000, ge=1, le=1000, strict=True)
    max_characters: int = Field(default=1000000, ge=1, le=1000000, strict=True)
    build_timeout_seconds: int = Field(default=900, ge=1, le=900, strict=True)


class IndexRequest(Contract):
    request_id: Identifier
    version_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def canonical_members(self) -> Self:
        if tuple(sorted(set(self.version_ids))) != self.version_ids:
            raise ValueError("文档版本集合必须唯一且按ID排序")
        return self


class IndexMember(Contract):
    chunk_id: Digest
    version_id: Identifier
    document_id: Identifier
    body_hash: Digest
    body: str = Field(min_length=1, max_length=8000, repr=False)

    def identity(self) -> dict[str, str]:
        return self.model_dump(exclude={"body"})


def member_hash(members: tuple[IndexMember, ...]) -> str:
    return canonical_hash([m.identity() for m in sorted(members, key=lambda member: member.chunk_id)])


class IndexBinding(Contract):
    profile: IndexProfile
    model_snapshot: JsonSnapshot
    embedding_binding_hash: Digest
    rerank_binding_hash: Digest
    backend_hash: Digest

    @model_validator(mode="after")
    def fixed_models(self) -> Self:
        snapshot = self.model_snapshot.as_object()
        embedding, rerank = snapshot.get("embedding"), snapshot.get("rerank")
        if not isinstance(embedding, dict) or not isinstance(rerank, dict):
            raise ValueError("索引必须绑定完整检索模型快照")
        profile = embedding.get("profile")
        if (canonical_hash(embedding) != self.embedding_binding_hash
            or canonical_hash(rerank) != self.rerank_binding_hash
            or not isinstance(profile, dict) or profile.get("dimensions") != self.profile.dimensions):
            raise ValueError("索引模型hash或维度与快照不一致")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class IndexJob(Contract):
    job_id: Identifier
    generation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    knowledge_base_id: Identifier
    version_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)
    binding: IndexBinding
    scope: RetrievalScope
    members_hash: Digest
    chunk_count: int = Field(ge=1, le=1000, strict=True)
    state: Literal["queued", "building", "ready", "failed", "offline"]
    batches_written: int = Field(default=0, ge=0)
    batches_expected: int = Field(default=0, ge=0, le=128)
    error_code: str | None = None

    @model_validator(mode="after")
    def bound_owner(self) -> Self:
        if (self.scope.owner_kind != "ingestion_job" or self.scope.owner_id != self.job_id
            or self.scope.knowledge_base_id != self.knowledge_base_id
            or self.scope.manifest_hash != self.binding.content_hash()
            or self.scope.embedding_binding_hash != self.binding.embedding_binding_hash
            or self.scope.rerank_binding_hash != self.binding.rerank_binding_hash
            or tuple(sorted(set(self.version_ids))) != self.version_ids):
            raise ValueError("代际必须绑定实际入库owner、固定知识范围及模型manifest")
        return self

    @property
    def collection_name(self) -> str:
        return "harness_g_" + self.generation_id


class IndexReadiness(Contract):
    generation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    binding_hash: Digest
    members_hash: Digest
    row_count: int = Field(ge=1, le=1000, strict=True)
    dense_ready: bool
    sparse_ready: bool
    sample_ids: tuple[Digest, ...] = Field(min_length=1, max_length=3)


class KnowledgeSnapshot(Contract):
    snapshot_id: Identifier
    knowledge_base_id: Identifier
    generation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    content_hash: Digest
    state: Literal["published", "offline"]


class GovernedEmbedding(Protocol):
    async def embed(self, scope: RetrievalScope, call_id: str,
                    request: EmbeddingRequest) -> EmbeddingResult: ...


class DualIndexBackend(Protocol):
    async def create(self, job: IndexJob) -> None: ...
    async def insert(self, job: IndexJob, members: tuple[IndexMember, ...],
                     vectors: tuple[Vector, ...]) -> None: ...
    async def seal(self, job: IndexJob, members: tuple[IndexMember, ...],
                   samples: tuple[tuple[IndexMember, Vector], ...]) -> IndexReadiness: ...


class IndexRepository(Protocol):
    async def begin(self, actor: Principal, kb_id: str, request: IndexRequest,
                    binding: IndexBinding) -> IndexJob: ...
    async def claim(self, actor: Principal, job_id: str) -> None: ...
    async def plan(self, actor: Principal, job_id: str, batches: int) -> None: ...
    async def members(self, actor: Principal, job_id: str) -> tuple[IndexMember, ...]: ...
    async def progress(self, actor: Principal, job_id: str, expected: int) -> None: ...
    async def ready(self, actor: Principal, job_id: str, proof: IndexReadiness) -> IndexJob: ...
    async def fail(self, actor: Principal, job_id: str, code: str) -> None: ...


def sample_members(members: tuple[IndexMember, ...]) -> tuple[IndexMember, ...]:
    return tuple(members[i] for i in sorted({0, len(members) // 2, len(members) - 1}))
