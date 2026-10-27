"""固定代际构建：授权正文→受治理Embedding→双路写入→核对→PG ready。"""

import asyncio
from datetime import UTC, datetime

import anyio

from harness.contracts.hashing import canonical_json
from harness.contracts.indexing import (
    DualIndexBackend, GovernedEmbedding, IndexBinding, IndexJob, IndexMember, IndexRepository,
    IndexRequest, sample_members,
)
from harness.contracts.retrieval_models import (
    EmbeddingRequest, RetrievalLimits, RetrievalModelProfile, Vector,
)
from harness.contracts.runtime import Principal, RuntimeFault


def embedding_batches(members: tuple[IndexMember, ...], profile: RetrievalModelProfile,
                      limits: RetrievalLimits) -> tuple[tuple[IndexMember, ...], ...]:
    """派发前确定完整批次，不能调用到一半再发现必然超过配置额度。"""
    batches: list[tuple[IndexMember, ...]] = []
    batch: list[IndexMember] = []

    def size(items: list[IndexMember] | tuple[IndexMember, ...]) -> int:
        return len(canonical_json(EmbeddingRequest(texts=tuple(m.body for m in items)).model_dump(
            mode="json")).encode("utf-8"))

    for member in members:
        if len(member.body) > profile.max_characters or size([member]) > profile.max_input_bytes:
            raise RuntimeFault("INDEX_MODEL_INPUT_LIMIT_EXCEEDED", 413)
        if batch and (len(batch) >= min(10, profile.max_items) or size([*batch, member]) > profile.max_input_bytes):
            batches.append(tuple(batch))
            batch = []
        batch.append(member)
    if batch:
        batches.append(tuple(batch))
    if not batches or len(batches) > limits.max_calls or sum(size(b) for b in batches) > limits.max_input_bytes:
        raise RuntimeFault("INDEX_MODEL_BUDGET_EXCEEDED", 429)
    return tuple(batches)


class IndexBuildService:
    def __init__(self, repository: IndexRepository, backend: DualIndexBackend,
                 embedding: GovernedEmbedding, binding: IndexBinding,
                 model_profile: RetrievalModelProfile, model_limits: RetrievalLimits) -> None:
        self.repository, self.backend, self.embedding = repository, backend, embedding
        self.binding, self.model_profile, self.model_limits = binding, model_profile, model_limits

    async def build(self, actor: Principal, kb_id: str, request: IndexRequest) -> IndexJob:
        job = await self.repository.begin(actor, kb_id, request, self.binding)
        if job.state != "queued":
            return job
        # CAS败者不能进入清理分支，不能把赢家标失败。
        await self.repository.claim(actor, job.job_id)
        try:
            remaining = (job.scope.deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise RuntimeFault("INDEX_BUILD_DEADLINE_EXCEEDED", 409)
            async with asyncio.timeout(remaining):
                members = await self.repository.members(actor, job.job_id)
                batches = embedding_batches(members, self.model_profile, self.model_limits)
                await self.repository.plan(actor, job.job_id, len(batches))
                await self.backend.create(job)
                wanted = {member.chunk_id for member in sample_members(members)}
                samples: list[tuple[IndexMember, Vector]] = []
                for index, batch in enumerate(batches):
                    # 网关在实际派发前、返回后重查PG owner/范围/撤权，并持久计量。
                    result = await self.embedding.embed(job.scope, f"index-batch-{index:04d}",
                                                        EmbeddingRequest(texts=tuple(m.body for m in batch)))
                    # 正文交给Milvus前再次检查实时授权；不靠第一次授权跨越整轮。
                    await self.repository.members(actor, job.job_id)
                    if len(result.vectors) != len(batch) or any(
                        len(v) != job.binding.profile.dimensions for v in result.vectors
                    ):
                        raise RuntimeFault("INDEX_VECTOR_DIMENSION_MISMATCH", 409)
                    await self.backend.insert(job, batch, result.vectors)
                    samples.extend((member, vector) for member, vector in zip(batch, result.vectors, strict=True)
                                   if member.chunk_id in wanted)
                    await self.repository.progress(actor, job.job_id, index)
                await self.repository.members(actor, job.job_id)
                proof = await self.backend.seal(job, members, tuple(samples))
                return await self.repository.ready(actor, job.job_id, proof)
        except BaseException as exc:
            code = (exc.code if isinstance(exc, RuntimeFault) else
                    "INDEX_CANCELLED" if isinstance(exc, asyncio.CancelledError) else
                    "INDEX_BUILD_TIMEOUT" if isinstance(exc, TimeoutError) else "INDEX_BUILD_FAILED")
            with anyio.CancelScope(shield=True):
                try:
                    async with asyncio.timeout(5):
                        await self.repository.fail(actor, job.job_id, code)
                except Exception:
                    # PG/权限失败保留building与调用账本，不能当作已收尾或自动接管。
                    pass
            raise
