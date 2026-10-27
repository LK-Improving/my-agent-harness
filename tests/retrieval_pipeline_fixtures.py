"""正式检索合成夹具；仅测试边界，不宣称真实PG/Milvus或模型质量。"""

from harness.contracts.retrieval import (
    Evidence, RecallHit, RetrievalProfile, RetrievalQuery, RetrievalRelease,
    RetrievalReservation, RetrievalTicket, SnapshotPin,
)
from harness.contracts.retrieval_models import EmbeddingResult, RerankItem, RerankResult
from harness.contracts.runtime import RuntimeFault
from tests.indexing_fixtures import index_members, index_values
from tests.unit.test_ingestion import actor


def retrieval_ticket():
    _, _, binding, scope = index_values()
    profile = RetrievalProfile()
    pin = SnapshotPin(knowledge_base_id="kb-1", snapshot_id="snapshot-1", snapshot_hash="d" * 64,
                      generation_id="d" * 32, index_binding_hash=binding.content_hash())
    configuration = RetrievalRelease(models=binding.model_snapshot, backend_hash=binding.backend_hash,
                                     profiles=(profile,), snapshots=(pin,))
    identity = actor()
    scope = scope.model_copy(update={"context_id": "retrieval-1", "owner_kind": "run", "owner_id": "run-1",
                                     "tenant_id": identity.tenant_id, "application_id": identity.application_id,
                                     "subject_id": identity.subject_id, "knowledge_base_id": pin.knowledge_base_id})
    ticket = RetrievalTicket(retrieval_id="retrieval-1", actor=identity, owner="worker-1", scope=scope,
                              pin=pin, index_binding=binding, profile=profile,
                              query=RetrievalQuery(request_id="query-1", query="中文资料"))
    evidence = tuple(Evidence(chunk_id=m.chunk_id, version_id=m.version_id, document_id=m.document_id,
                              body_hash=m.body_hash, body=m.body, snapshot_id=pin.snapshot_id,
                              generation_id=pin.generation_id) for m in index_members())
    return ticket, configuration, evidence


class MemoryRepository:
    def __init__(self, ticket, evidence):
        self.ticket, self.evidence = ticket, evidence
        self.state, self.result, self.denied, self.error = "queued", None, False, None
        self.stages = []

    async def open(self, *args):
        return RetrievalReservation(ticket=self.ticket, state=self.state, result=self.result)

    async def claim(self, ticket):
        assert self.state == "queued"
        self.state = "running"

    async def check(self, ticket):
        if self.denied:
            raise RuntimeFault("RETRIEVAL_ACCESS_DENIED", 403)

    async def hydrate(self, ticket, ids):
        await self.check(ticket)
        found = {item.chunk_id: item for item in self.evidence}
        return tuple(found[i] for i in ids if i in found)

    async def stage(self, ticket, name, data):
        await self.check(ticket)
        self.stages.append(name)

    async def finish(self, ticket, result):
        await self.check(ticket)
        self.result, self.state = result, "succeeded"

    async def deliver(self, ticket, result):
        await self.check(ticket)
        return result

    async def fail(self, ticket, code):
        if self.state == "running":
            self.error, self.state = code, "failed"


class Recall:
    def __init__(self, evidence):
        self.hits = tuple(RecallHit(chunk_id=e.chunk_id, score=float(i)) for i, e in enumerate(evidence))
        self.dense_error, self.sparse_error = None, None

    async def dense(self, ticket, vector):
        if self.dense_error:
            raise self.dense_error
        return self.hits

    async def sparse(self, ticket):
        if self.sparse_error:
            raise self.sparse_error
        return tuple(reversed(self.hits))


class Models:
    def __init__(self):
        self.embedding_error, self.rerank_error, self.after_rerank = None, None, None
        self.embed_calls, self.rerank_calls, self.documents = 0, 0, ()

    async def embed(self, scope, call_id, request):
        self.embed_calls += 1
        if self.embedding_error:
            raise self.embedding_error
        return EmbeddingResult(call_id=call_id, attempt_id="attempt-1", request_fingerprint="a" * 64, vectors=((0.5, 0.5),))

    async def rerank(self, scope, call_id, request):
        self.rerank_calls += 1
        self.documents = request.documents
        if self.rerank_error:
            raise self.rerank_error
        if self.after_rerank:
            self.after_rerank()
        return RerankResult(call_id=call_id, attempt_id="attempt-2", request_fingerprint="b" * 64,
                             items=tuple(RerankItem(index=i, relevance_score=0.8) for i in range(request.top_n)))
