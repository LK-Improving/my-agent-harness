"""正式Milvus双路只读召回；只返回ID/分数，过滤串由服务端固定ticket生成。"""

import json

from harness.adapters.milvus_index import MilvusDualIndex
from harness.contracts.retrieval import RecallHit, RecallUnavailable, RetrievalTicket
from harness.contracts.retrieval_models import Vector
from harness.contracts.runtime import RuntimeFault


class MilvusRecall:
    def __init__(self, transport: MilvusDualIndex) -> None:
        self.transport = transport

    async def dense(self, ticket: RetrievalTicket, vector: Vector) -> tuple[RecallHit, ...]:
        if len(vector) != ticket.index_binding.profile.dimensions:
            raise RuntimeFault("RETRIEVAL_VECTOR_DIMENSION_MISMATCH", 409)
        return await self._search(ticket, "dense", list(vector))

    async def sparse(self, ticket: RetrievalTicket) -> tuple[RecallHit, ...]:
        return await self._search(ticket, "sparse", ticket.query.query)

    async def _search(self, ticket: RetrievalTicket, field: str, value: object) -> tuple[RecallHit, ...]:
        ticket = RetrievalTicket.model_validate_json(ticket.model_dump_json())
        guard = " and ".join(f"{name} == {json.dumps(item, ensure_ascii=False)}" for name, item in {
            "tenant": ticket.actor.tenant_id, "application": ticket.actor.application_id,
            "kb": ticket.pin.knowledge_base_id, "generation": ticket.pin.generation_id,
        }.items())
        request = self.transport.Prepare.search_requests_with_expr(
            "harness_g_" + ticket.pin.generation_id, field,
            {"metric_type": "COSINE" if field == "dense" else "BM25", "params": {}},
            ticket.profile.recall_limit, data=[value], expr=guard, output_fields=[],
            use_default_consistency=False, consistency_level=0,
        )
        try:
            response = await self.transport.request(ticket.index_binding, ticket.scope.deadline, "Search", request)
        except RuntimeFault as exc:
            if exc.code in {"MILVUS_DEPENDENCY_UNAVAILABLE", "MILVUS_RPC_OUTCOME_UNKNOWN"}:
                raise RecallUnavailable() from None
            raise
        ids, scores = tuple(response.results.ids.str_id.data), tuple(response.results.scores)
        if (response.results.num_queries != 1 or len(ids) != len(scores)
            or len(ids) > ticket.profile.recall_limit):
            raise RuntimeFault("RETRIEVAL_RECALL_PROTOCOL_INVALID", 503)
        return tuple(RecallHit(chunk_id=chunk_id, score=score) for chunk_id, score in zip(ids, scores, strict=True))
