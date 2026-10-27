"""并行召回、名次融合、授权回表与有界重排；不把权限/PG故障当作空结果。"""

import asyncio
from collections.abc import Awaitable

import anyio

from harness.contracts.failures import ProviderError, ProviderErrorCode
from harness.contracts.hashing import canonical_json
from harness.contracts.retrieval import (
    Evidence, RecallHit, RecallPort, RecallUnavailable, RetrievalModels, RetrievalQuery,
    RetrievalRelease, RetrievalRepository, RetrievalResult, RetrievalTicket,
)
from harness.contracts.retrieval_models import EmbeddingRequest, RerankEvidence, RerankRequest
from harness.contracts.runtime import Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot

_AVAILABLE_FAILURES = frozenset({ProviderErrorCode.RATE_LIMITED, ProviderErrorCode.TIMEOUT,
                               ProviderErrorCode.TRANSPORT_ERROR, ProviderErrorCode.PROVIDER_UNAVAILABLE})


def rrf(rankings: tuple[tuple[RecallHit, ...], ...], k: int, limit: int) -> tuple[str, ...]:
    scores: dict[str, float] = {}
    for hits in rankings:
        # 每条通路先去重；原始分数不参与加权，避免量纲和重复ID污染。
        for rank, chunk_id in enumerate(dict.fromkeys(hit.chunk_id for hit in hits), 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return tuple(sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit])


class RetrievalService:
    def __init__(self, repository: RetrievalRepository, recall: RecallPort, models: RetrievalModels,
                 configuration: RetrievalRelease) -> None:
        self.repository, self.recall, self.models, self.configuration = repository, recall, models, configuration

    async def authorize_result(self, actor: Principal, owner: str, run_id: str,
                               result: RetrievalResult) -> RetrievalResult:
        return await self.repository.authorize_result(actor, owner, run_id, result)

    async def _recall(self, ticket: RetrievalTicket, call: Awaitable[tuple[RecallHit, ...]]) -> tuple[RecallHit, ...] | None:
        try:
            async with asyncio.timeout(ticket.profile.branch_timeout_seconds):
                hits = await call
            if len(hits) > ticket.profile.recall_limit:
                raise RuntimeFault("RETRIEVAL_RECALL_LIMIT_VIOLATED", 503)
            return tuple(RecallHit.model_validate_json(hit.model_dump_json()) for hit in hits)
        except (RecallUnavailable, TimeoutError):
            return None

    async def _dense(self, ticket: RetrievalTicket) -> tuple[RecallHit, ...] | None:
        try:
            result = await self.models.embed(ticket.scope, "query-embedding", EmbeddingRequest(texts=(ticket.query.query,)))
        except ProviderError as exc:
            if exc.failure.code in _AVAILABLE_FAILURES:
                return None
            raise
        await self.repository.check(ticket)
        if len(result.vectors) != 1 or len(result.vectors[0]) != ticket.index_binding.profile.dimensions:
            raise RuntimeFault("RETRIEVAL_QUERY_VECTOR_MISMATCH", 409)
        return await self._recall(ticket, self.recall.dense(ticket, result.vectors[0]))

    async def _parallel(self, ticket: RetrievalTicket) -> tuple[tuple[RecallHit, ...] | None, tuple[RecallHit, ...] | None]:
        async def sparse() -> tuple[RecallHit, ...] | None:
            return await self._recall(ticket, self.recall.sparse(ticket))

        tasks = (asyncio.create_task(self._dense(ticket)),
                 asyncio.create_task(sparse()))
        try:
            results = await asyncio.gather(*tasks)
            return results[0], results[1]
        except BaseException:
            for task in tasks:
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def retrieve(self, actor: Principal, owner: str, run_id: str, kb_id: str,
                       query: RetrievalQuery) -> RetrievalResult:
        query = RetrievalQuery.model_validate_json(query.model_dump_json())
        reservation = await self.repository.open(actor, owner, run_id, kb_id, query, self.configuration)
        ticket = reservation.ticket
        if reservation.state == "succeeded" and reservation.result is not None:
            return await self.repository.deliver(ticket, reservation.result)
        if reservation.state != "queued":
            raise RuntimeFault("RETRIEVAL_REQUEST_NOT_REPLAYABLE", 409)
        await self.repository.claim(ticket)
        try:
            await self.repository.check(ticket)
            dense, sparse = await self._parallel(ticket)
            degraded = []
            if dense is None:
                degraded.append("dense_unavailable")
            if sparse is None:
                degraded.append("sparse_unavailable")
            await self.repository.stage(ticket, "recall", JsonSnapshot.capture({
                "dense": None if dense is None else len(dense), "sparse": None if sparse is None else len(sparse),
                "degraded": degraded,
            }))
            if dense is None and sparse is None:
                raise RuntimeFault("RETRIEVAL_ALL_ROUTES_UNAVAILABLE", 503)
            ids = rrf(tuple(hits for hits in (dense, sparse) if hits is not None), ticket.profile.rrf_k,
                      ticket.profile.fusion_limit)
            authorized = await self.repository.hydrate(ticket, ids)
            selected: list[Evidence] = []
            characters = 0
            embedding_bytes = len(canonical_json(EmbeddingRequest(texts=(query.query,)).model_dump(mode="json")).encode("utf-8"))

            def rerank_payload(items: list[Evidence]) -> RerankRequest:
                return RerankRequest(query=query.query, documents=tuple(item.body for item in items),
                                     evidence=tuple(RerankEvidence(chunk_id=item.chunk_id, version_id=item.version_id,
                                                                   body_hash=item.body_hash) for item in items),
                                     top_n=min(ticket.profile.top_n, len(items)))

            for item in authorized:
                candidate = [*selected, item]
                candidate_bytes = len(canonical_json(rerank_payload(candidate).model_dump(mode="json")).encode("utf-8"))
                if (characters + len(item.body) <= min(ticket.profile.rerank_characters, ticket.profile.evidence_characters)
                    and embedding_bytes + candidate_bytes <= ticket.profile.model_limits.max_input_bytes):
                    selected.append(item)
                    characters += len(item.body)
            if authorized and not selected:
                raise RuntimeFault("RETRIEVAL_EVIDENCE_BUDGET_EXCEEDED", 429)
            if selected:
                payload = rerank_payload(selected)
                size = len(canonical_json(payload.model_dump(mode="json")).encode("utf-8"))
                # 网关仍做最终单批/累计预算预留；不能因输入过大悄悄跳过必需重排。
                if size > ticket.profile.model_limits.max_input_bytes:
                    raise RuntimeFault("RETRIEVAL_RERANK_INPUT_TOO_LARGE", 429)
                try:
                    ranked = await self.models.rerank(ticket.scope, "evidence-rerank", payload)
                    indexes = [item.index for item in ranked.items]
                    if len(indexes) != payload.top_n or len(set(indexes)) != len(indexes) or any(
                        i < 0 or i >= len(selected) for i in indexes
                    ):
                        raise RuntimeFault("RETRIEVAL_RERANK_RESULT_INVALID", 503)
                    selected = [selected[i] for i in indexes]
                except ProviderError as exc:
                    if exc.failure.code not in _AVAILABLE_FAILURES:
                        raise
                    degraded.append("rerank_unavailable")
                    selected = selected[:ticket.profile.top_n]
            await self.repository.stage(ticket, "evidence", JsonSnapshot.capture({
                "candidates": len(ids), "authorized": len(authorized), "delivered": len(selected), "degraded": degraded,
            }))
            result = RetrievalResult.model_validate({
                "retrieval_id": ticket.retrieval_id, "outcome": "evidence" if selected else "no_evidence",
                "evidence": tuple(selected), "degraded": tuple(degraded),
            })
            await self.repository.finish(ticket, result)
            return await self.repository.deliver(ticket, result)
        except BaseException as exc:
            code = exc.code if isinstance(exc, RuntimeFault) else (
                "RETRIEVAL_CANCELLED" if isinstance(exc, asyncio.CancelledError) else "RETRIEVAL_FAILED")
            with anyio.CancelScope(shield=True):
                try:
                    async with asyncio.timeout(5):
                        await self.repository.fail(ticket, code)
                except Exception:
                    pass
            raise
