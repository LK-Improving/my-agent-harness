"""M3-03融合/降级/证据边界定义；本轮未执行。"""

import asyncio

import pytest

from harness.contracts.failures import ProviderError, ProviderErrorCode, ProviderFailure
from harness.contracts.retrieval import RecallHit, RecallUnavailable
from harness.contracts.runtime import RuntimeFault
from harness.retrieval.pipeline import RetrievalService, rrf
from tests.retrieval_pipeline_fixtures import MemoryRepository, Models, Recall, retrieval_ticket


def service():
    ticket, configuration, evidence = retrieval_ticket()
    repository, recall, models = MemoryRepository(ticket, evidence), Recall(evidence), Models()
    return RetrievalService(repository, recall, models, configuration), ticket


async def query(service, ticket):
    return await service.retrieve(ticket.actor, ticket.owner, ticket.scope.owner_id,
                                   ticket.pin.knowledge_base_id, ticket.query)


def provider_error(code):
    return ProviderError(ProviderFailure(code=code, dispatch_status="unknown", billing_status="unknown"))


def test_rrf_ignores_raw_scores_deduplicates_and_breaks_ties_by_id():
    a, b = "a" * 64, "b" * 64
    first = (RecallHit(chunk_id=a, score=-999), RecallHit(chunk_id=a, score=999), RecallHit(chunk_id=b, score=1e30))
    second = (RecallHit(chunk_id=b, score=-10000), RecallHit(chunk_id=a, score=10000))
    assert rrf((first, second), 60, 2) == (a, b)


async def test_happy_path_is_cached_and_evidence_has_exact_versions_hashes():
    pipeline, ticket = service()
    result = await query(pipeline, ticket)
    assert result.outcome == "evidence" and not result.degraded
    assert {e.chunk_id for e in result.evidence} == {e.chunk_id for e in pipeline.repository.evidence}
    assert await query(pipeline, ticket) == result
    assert pipeline.models.embed_calls == pipeline.models.rerank_calls == 1
    assert pipeline.repository.stages == ["recall", "evidence"]


async def test_single_route_failure_is_explicit_but_both_routes_fail_closed():
    pipeline, ticket = service()
    pipeline.recall.dense_error = RecallUnavailable()
    assert (await query(pipeline, ticket)).degraded == ("dense_unavailable",)
    pipeline, ticket = service()
    pipeline.recall.dense_error = pipeline.recall.sparse_error = RecallUnavailable()
    with pytest.raises(RuntimeFault, match="ALL_ROUTES_UNAVAILABLE"):
        await query(pipeline, ticket)
    assert pipeline.repository.state == "failed"


async def test_empty_recall_is_no_evidence_and_skips_rerank():
    pipeline, ticket = service()
    pipeline.recall.hits = ()
    result = await query(pipeline, ticket)
    assert result.outcome == "no_evidence" and not result.evidence and not result.degraded
    assert pipeline.models.rerank_calls == 0


async def test_only_authorized_chunks_are_sent_to_rerank():
    pipeline, ticket = service()
    forbidden_body = pipeline.repository.evidence[0].body
    pipeline.repository.evidence = pipeline.repository.evidence[1:]
    result = await query(pipeline, ticket)
    assert forbidden_body not in pipeline.models.documents
    assert all(e.body != forbidden_body for e in result.evidence)


async def test_rerank_availability_fallback_and_authentication_failure_are_different():
    pipeline, ticket = service()
    pipeline.models.rerank_error = provider_error(ProviderErrorCode.PROVIDER_UNAVAILABLE)
    assert (await query(pipeline, ticket)).degraded == ("rerank_unavailable",)
    pipeline, ticket = service()
    pipeline.models.rerank_error = provider_error(ProviderErrorCode.ACCESS_DENIED)
    with pytest.raises(ProviderError):
        await query(pipeline, ticket)
    assert pipeline.repository.state == "failed"


async def test_revocation_after_rerank_and_on_cache_read_blocks_delivery():
    pipeline, ticket = service()
    pipeline.models.after_rerank = lambda: setattr(pipeline.repository, "denied", True)
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await query(pipeline, ticket)
    pipeline, ticket = service()
    await query(pipeline, ticket)
    pipeline.repository.denied = True
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await query(pipeline, ticket)
    assert pipeline.models.embed_calls == 1


@pytest.mark.parametrize("failure", [RuntimeFault("PG_UNAVAILABLE", 503), RuntimeFault("MILVUS_ACCESS_DENIED", 403), asyncio.CancelledError()])
async def test_pg_permission_and_cancellation_never_degrade(failure):
    pipeline, ticket = service()
    pipeline.recall.sparse_error = failure
    with pytest.raises(type(failure)):
        await query(pipeline, ticket)
    assert pipeline.repository.result is None


async def test_disabled_bootstrap_does_not_require_any_optional_configuration(tmp_path):
    from harness.bootstrap.retrieval import load_retrieval

    assert load_retrieval(tmp_path, enabled=False) is None
