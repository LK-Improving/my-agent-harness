"""正式代际/快照真实PG定义；模型与索引为受控替身，不能证明真实Milvus质量。"""

import asyncio
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from harness.adapters.indexing_postgres import PostgresIndexRepository
from harness.adapters.retrieval_models_postgres import PostgresRetrievalLedger
from harness.contracts.indexing import IndexRequest
from harness.contracts.ingestion import IngestionProfile
from harness.contracts.retrieval_models import EmbeddingRequest
from harness.contracts.runtime import RuntimeFault
from harness.ingestion.indexing import IndexBuildService
from harness.ingestion.text import parse_text
from harness.runtime.retrieval_models import GovernedRetrievalGateway
from tests.indexing_fixtures import ControlledBackend, index_values, readiness
from tests.integration.test_ingestion_postgres import persisting, queued
from tests.integration.test_ingestion_postgres import repository as repository
from tests.retrieval_fixtures import EmbeddingTransport, RerankTransport
from tests.unit.test_ingestion import actor, upload

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def index_repository(repository, pg_schema):
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute((ROOT / "migrations/rag/0003_index_snapshots.sql").read_text("utf-8").replace(
            "harness_rag", namespace), prepare=False)
    result = PostgresIndexRepository(dsn, lambda identity: actor())
    result.uow = repository.uow
    return result


async def version(repository, request_id="upload-1", body=b"index text"):
    job = await queued(repository, upload(request_id=request_id), body=body)
    await persisting(repository, job)
    return (await repository.finish(actor(), job.job_id, parse_text(body, IngestionProfile()))).version_id


def builder(index_repository, pg_schema):
    dsn, _ = pg_schema
    config, bindings, binding, _ = index_values()
    ledger = PostgresRetrievalLedger(dsn)
    ledger.uow = index_repository.uow
    transport = EmbeddingTransport()
    gateway = GovernedRetrievalGateway(ledger=ledger, authority=index_repository, embedding=transport,
                                       reranking=RerankTransport(), embedding_binding=bindings[0],
                                       rerank_binding=bindings[1], limits=config.limits, configuration_guard=lambda: None)
    service = IndexBuildService(index_repository, ControlledBackend(), gateway, binding, bindings[0].profile, config.limits)
    return service, transport


async def built(repository, index_repository, pg_schema, request_id="index-1", version_id=None):
    version_id = version_id or await version(repository)
    service, _ = builder(index_repository, pg_schema)
    return await service.build(actor(), "kb-1", IndexRequest(request_id=request_id, version_ids=(version_id,)))


async def test_formal_build_has_actual_pg_owner_model_receipts_and_idempotency(repository, index_repository, pg_schema):
    version_id = await version(repository)
    service, transport = builder(index_repository, pg_schema)
    request = IndexRequest(request_id="index-1", version_ids=(version_id,))
    job = await service.build(actor(), "kb-1", request)
    assert job.state == "ready" and job.batches_written == job.batches_expected == 1
    assert (await service.build(actor(), "kb-1", request)).job_id == job.job_id
    assert transport.calls == 1
    assert await index_repository.active(actor(), "kb-1") == {"snapshot_id": None, "revision": 0}
    changed = service.binding.model_copy(update={"backend_hash": "f" * 64})
    with pytest.raises(RuntimeFault, match="REQUEST_CONFLICT"):
        await index_repository.begin(actor(), "kb-1", request, changed)


async def test_parallel_claim_and_readiness_without_model_receipts_are_refused(repository, index_repository):
    version_id = await version(repository)
    _, _, binding, _ = index_values()
    request = IndexRequest(request_id="index-1", version_ids=(version_id,))
    first, second = await asyncio.gather(index_repository.begin(actor(), "kb-1", request, binding),
                                         index_repository.begin(actor(), "kb-1", request, binding))
    assert first.job_id == second.job_id
    results = await asyncio.gather(index_repository.claim(actor(), first.job_id),
                                    index_repository.claim(actor(), first.job_id), return_exceptions=True)
    assert sum(isinstance(result, RuntimeFault) for result in results) == 1
    await index_repository.plan(actor(), first.job_id, 1)
    await index_repository.progress(actor(), first.job_id, 0)
    members = await index_repository.members(actor(), first.job_id)
    with pytest.raises(RuntimeFault, match="MODEL_RECEIPTS_INCOMPLETE"):
        await index_repository.ready(actor(), first.job_id, readiness(first, members))
    with pytest.raises(RuntimeFault, match="NOT_READY"):
        await index_repository.publish(actor(), first.job_id)


async def test_missing_route_or_changed_proof_does_not_publish(repository, index_repository, pg_schema):
    version_id = await version(repository)
    service, _ = builder(index_repository, pg_schema)
    service.backend.sparse_ready = False
    job_request = IndexRequest(request_id="index-1", version_ids=(version_id,))
    with pytest.raises(RuntimeFault, match="READINESS_REJECTED"):
        await service.build(actor(), "kb-1", job_request)
    job = await index_repository.begin(actor(), "kb-1", job_request, service.binding)
    assert job.state == "failed"
    with pytest.raises(RuntimeFault, match="NOT_READY"):
        await index_repository.publish(actor(), job.job_id)


async def test_publish_activate_cas_switch_rollback_and_offline(repository, index_repository, pg_schema):
    first = await built(repository, index_repository, pg_schema)
    old = await index_repository.publish(actor(), first.job_id)
    assert await index_repository.publish(actor(), first.job_id) == old
    assert await index_repository.activate(actor(), old.snapshot_id, None, 0) == 1
    with pytest.raises(RuntimeFault, match="STILL_ACTIVE"):
        await index_repository.offline(actor(), old.snapshot_id)
    second = await built(repository, index_repository, pg_schema, "index-2", first.version_ids[0])
    assert second.generation_id != first.generation_id
    new = await index_repository.publish(actor(), second.job_id)
    with pytest.raises(RuntimeFault, match="ACTIVE_CONFLICT"):
        await index_repository.activate(actor(), new.snapshot_id, None, 0)
    assert await index_repository.activate(actor(), new.snapshot_id, old.snapshot_id, 1) == 2
    assert await index_repository.activate(actor(), old.snapshot_id, new.snapshot_id, 2) == 3
    await index_repository.offline(actor(), new.snapshot_id)
    with pytest.raises(RuntimeFault, match="NOT_READY"):
        await index_repository.activate(actor(), new.snapshot_id, old.snapshot_id, 3)
    assert await index_repository.deactivate(actor(), "kb-1", old.snapshot_id, 3) == 4
    await index_repository.offline(actor(), old.snapshot_id)


async def test_publication_audit_failure_rolls_back_snapshot_and_members(repository, index_repository, pg_schema):
    job = await built(repository, index_repository, pg_schema)
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("ALTER TABLE {}.index_audit ADD CONSTRAINT injected_failure "
                             "CHECK(action <> 'snapshot_published')").format(sql.Identifier(namespace)))
    with pytest.raises(psycopg.errors.CheckViolation):
        await index_repository.publish(actor(), job.job_id)
    with psycopg.connect(dsn) as conn:
        for table in ("knowledge_snapshots", "snapshot_members"):
            assert conn.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(namespace), sql.Identifier(table))).fetchone()[0] == 0


async def test_model_authority_rejects_forged_scope_body_cross_tenant_and_revocation(repository, index_repository, pg_schema):
    version_id = await version(repository)
    _, _, binding, _ = index_values()
    job = await index_repository.begin(actor(), "kb-1", IndexRequest(request_id="index-1", version_ids=(version_id,)), binding)
    await index_repository.claim(actor(), job.job_id)
    members = await index_repository.members(actor(), job.job_id)
    payload = EmbeddingRequest(texts=(members[0].body,))
    await index_repository.authorize(job.scope, payload)
    with pytest.raises(RuntimeFault, match="BODY_DENIED"):
        await index_repository.authorize(job.scope, EmbeddingRequest(texts=("not from source",)))
    with pytest.raises(RuntimeFault, match="SCOPE_MISMATCH"):
        await index_repository.authorize(job.scope.model_copy(update={"manifest_hash": "e" * 64}), payload)
    with pytest.raises(RuntimeFault, match="NOT_FOUND"):
        await index_repository.read(actor().model_copy(update={"tenant_id": "other"}), job.job_id)
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("UPDATE {}.documents SET revoked_at=clock_timestamp()").format(sql.Identifier(namespace)))
    with pytest.raises(RuntimeFault, match="DOCUMENT_ACCESS_DENIED"):
        await index_repository.authorize(job.scope, payload)


async def test_revoked_members_cannot_activate_but_can_be_taken_offline(repository, index_repository, pg_schema):
    job = await built(repository, index_repository, pg_schema)
    snapshot = await index_repository.publish(actor(), job.job_id)
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("UPDATE {}.documents SET revoked_at=clock_timestamp()").format(sql.Identifier(namespace)))
    with pytest.raises(RuntimeFault, match="DOCUMENT_ACCESS_DENIED"):
        await index_repository.activate(actor(), snapshot.snapshot_id, None, 0)
    await index_repository.offline(actor(), snapshot.snapshot_id)
