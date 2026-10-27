"""正式入库真实PG用例定义；不以mock替代并发/回滚/撤权结论，本轮未执行。"""

import asyncio
import hashlib
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from harness.adapters.ingestion_postgres import PostgresIngestionRepository
from harness.contracts.ingestion import IngestionProfile, SourceRef
from harness.contracts.runtime import RuntimeFault
from harness.ingestion.text import parse_text
from tests.integration.test_retrieval_model_ledger import ScopedUow
from tests.unit.test_ingestion import actor, upload

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repository(pg_schema):
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        for name in ("0001_retrieval_models.sql", "0002_ingestion.sql"):
            conn.execute((ROOT / "migrations/rag" / name).read_text("utf-8").replace(
                "harness_rag", namespace), prepare=False)
    repo = PostgresIngestionRepository(dsn, lambda principal: principal)
    repo.uow = ScopedUow(dsn, namespace)
    return repo


def source(body=b"text"):
    return SourceRef(object_id="a" * 64, sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))


async def queued(repository, request=None, body=b"text"):
    await repository.create_knowledge_base(actor(), "kb-1")
    return await repository.begin(actor(), "kb-1", request or upload(), source(body), IngestionProfile())


async def persisting(repository, job):
    for expected, target in [("queued", "storing"), ("storing", "parsing"), ("parsing", "persisting")]:
        await repository.advance(actor(), job.job_id, expected, target)


async def test_parallel_upload_dedup_and_changed_body_conflict(repository):
    await repository.create_knowledge_base(actor(), "kb-1")
    first, second = await asyncio.gather(queued(repository), queued(repository))
    assert first.job_id == second.job_id and first.version_id == second.version_id
    with pytest.raises(RuntimeFault, match="REQUEST_CONFLICT"):
        await queued(repository, body=b"other")
    results = await asyncio.gather(
        repository.advance(actor(), first.job_id, "queued", "storing"),
        repository.advance(actor(), first.job_id, "queued", "storing"), return_exceptions=True,
    )
    assert sum(isinstance(result, RuntimeFault) for result in results) == 1


async def test_version_chunks_and_success_commit_together(repository):
    first = await queued(repository)
    await persisting(repository, first)
    chunks = parse_text(b"text", IngestionProfile())
    result = await repository.finish(actor(), first.job_id, chunks)
    assert result.state == "succeeded" and result.chunks_written == 1
    second = await queued(repository, upload(request_id="request-2"))
    assert second.version_id != first.version_id
    assert (await queued(repository)).version_id == first.version_id
    with pytest.raises(RuntimeFault, match="INVALID_TRANSITION"):
        await repository.finish(actor(), first.job_id, chunks)


async def test_failure_during_chunk_insert_rolls_back_version_and_success(repository, pg_schema):
    job = await queued(repository)
    await persisting(repository, job)
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("ALTER TABLE {}.chunks ADD CONSTRAINT injected_failure CHECK(false)")
                     .format(sql.Identifier(namespace)))
    with pytest.raises(psycopg.errors.CheckViolation):
        await repository.finish(actor(), job.job_id, parse_text(b"text", IngestionProfile()))
    assert (await repository.read(actor(), job.job_id)).state == "persisting"
    with psycopg.connect(dsn) as conn:
        count = conn.execute(sql.SQL("SELECT count(*) FROM {}.document_versions")
                             .format(sql.Identifier(namespace))).fetchone()[0]
    assert count == 0


async def test_revocation_and_cross_tenant_block_progress_and_read(repository, pg_schema):
    job = await queued(repository)
    for changed in ({"tenant_id": "other"}, {"application_id": "other"}, {"subject_id": "other"}):
        with pytest.raises(RuntimeFault, match="NOT_FOUND"):
            await repository.read(actor().model_copy(update=changed), job.job_id)
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("DELETE FROM {}.knowledge_grants").format(sql.Identifier(namespace)))
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await repository.advance(actor(), job.job_id, "queued", "storing")
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await repository.read(actor(), job.job_id)


async def test_current_identity_revocation_and_invalid_hash_block_commit(repository):
    job = await queued(repository)
    await persisting(repository, job)
    chunks = parse_text(b"text", IngestionProfile())
    with pytest.raises(RuntimeFault, match="CHUNKS_INVALID"):
        await repository.finish(actor(), job.job_id, (chunks[0].model_copy(update={"body_hash": "f" * 64}),))
    repository.current_identity = lambda principal: principal.model_copy(update={"roles": frozenset()})
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await repository.finish(actor(), job.job_id, chunks)


async def test_immutable_content_and_failure_does_not_downgrade_success(repository, pg_schema):
    job = await queued(repository)
    await persisting(repository, job)
    await repository.finish(actor(), job.job_id, parse_text(b"text", IngestionProfile()))
    await repository.fail(actor(), job.job_id, "LATE_FAILURE")
    assert (await repository.read(actor(), job.job_id)).state == "succeeded"
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute(sql.SQL("UPDATE {}.chunks SET body='edit'").format(sql.Identifier(namespace)))


async def test_formal_pipeline_raw_storage_parser_and_pg_commit(repository, tmp_path):
    from harness.adapters.ingestion_sources import LocalSourceStore
    from harness.adapters.ingestion_text import IsolatedTextParser
    from harness.ingestion.pipeline import IngestionService

    await repository.create_knowledge_base(actor(), "kb-1")
    service = IngestionService(repository, LocalSourceStore(tmp_path), IsolatedTextParser(), IngestionProfile())
    body = "独立基座中文资料\r\n".encode()
    job = await service.upload(actor(), "kb-1", upload(), body)
    assert job.state == "succeeded"
    assert (await service.upload(actor(), "kb-1", upload(), body)).job_id == job.job_id
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [body]


async def test_document_revocation_blocks_old_request_and_new_version(repository, pg_schema):
    job = await queued(repository)
    dsn, namespace = pg_schema
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("UPDATE {}.documents SET revoked_at=clock_timestamp()")
                     .format(sql.Identifier(namespace)))
    with pytest.raises(RuntimeFault, match="DOCUMENT_ACCESS_DENIED"):
        await queued(repository)
    with pytest.raises(RuntimeFault, match="DOCUMENT_ACCESS_DENIED"):
        await queued(repository, upload(request_id="request-2"))
    with pytest.raises(RuntimeFault, match="DOCUMENT_ACCESS_DENIED"):
        await repository.read(actor(), job.job_id)
