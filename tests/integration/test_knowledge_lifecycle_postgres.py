"""生命周期真实隔离PG/CAS/回滚定义；索引外部端为替身，全部未运行。"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from harness.adapters.ingestion_sources import LocalSourceStore
from harness.adapters.lifecycle_postgres import PostgresLifecycleRepository
from harness.contracts.ingestion import IngestionProfile, SourceRef
from harness.contracts.knowledge_lifecycle import DeleteDocument
from harness.contracts.runtime import RuntimeFault
from harness.ingestion.lifecycle import KnowledgeLifecycle
from harness.ingestion.text import parse_text
from tests.integration.test_ingestion_postgres import persisting
from tests.integration.test_indexing_postgres import index_repository as index_repository
from tests.integration.test_ingestion_postgres import repository as repository
from tests.integration.test_retrieval_pipeline_postgres import retrieval_case as retrieval_case
from tests.unit.test_ingestion import actor, upload

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
async def lifecycle_case(retrieval_case, tmp_path):
    c = retrieval_case
    with psycopg.connect(c.dsn) as conn:
        text = (ROOT / "migrations/rag/0005_lifecycle.sql").read_text("utf-8")
        conn.execute(text.replace("harness_rag.", c.namespace + ".").replace("harness.", c.namespace + ".runtime_"), prepare=False)
        # 此fixture尚未执行运行/模型；移除刚创建的空核心引用，仅隔离清理成功路径。
        conn.execute(sql.SQL("UPDATE {}.runtime_sessions SET active_run_id=NULL").format(sql.Identifier(c.namespace)))
        conn.execute(sql.SQL("DELETE FROM {}.runtime_runs").format(sql.Identifier(c.namespace)))
        conn.execute(sql.SQL("DELETE FROM {}.runtime_sessions").format(sql.Identifier(c.namespace)))
        row = conn.execute(sql.SQL("SELECT source FROM {}.ingestion_jobs LIMIT 1").format(sql.Identifier(c.namespace))).fetchone()
    repo = PostgresLifecycleRepository(c.dsn, lambda identity: actor())
    repo.uow = c.repo.uow
    sources = LocalSourceStore(tmp_path / "sources")
    await sources.put(SourceRef.model_validate(row[0]), b"index text")

    class Indexes:
        calls, failing = 0, False

        async def delete(self, target):
            self.calls += 1
            if self.failing:
                raise RuntimeFault("MILVUS_RPC_OUTCOME_UNKNOWN", 503)

    indexes = Indexes()
    c.cleanup, c.sources, c.indexes = repo, sources, indexes
    c.request = DeleteDocument(request_id="delete-1", document_id="doc-1",
        retain_until=datetime.now(UTC) - timedelta(seconds=1), source_store_ref=sources.storage_ref())
    c.service_cleanup = KnowledgeLifecycle(repo, sources, indexes)
    return c


async def test_revoke_idempotent_plan_staged_cleanup_and_tombstone(lifecycle_case):
    c = lifecycle_case
    first, second = await asyncio.gather(c.cleanup.request_delete(actor(), "kb-1", c.request),
                                        c.cleanup.request_delete(actor(), "kb-1", c.request))
    assert first.job_id == second.job_id
    result = await c.service_cleanup.advance(actor(), first.job_id)
    assert result.state == "succeeded" and result.succeeded_steps == result.total_steps == 3
    assert c.indexes.calls == 1
    assert (await c.service_cleanup.advance(actor(), first.job_id)).state == "succeeded"
    with psycopg.connect(c.dsn) as conn:
        row = conn.execute(sql.SQL("SELECT body,purged_at FROM {}.chunks LIMIT 1").format(sql.Identifier(c.namespace))).fetchone()
        assert row[0] is None and row[1] is not None
        assert conn.execute(sql.SQL("SELECT revoked_at FROM {}.documents").format(sql.Identifier(c.namespace))).fetchone()[0]
        assert conn.execute(sql.SQL("SELECT count(*) FROM {}.snapshot_members").format(sql.Identifier(c.namespace))).fetchone()[0] > 0
    assert not list(c.sources.root.iterdir())


async def test_failures_are_bounded_and_never_restore_document(lifecycle_case):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    c.indexes.failing = True
    for _ in range(3):
        with pytest.raises(RuntimeFault, match="OUTCOME_UNKNOWN"):
            await c.service_cleanup.advance(actor(), job.job_id)
    assert (await c.cleanup.read(actor(), job.job_id)).state == "manual"
    await c.service_cleanup.advance(actor(), job.job_id)
    assert c.indexes.calls == 3
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT revoked_at FROM {}.documents").format(sql.Identifier(c.namespace))).fetchone()[0]
        assert conn.execute(sql.SQL("SELECT body FROM {}.chunks LIMIT 1").format(sql.Identifier(c.namespace))).fetchone()[0]


async def test_retention_producer_and_active_snapshot_block_without_spending_attempts(lifecycle_case):
    c = lifecycle_case
    request = c.request.model_copy(update={"retain_until": datetime.now(UTC) + timedelta(days=1)})
    job = await c.cleanup.request_delete(actor(), "kb-1", request)
    assert (await c.service_cleanup.advance(actor(), job.job_id)).reason == "RETENTION_WINDOW"
    assert c.indexes.calls == 0
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT sum(attempts) FROM {}.cleanup_steps").format(sql.Identifier(c.namespace))).fetchone()[0] == 0


async def test_conflicting_request_and_cross_tenant_are_denied(lifecycle_case):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    with pytest.raises(RuntimeFault, match="REQUEST_CONFLICT"):
        await c.cleanup.request_delete(actor(), "kb-1", c.request.model_copy(update={"request_id": "changed"}))
    with pytest.raises(RuntimeFault, match="NOT_FOUND"):
        await c.cleanup.read(actor().model_copy(update={"tenant_id": "other"}), job.job_id)


async def test_parallel_claim_does_not_dispatch_same_step_twice(lifecycle_case):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    claims = await asyncio.gather(c.cleanup.claim(actor(), job.job_id), c.cleanup.claim(actor(), job.job_id))
    assert sum(value is not None for value in claims) == 1
    assert (await c.cleanup.read(actor(), job.job_id)).reason == "ATTEMPT_UNRESOLVED"


@pytest.mark.parametrize("blocker,reason", [
    ("producer", "PRODUCER_NOT_DRAINED"), ("snapshot", "ACTIVE_SNAPSHOT"), ("hold", "RETENTION_HOLD"),
])
async def test_blocking_references_keep_content_and_zero_dispatch(lifecycle_case, blocker, reason):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    with psycopg.connect(c.dsn) as conn:
        ns = sql.Identifier(c.namespace)
        if blocker == "producer":
            conn.execute(sql.SQL("UPDATE {}.index_jobs SET state='building',readiness=NULL").format(ns))
        elif blocker == "snapshot":
            conn.execute(sql.SQL("INSERT INTO {}.active_snapshots(tenant_id,application_id,knowledge_base_id,snapshot_id) "
                                 "VALUES('tenant-1','app-1','kb-1',%s)").format(ns), (c.pin.snapshot_id,))
        else:
            conn.execute(sql.SQL("INSERT INTO {}.retention_holds VALUES('tenant-1','app-1','kb-1','doc-1',"
                                 "'hold-1','REVIEW',clock_timestamp()+interval '1 day')").format(ns))
    result = await c.service_cleanup.advance(actor(), job.job_id)
    assert result.state == "blocked" and result.reason == reason and c.indexes.calls == 0


async def test_shared_source_is_retained_until_last_document_releases_it(lifecycle_case, repository):
    c = lifecycle_case
    with psycopg.connect(c.dsn) as conn:
        data = conn.execute(sql.SQL("SELECT source FROM {}.ingestion_jobs LIMIT 1").format(sql.Identifier(c.namespace))).fetchone()[0]
    second = await repository.begin(actor(), "kb-1", upload(request_id="second-upload", document_id="doc-2"),
                                    SourceRef.model_validate(data), IngestionProfile())
    await persisting(repository, second)
    await repository.finish(actor(), second.job_id, parse_text(b"index text", IngestionProfile()))
    first = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    result = await c.service_cleanup.advance(actor(), first.job_id)
    assert result.state == "succeeded" and result.retained_sources == 1
    assert (c.sources.root / data["object_id"]).exists()
    second_job = await c.cleanup.request_delete(actor(), "kb-1", c.request.model_copy(update={
        "request_id": "delete-2", "document_id": "doc-2"}))
    assert (await c.service_cleanup.advance(actor(), second_job.job_id)).state == "succeeded"
    assert not (c.sources.root / data["object_id"]).exists()


async def test_running_source_delete_blocks_recreation_and_changed_target(lifecycle_case, repository):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    index_step = await c.cleanup.claim(actor(), job.job_id)
    # 测试直接模拟外部删除成功回执；此处不构成真实Milvus删除证据。
    await c.cleanup.finish(actor(), index_step)
    source_step = await c.cleanup.claim(actor(), job.job_id)
    data = source_step.target.as_object()["source"]
    with pytest.raises(RuntimeFault, match="SOURCE_CLEANUP_IN_PROGRESS"):
        await repository.begin(actor(), "kb-1", upload(request_id="new-upload", document_id="new-doc"),
                                SourceRef.model_validate(data), IngestionProfile())
    with pytest.raises(RuntimeFault, match="ATTEMPT_CONFLICT"):
        await c.cleanup.finish(actor(), source_step.model_copy(update={"attempt": 2}))


async def test_purge_preserves_hash_and_cannot_restore_or_mutate_snapshot_members(lifecycle_case):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    await c.service_cleanup.advance(actor(), job.job_id)
    with psycopg.connect(c.dsn) as conn:
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute(sql.SQL("UPDATE {}.chunks SET body='restored',purged_at=NULL").format(sql.Identifier(c.namespace)))
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT body_hash FROM {}.chunks LIMIT 1").format(sql.Identifier(c.namespace))).fetchone()[0]


@pytest.mark.parametrize("reference,reason", [("session", "SESSION_EVIDENCE_RETAINED"),
                                             ("evaluation", "EVALUATION_REFERENCE")])
async def test_core_evidence_reference_is_not_removed_by_cleanup(lifecycle_case, reference, reason):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    with psycopg.connect(c.dsn) as conn:
        ns = sql.Identifier(c.namespace)
        release_id, agent_id = conn.execute(sql.SQL("SELECT release_id,agent_id FROM {}.runtime_releases LIMIT 1").format(ns)).fetchone()
        if reference == "session":
            conn.execute(sql.SQL("INSERT INTO {}.runtime_sessions(session_id,tenant_id,subject_id,application_id,agent_id,release_id) "
                "VALUES('retained-session','tenant-1','manager-1','app-1',%s,%s)").format(ns), (agent_id, release_id))
        else:
            conn.execute(sql.SQL("INSERT INTO {}.runtime_evaluation_contexts(context_id,tenant_id,subject_id,application_id,release_id,"
                "context,expires_at) VALUES('retained-evaluation','tenant-1','manager-1','app-1',%s,'{{}}',clock_timestamp()+interval '1 day')")
                .format(ns), (release_id,))
    result = await c.service_cleanup.advance(actor(), job.job_id)
    assert result.state == "blocked" and result.reason == reason and c.indexes.calls == 0


async def test_judge_run_copy_blocks_knowledge_cleanup_when_source_run_is_absent(lifecycle_case):
    c = lifecycle_case
    job = await c.cleanup.request_delete(actor(), "kb-1", c.request)
    with psycopg.connect(c.dsn) as conn:
        ns = sql.Identifier(c.namespace)
        conn.execute(sql.SQL("INSERT INTO {}.runtime_releases(tenant_id,application_id,release_id,agent_id,manifest_hash,manifest) "
            "SELECT tenant_id,application_id,'judge-release','judge',%s,"
            "jsonb_set(manifest,'{{profiles,judge}}',%s,true) FROM {}.runtime_releases LIMIT 1").format(ns, ns),
            ("f" * 64, Jsonb({"implementation": "synthetic-judge@1"})))
        conn.execute(sql.SQL("INSERT INTO {}.runtime_sessions(session_id,tenant_id,subject_id,application_id,agent_id,release_id) "
            "VALUES('judge-session','tenant-1','manager-1','app-1','judge','judge-release')").format(ns))
        conn.execute(sql.SQL("INSERT INTO {}.runtime_runs(run_id,session_id,request_id,request_hash,request,manifest_hash,"
            "environment_profile_hash,principal,queue_deadline,expires_at) VALUES('judge-run','judge-session','judge-request',"
            "%s,'{{}}',%s,%s,'{{}}',clock_timestamp()+interval '1 minute',clock_timestamp()+interval '1 hour')").format(ns),
            ("a" * 64, "f" * 64, "b" * 64))
    result = await c.service_cleanup.advance(actor(), job.job_id)
    assert result.state == "blocked" and result.reason == "JUDGE_EVIDENCE_RETAINED" and c.indexes.calls == 0
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT body FROM {}.chunks LIMIT 1").format(sql.Identifier(c.namespace))).fetchone()[0]


async def test_revocation_audit_failure_rolls_back_marker(lifecycle_case):
    c = lifecycle_case
    with psycopg.connect(c.dsn) as conn:
        ns = sql.Identifier(c.namespace)
        conn.execute(sql.SQL("CREATE FUNCTION {}.reject_lifecycle_audit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'injected audit failure'; END; $$").format(ns))
        conn.execute(sql.SQL("CREATE TRIGGER reject_lifecycle_audit BEFORE INSERT ON {}.index_audit "
            "FOR EACH ROW EXECUTE FUNCTION {}.reject_lifecycle_audit()").format(ns, ns))
    with pytest.raises(psycopg.errors.RaiseException):
        await c.cleanup.revoke(actor(), "kb-1", "doc-1")
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT revoked_at FROM {}.documents").format(sql.Identifier(c.namespace))).fetchone()[0] is None
