"""真实PG固定run/额度/ACL用例定义；模型和召回为受控替身，本轮未执行。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.postgres.connection import PostgresUnitOfWork
from harness.adapters.retrieval_models_postgres import PostgresRetrievalLedger
from harness.adapters.retrieval_postgres import PostgresRetrievalRepository
from harness.contracts.execution import ExecutionProfile
from harness.contracts.release import ReleaseManifest
from harness.contracts.retrieval import RecallHit, RetrievalProfile, RetrievalQuery, RetrievalRelease, SnapshotPin
from harness.contracts.retrieval_models import RerankItem, RerankResult
from harness.contracts.runtime import AdmissionProfile, RuntimeFault
from harness.contracts.specs import AgentSpec, KnowledgeBinding
from harness.retrieval.pipeline import RetrievalService
from harness.runtime.retrieval_models import GovernedRetrievalGateway
from tests.integration.test_indexing_postgres import built
from tests.integration.test_indexing_postgres import index_repository as index_repository
from tests.integration.test_ingestion_postgres import repository as repository
from tests.retrieval_fixtures import EmbeddingTransport
from tests.unit.test_ingestion import actor

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


class Connection:
    def __init__(self, connection, namespace):
        self.connection, self.namespace = connection, namespace

    async def execute(self, statement, parameters=()):
        return await self.connection.execute(statement.replace("harness_rag.", self.namespace + ".").replace(
            "harness.", self.namespace + ".runtime_"), parameters)


class Uow:
    def __init__(self, dsn, namespace):
        self.uow, self.namespace = PostgresUnitOfWork(dsn), namespace

    @asynccontextmanager
    async def transaction(self):
        async with self.uow.transaction() as connection:
            yield Connection(connection, self.namespace)


class Reranker:
    def __init__(self):
        self.calls, self.documents, self.after = 0, (), None

    async def rerank(self, request):
        self.calls += 1
        self.documents = request.payload.documents
        if self.after:
            await self.after()
        return RerankResult(call_id=request.call_id, attempt_id=request.attempt_id,
                            request_fingerprint=request.fingerprint(),
                            items=tuple(RerankItem(index=i, relevance_score=0.9) for i in range(request.payload.top_n)))


class Recall:
    def __init__(self, ids):
        self.hits = tuple(RecallHit(chunk_id=i, score=1.0) for i in ids)
        self.sparse_calls = 0

    async def dense(self, ticket, vector):
        return self.hits

    async def sparse(self, ticket):
        self.sparse_calls += 1
        return self.hits


@pytest.fixture
async def retrieval_case(repository, index_repository, pg_schema):
    dsn, namespace = pg_schema
    job = await built(repository, index_repository, pg_schema)
    snapshot = await index_repository.publish(actor(), job.job_id)
    # built使用独立但同结构的合成模型证据；以实际PG generation绑定为准。
    embedding_snapshot = job.binding.model_snapshot.as_object()["embedding"]
    rerank_snapshot = job.binding.model_snapshot.as_object()["rerank"]
    from harness.contracts.retrieval_models import RetrievalModelBinding
    from harness.contracts.snapshots import JsonSnapshot

    def restore(data):
        return RetrievalModelBinding(connection=data["connection"], profile=data["profile"],
                                      implementation=data["implementation"], evidence=data["evidence"],
                                      snapshot=JsonSnapshot.capture(data))

    bindings = (restore(embedding_snapshot), restore(rerank_snapshot))
    profile = RetrievalProfile()
    pin = SnapshotPin(knowledge_base_id="kb-1", snapshot_id=snapshot.snapshot_id, snapshot_hash=snapshot.content_hash,
                      generation_id=job.generation_id, index_binding_hash=job.binding.content_hash())
    configuration = RetrievalRelease(models=job.binding.model_snapshot, backend_hash=job.binding.backend_hash,
                                     profiles=(profile,), snapshots=(pin,))
    execution_profile = ExecutionProfile.model_validate_json((ROOT / "examples/execution-profile.example.json").read_text("utf-8"))
    spec = AgentSpec.model_validate_json((ROOT / "agents/operations.json").read_text("utf-8"))
    spec = spec.model_copy(update={"graph_template": "rag_chat", "knowledge_bindings": (
        KnowledgeBinding(knowledge_base_id="kb-1", snapshot_id=snapshot.snapshot_id, retrieval_profile_ref=profile.ref),)})
    manifest = ReleaseManifest(spec=spec, source_revision=None, source_tree_hash="1" * 64, dependency_lock_hash="2" * 64,
                                prompt_hash="3" * 64, tool_schema_hashes={f"{t.name}@{t.version}": "4" * 64 for t in spec.allowed_tools},
                                model_profile={}, profiles={"execution": execution_profile.model_dump(mode="json"),
                                                            "retrieval": configuration.model_dump(mode="json")},
                                environment_profile_hash="a" * 64)
    session_id, run_id = uuid4().hex, uuid4().hex
    with psycopg.connect(dsn) as conn:
        for name in ("0001_runtime.sql", "0002_writes.sql", "0003_execution.sql", "0004_extensions.sql"):
            text = (ROOT / "migrations" / name).read_text("utf-8").replace("CREATE SCHEMA IF NOT EXISTS harness;", "")
            conn.execute(text.replace("harness.", namespace + ".runtime_"), prepare=False)
        text = (ROOT / "migrations/rag/0004_retrieval.sql").read_text("utf-8")
        conn.execute(text.replace("harness_rag.", namespace + ".").replace("harness.", namespace + ".runtime_"), prepare=False)
        conn.execute(sql.SQL("INSERT INTO {}.runtime_releases(tenant_id,application_id,release_id,agent_id,manifest_hash,manifest) "
                             "VALUES(%s,%s,%s,%s,%s,%s)").format(sql.Identifier(namespace)),
                     (actor().tenant_id, actor().application_id, spec.release_id, spec.agent_id, manifest.content_hash(),
                      Jsonb(manifest.model_dump(mode="json"))))
        conn.execute(sql.SQL("INSERT INTO {}.runtime_sessions(session_id,tenant_id,subject_id,application_id,agent_id,release_id) "
                             "VALUES(%s,%s,%s,%s,%s,%s)").format(sql.Identifier(namespace)),
                     (session_id, actor().tenant_id, actor().subject_id, actor().application_id, spec.agent_id, spec.release_id))
        conn.execute(sql.SQL("INSERT INTO {}.runtime_runs(run_id,session_id,request_id,request_hash,request,manifest_hash,"
                             "environment_profile_hash,principal,status,queue_deadline,expires_at,execution_deadline,owner_id) "
                             "VALUES(%s,%s,'run-request',%s,'{{}}',%s,%s,%s,'running',clock_timestamp()+interval '10 minutes',"
                             "clock_timestamp()+interval '10 minutes',clock_timestamp()+interval '5 minutes','worker-1')")
                     .format(sql.Identifier(namespace)),
                     (run_id, session_id, "e" * 64, manifest.content_hash(), manifest.environment_profile_hash,
                      Jsonb(actor().model_dump(mode="json"))))
        conn.execute(sql.SQL("UPDATE {}.runtime_sessions SET active_run_id=%s WHERE session_id=%s")
                     .format(sql.Identifier(namespace)), (run_id, session_id))
        ids = [r[0] for r in conn.execute(sql.SQL("SELECT chunk_id FROM {}.snapshot_members WHERE snapshot_id=%s ORDER BY chunk_id")
                                         .format(sql.Identifier(namespace)), (snapshot.snapshot_id,)).fetchall()]
    execution = PostgresExecutionStore(dsn, AdmissionProfile())
    execution.identity_source = SimpleNamespace(current=lambda identity: actor())
    repo = PostgresRetrievalRepository(dsn, execution, lambda identity: actor())
    repo.uow = Uow(dsn, namespace)
    ledger = PostgresRetrievalLedger(dsn)
    ledger.uow = repo.uow
    embed, rerank, recall = EmbeddingTransport(), Reranker(), Recall(ids)
    models = GovernedRetrievalGateway(ledger=ledger, authority=repo, embedding=embed, reranking=rerank,
                                      embedding_binding=bindings[0], rerank_binding=bindings[1], limits=profile.model_limits,
                                      configuration_guard=lambda: None)
    return SimpleNamespace(repo=repo, configuration=configuration, service=RetrievalService(repo, recall, models, configuration),
                            run_id=run_id, embed=embed, rerank=rerank, recall=recall, namespace=namespace, dsn=dsn,
                            pin=pin, profile=profile, query=RetrievalQuery(request_id="query-1", query="index text"))


async def run_query(case):
    return await case.service.retrieve(actor(), "worker-1", case.run_id, "kb-1", case.query)


async def test_formal_pg_owner_quota_model_ledger_and_cache(retrieval_case):
    case = retrieval_case
    result = await run_query(case)
    assert result.outcome == "evidence"
    assert await run_query(case) == result
    assert case.embed.calls == case.rerank.calls == case.recall.sparse_calls == 1
    with psycopg.connect(case.dsn) as conn:
        tokens = conn.execute(sql.SQL("SELECT model_reserved_tokens FROM {}.runtime_runs WHERE run_id=%s")
                               .format(sql.Identifier(case.namespace)), (case.run_id,)).fetchone()[0]
        assert tokens == case.profile.parent_token_reservation()
    changed = case.query.model_copy(update={"query": "different"})
    with pytest.raises(RuntimeFault, match="RUN_QUERY_CONFLICT"):
        await case.service.retrieve(actor(), "worker-1", case.run_id, "kb-1", changed)


async def test_concurrent_request_only_reserves_parent_once(retrieval_case):
    c = retrieval_case
    first, second = await asyncio.gather(c.repo.open(actor(), "worker-1", c.run_id, "kb-1", c.query, c.configuration),
                                         c.repo.open(actor(), "worker-1", c.run_id, "kb-1", c.query, c.configuration))
    assert first.ticket == second.ticket
    results = await asyncio.gather(c.repo.claim(first.ticket), c.repo.claim(second.ticket), return_exceptions=True)
    assert sum(isinstance(item, RuntimeFault) for item in results) == 1


async def test_document_acl_blocks_body_before_rerank(retrieval_case):
    c = retrieval_case
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("INSERT INTO {}.document_access(tenant_id,application_id,knowledge_base_id,document_id) "
                             "SELECT tenant_id,application_id,knowledge_base_id,document_id FROM {}.documents")
                     .format(sql.Identifier(c.namespace), sql.Identifier(c.namespace)))
    result = await run_query(c)
    assert result.outcome == "no_evidence" and not result.evidence and c.rerank.calls == 0


async def test_revocation_during_rerank_and_cache_read_fail_closed(retrieval_case):
    c = retrieval_case

    async def revoke():
        with psycopg.connect(c.dsn) as conn:
            conn.execute(sql.SQL("UPDATE {}.documents SET revoked_at=clock_timestamp()")
                         .format(sql.Identifier(c.namespace)))

    c.rerank.after = revoke
    with pytest.raises(RuntimeFault, match="EVIDENCE_DENIED"):
        await run_query(c)
    assert c.rerank.calls == 1


async def test_completed_cache_rechecks_grant(retrieval_case):
    c = retrieval_case
    await run_query(c)
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("DELETE FROM {}.knowledge_grants").format(sql.Identifier(c.namespace)))
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await run_query(c)
    assert c.embed.calls == 1


async def test_parent_budget_cancel_and_cross_identity_refused(retrieval_case):
    c = retrieval_case
    with pytest.raises(RuntimeFault):
        await c.service.retrieve(actor().model_copy(update={"tenant_id": "other"}), "worker-1", c.run_id, "kb-1", c.query)
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("UPDATE {}.runtime_runs SET model_reserved_tokens=31999 WHERE run_id=%s")
                     .format(sql.Identifier(c.namespace)), (c.run_id,))
    with pytest.raises(RuntimeFault, match="PARENT_BUDGET_EXCEEDED"):
        await run_query(c)
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("UPDATE {}.runtime_runs SET model_reserved_tokens=0,cancel_requested=true WHERE run_id=%s")
                     .format(sql.Identifier(c.namespace)), (c.run_id,))
    with pytest.raises(RuntimeFault, match="EXECUTION_STOPPED"):
        await run_query(c)
