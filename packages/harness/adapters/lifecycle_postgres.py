"""PG先撤权、固定清理目标和短事务CAS；正文副本/活动引用保守阻断清理。"""

from collections.abc import Callable
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.ingestion_postgres import PostgresIngestionRepository
from harness.adapters.postgres.connection import Connection, PostgresUnitOfWork, Row
from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import IndexJob
from harness.contracts.knowledge_lifecycle import CleanupJob, CleanupStep, DeleteDocument
from harness.contracts.runtime import Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


class PostgresLifecycleRepository:
    def __init__(self, dsn: str, current_identity: Callable[[Principal], Principal]) -> None:
        self.uow = PostgresUnitOfWork(dsn)
        self.permissions = PostgresIngestionRepository(dsn, current_identity)

    async def _document(self, conn: Connection, actor: Principal, kb: str, document: str) -> Row:
        await self.permissions.authorize_in_transaction(conn, actor, kb)
        row = await (await conn.execute("SELECT * FROM harness_rag.documents WHERE tenant_id=%s "
            "AND application_id=%s AND knowledge_base_id=%s AND document_id=%s FOR UPDATE",
            (actor.tenant_id, actor.application_id, kb, document))).fetchone()
        if row is None:
            raise RuntimeFault("CLEANUP_DOCUMENT_NOT_FOUND", 404)
        return row

    async def _audit(self, conn: Connection, actor: Principal, kb: str, action: str, resource: str,
                     details: dict[str, object]) -> None:
        await conn.execute("INSERT INTO harness_rag.index_audit(tenant_id,application_id,knowledge_base_id,"
            "subject_id,action,resource_id,details) VALUES(%s,%s,%s,%s,%s,%s,%s)",
            (actor.tenant_id, actor.application_id, kb, actor.subject_id, action, resource, Jsonb(details)))

    async def _revoke(self, conn: Connection, actor: Principal, kb: str, document: str) -> None:
        await self._document(conn, actor, kb, document)
        await conn.execute("UPDATE harness_rag.documents SET revoked_at=coalesce(revoked_at,clock_timestamp()) "
            "WHERE tenant_id=%s AND application_id=%s AND knowledge_base_id=%s AND document_id=%s",
            (actor.tenant_id, actor.application_id, kb, document))
        await self._audit(conn, actor, kb, "document_revoked", document, {})

    async def revoke(self, actor: Principal, kb_id: str, document_id: str) -> None:
        async with self.uow.transaction() as conn:
            await self._revoke(conn, actor, kb_id, document_id)

    async def request_delete(self, actor: Principal, kb_id: str, request: DeleteDocument) -> CleanupJob:
        request = DeleteDocument.model_validate_json(request.model_dump_json())
        # 撤权单独提交，后续计划上限/故障不能回滚已经生效的拒读标记。
        await self.revoke(actor, kb_id, request.document_id)
        fingerprint = canonical_hash(request.model_dump(mode="json"))
        async with self.uow.transaction() as conn:
            await self._document(conn, actor, kb_id, request.document_id)
            existing = await (await conn.execute("SELECT * FROM harness_rag.cleanup_jobs WHERE tenant_id=%s "
                "AND application_id=%s AND knowledge_base_id=%s AND (request_id=%s OR document_id=%s) FOR UPDATE",
                (actor.tenant_id, actor.application_id, kb_id, request.request_id, request.document_id))).fetchall()
            if existing:
                if len(existing) != 1 or existing[0]["fingerprint"] != fingerprint:
                    raise RuntimeFault("CLEANUP_REQUEST_CONFLICT", 409)
                return await self._view(conn, existing[0])
            sources = await (await conn.execute("SELECT DISTINCT source FROM harness_rag.ingestion_jobs WHERE "
                "tenant_id=%s AND application_id=%s AND knowledge_base_id=%s AND document_id=%s LIMIT 129",
                (actor.tenant_id, actor.application_id, kb_id, request.document_id))).fetchall()
            indexes = await (await conn.execute("SELECT DISTINCT i.* FROM harness_rag.index_jobs i "
                "JOIN harness_rag.generation_members m USING(generation_id) "
                "JOIN harness_rag.ingestion_jobs j ON j.version_id=m.version_id "
                "WHERE j.tenant_id=%s AND j.application_id=%s AND j.knowledge_base_id=%s AND j.document_id=%s "
                "ORDER BY i.generation_id LIMIT 129",
                (actor.tenant_id, actor.application_id, kb_id, request.document_id))).fetchall()
            if len(sources) > 128 or len(indexes) > 128:
                raise RuntimeFault("CLEANUP_PLAN_LIMIT_EXCEEDED", 413)
            targets: list[tuple[str, dict[str, object]]] = []
            for index in indexes:
                chunks = await (await conn.execute("SELECT m.chunk_id FROM harness_rag.generation_members m "
                    "JOIN harness_rag.ingestion_jobs j ON j.version_id=m.version_id WHERE m.generation_id=%s "
                    "AND j.document_id=%s AND j.tenant_id=%s AND j.application_id=%s AND j.knowledge_base_id=%s "
                    "ORDER BY m.chunk_id", (index["generation_id"], request.document_id, actor.tenant_id,
                                           actor.application_id, kb_id))).fetchall()
                job = IndexJob.model_validate({k: index[k] for k in IndexJob.model_fields})
                targets.append(("index", {"job": job.model_dump(mode="json"), "chunk_ids": [r["chunk_id"] for r in chunks]}))
            targets.extend(("source", {"source": row["source"], "source_store_ref": request.source_store_ref})
                           for row in sorted(sources, key=lambda r: r["source"]["object_id"]))
            targets.append(("postgres", {}))
            job_id = uuid4().hex
            await conn.execute("INSERT INTO harness_rag.cleanup_jobs(job_id,tenant_id,application_id,knowledge_base_id,"
                "document_id,subject_id,request_id,fingerprint,retain_until) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (job_id, actor.tenant_id, actor.application_id, kb_id, request.document_id, actor.subject_id,
                 request.request_id, fingerprint, request.retain_until))
            for position, (kind, target) in enumerate(targets):
                await conn.execute("INSERT INTO harness_rag.cleanup_steps(step_id,job_id,position,kind,target) "
                    "VALUES(%s,%s,%s,%s,%s)", (uuid4().hex, job_id, position, kind, Jsonb(target)))
            await self._audit(conn, actor, kb_id, "cleanup_requested", job_id, {"steps": len(targets)})
        return await self.read(actor, job_id)

    async def _job(self, conn: Connection, actor: Principal, job_id: str) -> Row:
        hint = await (await conn.execute("SELECT * FROM harness_rag.cleanup_jobs WHERE job_id=%s "
            "AND tenant_id=%s AND application_id=%s", (job_id, actor.tenant_id, actor.application_id))).fetchone()
        if hint is None:
            raise RuntimeFault("CLEANUP_JOB_NOT_FOUND", 404)
        document = await self._document(conn, actor, hint["knowledge_base_id"], hint["document_id"])
        if document["revoked_at"] is None:
            raise RuntimeFault("CLEANUP_REVOKE_REQUIRED", 409)
        row = await (await conn.execute("SELECT * FROM harness_rag.cleanup_jobs WHERE job_id=%s FOR UPDATE", (job_id,))).fetchone()
        if row is None:
            raise RuntimeFault("CLEANUP_JOB_NOT_FOUND", 404)
        return row

    async def _view(self, conn: Connection, row: Row) -> CleanupJob:
        counts = await (await conn.execute("SELECT count(*) AS total, count(*) FILTER(WHERE state IN "
            "('succeeded','retained')) AS done, count(*) FILTER(WHERE state='retained') AS retained "
            "FROM harness_rag.cleanup_steps WHERE job_id=%s", (row["job_id"],))).fetchone()
        return CleanupJob(job_id=row["job_id"], knowledge_base_id=row["knowledge_base_id"], document_id=row["document_id"],
            state=row["state"], succeeded_steps=counts["done"], total_steps=counts["total"],
            retained_sources=counts["retained"], reason=row["reason"])

    async def read(self, actor: Principal, job_id: str) -> CleanupJob:
        async with self.uow.transaction() as conn:
            return await self._view(conn, await self._job(conn, actor, job_id))

    async def _blocked(self, conn: Connection, job: Row) -> str | None:
        scope = (job["tenant_id"], job["application_id"], job["knowledge_base_id"])
        clock = await (await conn.execute("SELECT clock_timestamp() AS now")).fetchone()
        if clock["now"] < job["retain_until"]:
            return "RETENTION_WINDOW"
        hold = await (await conn.execute("SELECT 1 FROM harness_rag.retention_holds WHERE tenant_id=%s "
            "AND application_id=%s AND knowledge_base_id=%s AND document_id=%s AND until_at>clock_timestamp()",
            (*scope, job["document_id"]))).fetchone()
        if hold:
            return "RETENTION_HOLD"
        producers = await (await conn.execute("SELECT 1 FROM harness_rag.ingestion_jobs WHERE tenant_id=%s "
            "AND application_id=%s AND knowledge_base_id=%s AND state IN ('storing','parsing','persisting') "
            "UNION ALL SELECT 1 FROM harness_rag.index_jobs WHERE tenant_id=%s AND application_id=%s "
            "AND knowledge_base_id=%s AND state='building' LIMIT 1", (*scope, *scope))).fetchone()
        if producers:
            return "PRODUCER_NOT_DRAINED"
        active = await (await conn.execute("SELECT 1 FROM harness_rag.active_snapshots a "
            "JOIN harness_rag.snapshot_members m USING(snapshot_id) JOIN harness_rag.ingestion_jobs j USING(version_id) "
            "WHERE a.tenant_id=%s AND a.application_id=%s AND a.knowledge_base_id=%s AND j.document_id=%s LIMIT 1",
            (*scope, job["document_id"]))).fetchone()
        if active:
            return "ACTIVE_SNAPSHOT"
        # 保守保护整个KB的会话/检索/有效评测；不按普通TTL破坏正文副本及unknown证据。
        references = await (await conn.execute("SELECT 1 FROM harness_rag.retrieval_requests WHERE tenant_id=%s "
            "AND application_id=%s AND knowledge_base_id=%s LIMIT 1", scope)).fetchone()
        if references:
            return "RUN_EVIDENCE_RETAINED"
        binding = Jsonb([{"knowledge_base_id": job["knowledge_base_id"]}])
        sessions = await (await conn.execute("SELECT 1 FROM harness.sessions s JOIN harness.releases r "
            "USING(tenant_id,application_id,release_id) WHERE s.tenant_id=%s AND s.application_id=%s "
            "AND r.manifest->'spec'->'knowledge_bindings' @> %s LIMIT 1", (*scope[:2], binding))).fetchone()
        if sessions:
            return "SESSION_EVIDENCE_RETAINED"
        evaluations = await (await conn.execute("SELECT 1 FROM harness.evaluation_contexts e JOIN harness.releases r "
            "ON r.tenant_id=e.tenant_id AND r.application_id=e.application_id AND r.release_id=e.release_id "
            "WHERE e.tenant_id=%s AND e.application_id=%s AND e.expires_at>clock_timestamp() "
            "AND r.manifest->'spec'->'knowledge_bindings' @> %s LIMIT 1", (*scope[:2], binding))).fetchone()
        return "EVALUATION_REFERENCE" if evaluations else None

    async def _state(self, conn: Connection, job: Row, state: str, reason: str | None) -> None:
        await conn.execute("UPDATE harness_rag.cleanup_jobs SET state=%s,reason=%s,updated_at=clock_timestamp() "
                           "WHERE job_id=%s", (state, reason, job["job_id"]))

    async def claim(self, actor: Principal, job_id: str) -> CleanupStep | None:
        async with self.uow.transaction() as conn:
            job = await self._job(conn, actor, job_id)
            if job["state"] in {"succeeded", "manual"}:
                return None
            reason = await self._blocked(conn, job)
            if reason:
                await self._state(conn, job, "blocked", reason)
                return None
            while True:
                step = await (await conn.execute("SELECT * FROM harness_rag.cleanup_steps WHERE job_id=%s "
                    "AND state NOT IN ('succeeded','retained') ORDER BY position LIMIT 1 FOR UPDATE", (job_id,))).fetchone()
                if step is None:
                    await self._state(conn, job, "succeeded", None)
                    return None
                if step["state"] == "running" or step["attempts"] >= 3:
                    await self._state(conn, job, "manual", "ATTEMPT_UNRESOLVED" if step["state"] == "running" else "RETRY_LIMIT")
                    return None
                if step["kind"] == "source":
                    shared = await (await conn.execute("SELECT 1 FROM harness_rag.ingestion_jobs j WHERE "
                        "source->>'object_id'=%s AND NOT(tenant_id=%s AND application_id=%s AND knowledge_base_id=%s "
                        "AND document_id=%s) AND NOT EXISTS(SELECT 1 FROM harness_rag.source_releases x "
                        "WHERE (x.tenant_id,x.application_id,x.knowledge_base_id,x.document_id,x.object_id)="
                        "(j.tenant_id,j.application_id,j.knowledge_base_id,j.document_id,j.source->>'object_id')) LIMIT 1",
                        (step["target"]["source"]["object_id"], job["tenant_id"],
                        job["application_id"], job["knowledge_base_id"], job["document_id"]))).fetchone()
                    if shared:
                        await conn.execute("UPDATE harness_rag.cleanup_steps SET state='retained',error_code='SHARED_SOURCE',"
                            "updated_at=clock_timestamp() WHERE step_id=%s", (step["step_id"],))
                        await self._audit(conn, actor, job["knowledge_base_id"], "cleanup_source_retained", step["step_id"], {})
                        await self._release_source(conn, job, step["target"]["source"]["object_id"])
                        continue
                claimed = await (await conn.execute("UPDATE harness_rag.cleanup_steps SET state='running',attempts=attempts+1,"
                    "error_code=NULL,updated_at=clock_timestamp() WHERE step_id=%s RETURNING attempts", (step["step_id"],))).fetchone()
                await self._state(conn, job, "pending", None)
                await self._audit(conn, actor, job["knowledge_base_id"], "cleanup_dispatch", step["step_id"],
                                  {"attempt": claimed["attempts"], "kind": step["kind"]})
                return CleanupStep(step_id=step["step_id"], job_id=job_id, kind=step["kind"],
                                   attempt=claimed["attempts"], target=JsonSnapshot.capture(step["target"]))

    async def _attempt(self, conn: Connection, actor: Principal, step: CleanupStep) -> Row:
        job = await self._job(conn, actor, step.job_id)
        row = await (await conn.execute("SELECT * FROM harness_rag.cleanup_steps WHERE step_id=%s AND job_id=%s FOR UPDATE",
                                        (step.step_id, step.job_id))).fetchone()
        if (row is None or row["state"] != "running" or row["attempts"] != step.attempt
            or row["kind"] != step.kind or JsonSnapshot.capture(row["target"]) != step.target):
            raise RuntimeFault("CLEANUP_ATTEMPT_CONFLICT", 409)
        return job

    async def finish(self, actor: Principal, step: CleanupStep, *, retained: bool = False) -> None:
        if retained:
            raise RuntimeFault("CLEANUP_RETENTION_NOT_CALLER_CONTROLLED", 403)
        async with self.uow.transaction() as conn:
            job = await self._attempt(conn, actor, step)
            if step.kind == "source":
                await self._release_source(conn, job, step.target.as_object()["source"]["object_id"])
            if step.kind == "postgres":
                reason = await self._blocked(conn, job)
                if reason:
                    raise RuntimeFault("CLEANUP_REFERENCE_CHANGED", 409)
                await conn.execute("UPDATE harness_rag.chunks c SET body=NULL,purged_at=clock_timestamp() "
                    "FROM harness_rag.ingestion_jobs j WHERE c.version_id=j.version_id AND c.body IS NOT NULL "
                    "AND j.tenant_id=%s AND j.application_id=%s AND j.knowledge_base_id=%s AND j.document_id=%s",
                    (job["tenant_id"], job["application_id"], job["knowledge_base_id"], job["document_id"]))
            await conn.execute("UPDATE harness_rag.cleanup_steps SET state='succeeded',updated_at=clock_timestamp() "
                               "WHERE step_id=%s", (step.step_id,))
            await self._audit(conn, actor, job["knowledge_base_id"], "cleanup_step_succeeded", step.step_id,
                              {"attempt": step.attempt, "kind": step.kind})
            pending = await (await conn.execute("SELECT 1 FROM harness_rag.cleanup_steps WHERE job_id=%s "
                "AND state NOT IN ('succeeded','retained') LIMIT 1", (step.job_id,))).fetchone()
            if pending is None:
                await self._state(conn, job, "succeeded", None)
            else:
                # 并发观察者可能把运行中的步骤标为人工关注；原尝试正常结束后恢复进度。
                await self._state(conn, job, "pending", None)

    async def _release_source(self, conn: Connection, job: Row, object_id: str) -> None:
        await conn.execute("INSERT INTO harness_rag.source_releases(tenant_id,application_id,knowledge_base_id,"
            "document_id,object_id) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (job["tenant_id"], job["application_id"], job["knowledge_base_id"], job["document_id"], object_id))

    async def fail(self, actor: Principal, step: CleanupStep, code: str) -> None:
        async with self.uow.transaction() as conn:
            job = await self._attempt(conn, actor, step)
            # 不持久化第三方异常正文或任意错误字符串。
            allowed = {"CLEANUP_CANCELLED", "CLEANUP_IO_FAILED", "CLEANUP_REFERENCE_CHANGED",
                       "CLEANUP_SOURCE_STORE_CHANGED",
                       "CLEANUP_SOURCE_HASH_MISMATCH", "CLEANUP_SOURCE_PATH_INVALID", "INDEX_BACKEND_BINDING_MISMATCH",
                       "MILVUS_ACCESS_DENIED", "MILVUS_RPC_REJECTED", "MILVUS_RPC_OUTCOME_UNKNOWN",
                       "MILVUS_DEPENDENCY_UNAVAILABLE", "CLEANUP_INDEX_NOT_EMPTY"}
            code = code if code in allowed else "CLEANUP_IO_FAILED"
            await conn.execute("UPDATE harness_rag.cleanup_steps SET state='failed',error_code=%s,"
                "updated_at=clock_timestamp() WHERE step_id=%s", (code, step.step_id))
            await self._state(conn, job, "manual" if step.attempt == 3 else "pending", code)
            await self._audit(conn, actor, job["knowledge_base_id"], "cleanup_step_failed", step.step_id,
                              {"attempt": step.attempt, "code": code})
