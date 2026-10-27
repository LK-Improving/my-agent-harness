"""PG索引代际/快照权威仓库；KB→文档→job→快照/指针短事务。"""

from collections import Counter
from collections.abc import Callable
from datetime import timedelta
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.ingestion_postgres import PostgresIngestionRepository
from harness.adapters.postgres.connection import Connection, PostgresUnitOfWork, Row
from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import (
    IndexBinding, IndexJob, IndexMember, IndexReadiness, IndexRequest, KnowledgeSnapshot,
    member_hash, sample_members,
)
from harness.contracts.retrieval_models import EmbeddingRequest, RetrievalPayload, RetrievalScope
from harness.contracts.runtime import Principal, RuntimeFault


def index_job(row: Row) -> IndexJob:
    return IndexJob.model_validate({key: row[key] for key in IndexJob.model_fields})


def snapshot_view(row: Row) -> KnowledgeSnapshot:
    return KnowledgeSnapshot.model_validate({key: row[key] for key in KnowledgeSnapshot.model_fields})


class PostgresIndexRepository:
    def __init__(self, dsn: str, current_identity: Callable[[Principal], Principal]) -> None:
        self.uow = PostgresUnitOfWork(dsn)
        self.permissions = PostgresIngestionRepository(dsn, current_identity)

    async def _audit(self, conn: Connection, actor: Principal, kb_id: str,
                     action: str, resource: str, details: dict[str, object]) -> None:
        await conn.execute(
            "INSERT INTO harness_rag.index_audit(tenant_id,application_id,knowledge_base_id,"
            "subject_id,action,resource_id,details) VALUES(%s,%s,%s,%s,%s,%s,%s)",
            (actor.tenant_id, actor.application_id, kb_id, actor.subject_id, action, resource, Jsonb(details)),
        )

    async def _versions(self, conn: Connection, actor: Principal, kb_id: str,
                        version_ids: tuple[str, ...]) -> None:
        rows = await (await conn.execute(
            "SELECT j.document_id,v.version_id FROM harness_rag.document_versions v "
            "JOIN harness_rag.ingestion_jobs j ON j.job_id=v.job_id "
            "WHERE v.version_id=ANY(%s) AND j.tenant_id=%s AND j.application_id=%s "
            "AND j.knowledge_base_id=%s AND j.state='succeeded' ORDER BY j.document_id,v.version_id",
            (list(version_ids), actor.tenant_id, actor.application_id, kb_id),
        )).fetchall()
        if len(rows) != len(version_ids) or len({r["document_id"] for r in rows}) != len(rows):
            raise RuntimeFault("INDEX_VERSION_SCOPE_INVALID", 403)
        for row in rows:
            await self.permissions.authorize_document_in_transaction(conn, actor, kb_id, row["document_id"])

    async def _members(self, conn: Connection, version_ids: tuple[str, ...]) -> tuple[IndexMember, ...]:
        rows = await (await conn.execute(
            "SELECT c.chunk_id,c.version_id,j.document_id,c.body_hash,c.body FROM harness_rag.chunks c "
            "JOIN harness_rag.document_versions v ON v.version_id=c.version_id "
            "JOIN harness_rag.ingestion_jobs j ON j.job_id=v.job_id "
            "WHERE c.version_id=ANY(%s) ORDER BY c.chunk_id LIMIT 1001", (list(version_ids),),
        )).fetchall()
        if not rows or len(rows) > 1000:
            raise RuntimeFault("INDEX_MEMBER_LIMIT_EXCEEDED", 413)
        members = tuple(IndexMember.model_validate(row) for row in rows)
        if any(m.body_hash != canonical_hash(m.body) for m in members):
            raise RuntimeFault("INDEX_AUTHORITY_HASH_MISMATCH", 409)
        expected = await (await conn.execute(
            "SELECT sum(chunk_count) AS count FROM harness_rag.document_versions WHERE version_id=ANY(%s)",
            (list(version_ids),),
        )).fetchone()
        if expected is None or expected["count"] != len(members):
            raise RuntimeFault("INDEX_MEMBER_COUNT_MISMATCH", 409)
        return members

    async def _job(self, conn: Connection, actor: Principal, job_id: str,
                   *, owner_only: bool = True, require_versions: bool = True) -> Row:
        hint = await (await conn.execute(
            "SELECT * FROM harness_rag.index_jobs WHERE job_id=%s AND tenant_id=%s AND application_id=%s",
            (job_id, actor.tenant_id, actor.application_id),
        )).fetchone()
        if hint is None or (owner_only and hint["subject_id"] != actor.subject_id):
            raise RuntimeFault("INDEX_JOB_NOT_FOUND", 404)
        await self.permissions.authorize_in_transaction(conn, actor, hint["knowledge_base_id"])
        if require_versions:
            await self._versions(conn, actor, hint["knowledge_base_id"], tuple(hint["version_ids"]))
        row = await (await conn.execute(
            "SELECT * FROM harness_rag.index_jobs WHERE job_id=%s FOR UPDATE", (job_id,),
        )).fetchone()
        if row is None:
            raise RuntimeFault("INDEX_JOB_NOT_FOUND", 404)
        return row

    async def begin(self, actor: Principal, kb_id: str, request: IndexRequest,
                    binding: IndexBinding) -> IndexJob:
        request = IndexRequest.model_validate_json(request.model_dump_json())
        binding = IndexBinding.model_validate_json(binding.model_dump_json())
        fingerprint = canonical_hash({"request": request.model_dump(mode="json"),
                                      "binding": binding.model_dump(mode="json")})
        async with self.uow.transaction() as conn:
            await self.permissions.authorize_in_transaction(conn, actor, kb_id)
            await self._versions(conn, actor, kb_id, request.version_ids)
            old = await (await conn.execute(
                "SELECT * FROM harness_rag.index_jobs WHERE tenant_id=%s AND application_id=%s "
                "AND knowledge_base_id=%s AND request_id=%s FOR UPDATE",
                (actor.tenant_id, actor.application_id, kb_id, request.request_id),
            )).fetchone()
            if old is not None:
                if old["subject_id"] != actor.subject_id or old["fingerprint"] != fingerprint:
                    raise RuntimeFault("INDEX_REQUEST_CONFLICT", 409)
                return index_job(old)
            members = await self._members(conn, request.version_ids)
            if (len(members) > binding.profile.max_chunks
                or sum(len(m.body) for m in members) > binding.profile.max_characters):
                raise RuntimeFault("INDEX_MEMBER_LIMIT_EXCEEDED", 413)
            now = await (await conn.execute("SELECT clock_timestamp() AS now")).fetchone()
            if now is None:
                raise RuntimeFault("INDEX_CLOCK_UNAVAILABLE", 503)
            job_id, generation = uuid4().hex, uuid4().hex
            scope = RetrievalScope(
                context_id=uuid4().hex, tenant_id=actor.tenant_id, application_id=actor.application_id,
                subject_id=actor.subject_id, owner_kind="ingestion_job", owner_id=job_id,
                knowledge_base_id=kb_id, manifest_hash=binding.content_hash(),
                embedding_binding_hash=binding.embedding_binding_hash,
                rerank_binding_hash=binding.rerank_binding_hash,
                deadline=now["now"] + timedelta(seconds=binding.profile.build_timeout_seconds),
            )
            row = await (await conn.execute(
                "INSERT INTO harness_rag.index_jobs(job_id,generation_id,tenant_id,application_id,"
                "knowledge_base_id,subject_id,request_id,fingerprint,version_ids,binding,scope,"
                "members_hash,chunk_count,state) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued') "
                "RETURNING *",
                (job_id, generation, actor.tenant_id, actor.application_id, kb_id, actor.subject_id,
                 request.request_id, fingerprint, Jsonb(list(request.version_ids)),
                 Jsonb(binding.model_dump(mode="json")), Jsonb(scope.model_dump(mode="json")),
                 member_hash(members), len(members)),
            )).fetchone()
            for member in members:
                await conn.execute(
                    "INSERT INTO harness_rag.generation_members VALUES(%s,%s,%s,%s)",
                    (generation, member.chunk_id, member.version_id, member.body_hash),
                )
            await self._audit(conn, actor, kb_id, "index_created", job_id,
                              {"generation_id": generation, "member_count": len(members)})
            if row is None:
                raise RuntimeFault("INDEX_JOB_CREATE_FAILED")
            return index_job(row)

    async def read(self, actor: Principal, job_id: str) -> IndexJob:
        async with self.uow.transaction() as conn:
            return index_job(await self._job(conn, actor, job_id))

    @staticmethod
    def _building(row: Row) -> None:
        if row["state"] != "building":
            raise RuntimeFault("INDEX_JOB_NOT_BUILDING", 409)

    async def claim(self, actor: Principal, job_id: str) -> None:
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, job_id)
            if row["state"] != "queued":
                raise RuntimeFault("INDEX_JOB_BUSY", 409)
            await self._deadline(conn, row)
            await conn.execute("UPDATE harness_rag.index_jobs SET state='building',"
                               "updated_at=clock_timestamp() WHERE job_id=%s", (job_id,))
            await self._audit(conn, actor, row["knowledge_base_id"], "index_claimed", job_id, {})

    async def members(self, actor: Principal, job_id: str) -> tuple[IndexMember, ...]:
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, job_id)
            self._building(row)
            await self._deadline(conn, row)
            result = await self._members(conn, tuple(row["version_ids"]))
            if member_hash(result) != row["members_hash"] or len(result) != row["chunk_count"]:
                raise RuntimeFault("INDEX_MEMBERS_CHANGED", 409)
            return result

    async def authorize(self, scope: RetrievalScope, payload: RetrievalPayload) -> None:
        """实际PG入库owner授权；只允许该job成员正文，不接受run/重排或伪scope。"""
        if scope.owner_kind != "ingestion_job" or not isinstance(payload, EmbeddingRequest):
            raise RuntimeFault("INDEX_MODEL_TASK_DENIED", 403)
        actor = Principal(tenant_id=scope.tenant_id, application_id=scope.application_id,
                          subject_id=scope.subject_id)
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, scope.owner_id)
            self._building(row)
            await self._deadline(conn, row)
            if RetrievalScope.model_validate(row["scope"]) != scope:
                raise RuntimeFault("INDEX_MODEL_SCOPE_MISMATCH", 403)
            members = await self._members(conn, tuple(row["version_ids"]))
            if member_hash(members) != row["members_hash"]:
                raise RuntimeFault("INDEX_MEMBERS_CHANGED", 409)
            available = Counter(member.body for member in members)
            if Counter(payload.texts) - available:
                raise RuntimeFault("INDEX_MODEL_BODY_DENIED", 403)

    async def progress(self, actor: Principal, job_id: str, expected: int) -> None:
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, job_id)
            self._building(row)
            if row["batches_written"] != expected or expected >= row["batches_expected"]:
                raise RuntimeFault("INDEX_PROGRESS_CONFLICT", 409)
            await conn.execute("UPDATE harness_rag.index_jobs SET batches_written=batches_written+1,"
                               "updated_at=clock_timestamp() WHERE job_id=%s", (job_id,))

    async def plan(self, actor: Principal, job_id: str, batches: int) -> None:
        if not 1 <= batches <= 128:
            raise RuntimeFault("INDEX_BATCH_PLAN_INVALID", 422)
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, job_id)
            self._building(row)
            if row["batches_expected"] != 0 or row["batches_written"] != 0:
                raise RuntimeFault("INDEX_BATCH_PLAN_CONFLICT", 409)
            await conn.execute("UPDATE harness_rag.index_jobs SET batches_expected=%s "
                               "WHERE job_id=%s", (batches, job_id))

    async def ready(self, actor: Principal, job_id: str, proof: IndexReadiness) -> IndexJob:
        proof = IndexReadiness.model_validate_json(proof.model_dump_json())
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, job_id)
            self._building(row)
            await self._deadline(conn, row)
            job = index_job(row)
            members = await self._members(conn, job.version_ids)
            if (proof.generation_id != job.generation_id or proof.binding_hash != job.binding.content_hash()
                or proof.members_hash != job.members_hash or member_hash(members) != job.members_hash
                or proof.row_count != job.chunk_count or not proof.dense_ready or not proof.sparse_ready
                or proof.sample_ids != tuple(m.chunk_id for m in sample_members(members))
                or row["batches_expected"] < 1 or row["batches_written"] != row["batches_expected"]):
                raise RuntimeFault("INDEX_READINESS_REJECTED", 409)
            calls = await (await conn.execute(
                "SELECT count(*) AS count,count(*) FILTER(WHERE state='succeeded' AND task_type='embedding') "
                "AS succeeded FROM harness_rag.model_calls WHERE context_id=%s", (job.scope.context_id,),
            )).fetchone()
            if calls is None or calls["count"] != row["batches_expected"] or calls["succeeded"] != calls["count"]:
                raise RuntimeFault("INDEX_MODEL_RECEIPTS_INCOMPLETE", 409)
            updated = await (await conn.execute(
                "UPDATE harness_rag.index_jobs SET state='ready',readiness=%s,updated_at=clock_timestamp() "
                "WHERE job_id=%s RETURNING *", (Jsonb(proof.model_dump(mode="json")), job_id),
            )).fetchone()
            await self._audit(conn, actor, job.knowledge_base_id, "index_ready", job_id,
                              {"proof_hash": canonical_hash(proof.model_dump(mode="json"))})
            if updated is None:
                raise RuntimeFault("INDEX_READY_FAILED")
            return index_job(updated)

    @staticmethod
    async def _deadline(conn: Connection, row: Row) -> None:
        clock = await (await conn.execute("SELECT clock_timestamp() AS now")).fetchone()
        if clock is None or clock["now"] >= RetrievalScope.model_validate(row["scope"]).deadline:
            raise RuntimeFault("INDEX_BUILD_DEADLINE_EXCEEDED", 409)

    async def fail(self, actor: Principal, job_id: str, code: str) -> None:
        async with self.uow.transaction() as conn:
            # 调用者排空有deadline的RPC后收尾；撤权不能使生产者永远卡在building。
            row = await self._job(conn, actor, job_id, require_versions=False)
            if row["state"] != "building":
                return
            await conn.execute("UPDATE harness_rag.index_jobs SET state='failed',error_code=%s,"
                               "updated_at=clock_timestamp() WHERE job_id=%s", (code, job_id))
            await self._audit(conn, actor, row["knowledge_base_id"], "index_failed", job_id, {"code": code})

    async def publish(self, actor: Principal, job_id: str) -> KnowledgeSnapshot:
        async with self.uow.transaction() as conn:
            row = await self._job(conn, actor, job_id, owner_only=False)
            job = index_job(row)
            if job.state != "ready":
                raise RuntimeFault("INDEX_NOT_READY", 409)
            members = await self._members(conn, job.version_ids)
            if member_hash(members) != job.members_hash:
                raise RuntimeFault("INDEX_MEMBERS_CHANGED", 409)
            content_hash = canonical_hash({"generation_id": job.generation_id,
                                           "binding": job.binding.content_hash(), "members": job.members_hash})
            snapshot_id = "snapshot-" + job.generation_id
            result = await (await conn.execute(
                "INSERT INTO harness_rag.knowledge_snapshots(snapshot_id,tenant_id,application_id,"
                "knowledge_base_id,generation_id,content_hash,state) VALUES(%s,%s,%s,%s,%s,%s,'published') "
                "ON CONFLICT DO NOTHING RETURNING *",
                (snapshot_id, actor.tenant_id, actor.application_id, job.knowledge_base_id,
                 job.generation_id, content_hash),
            )).fetchone()
            if result is None:
                result = await (await conn.execute("SELECT * FROM harness_rag.knowledge_snapshots "
                                                   "WHERE snapshot_id=%s", (snapshot_id,))).fetchone()
                if result is None or result["state"] != "published" or result["content_hash"] != content_hash:
                    raise RuntimeFault("SNAPSHOT_PUBLICATION_CONFLICT", 409)
                return snapshot_view(result)
            await conn.execute(
                "INSERT INTO harness_rag.snapshot_members SELECT %s,chunk_id,version_id,body_hash "
                "FROM harness_rag.generation_members WHERE generation_id=%s", (snapshot_id, job.generation_id),
            )
            await self._audit(conn, actor, job.knowledge_base_id, "snapshot_published", snapshot_id,
                              {"content_hash": content_hash})
            return snapshot_view(result)

    async def _snapshot(self, conn: Connection, actor: Principal, snapshot_id: str,
                        *, require_versions: bool = True) -> tuple[Row, Row]:
        hint = await (await conn.execute(
            "SELECT j.job_id FROM harness_rag.knowledge_snapshots s JOIN harness_rag.index_jobs j "
            "ON j.generation_id=s.generation_id WHERE s.snapshot_id=%s AND s.tenant_id=%s AND s.application_id=%s",
            (snapshot_id, actor.tenant_id, actor.application_id),
        )).fetchone()
        if hint is None:
            raise RuntimeFault("SNAPSHOT_NOT_FOUND", 404)
        job = await self._job(conn, actor, hint["job_id"], owner_only=False, require_versions=require_versions)
        row = await (await conn.execute("SELECT * FROM harness_rag.knowledge_snapshots "
                                       "WHERE snapshot_id=%s FOR UPDATE", (snapshot_id,))).fetchone()
        if row is None:
            raise RuntimeFault("SNAPSHOT_NOT_FOUND", 404)
        return row, job

    async def activate(self, actor: Principal, snapshot_id: str, expected_snapshot: str | None,
                       expected_revision: int) -> int:
        async with self.uow.transaction() as conn:
            snapshot, job = await self._snapshot(conn, actor, snapshot_id)
            if snapshot["state"] != "published" or job["state"] != "ready":
                raise RuntimeFault("SNAPSHOT_NOT_READY", 409)
            kb_id = job["knowledge_base_id"]
            await conn.execute("INSERT INTO harness_rag.active_snapshots(tenant_id,application_id,"
                               "knowledge_base_id) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                               (actor.tenant_id, actor.application_id, kb_id))
            updated = await (await conn.execute(
                "UPDATE harness_rag.active_snapshots SET snapshot_id=%s,revision=revision+1 "
                "WHERE tenant_id=%s AND application_id=%s AND knowledge_base_id=%s AND revision=%s "
                "AND snapshot_id IS NOT DISTINCT FROM %s RETURNING revision",
                (snapshot_id, actor.tenant_id, actor.application_id, kb_id, expected_revision, expected_snapshot),
            )).fetchone()
            if updated is None:
                raise RuntimeFault("SNAPSHOT_ACTIVE_CONFLICT", 409)
            await self._audit(conn, actor, kb_id, "snapshot_activated", snapshot_id,
                              {"previous": expected_snapshot, "revision": updated["revision"]})
            return int(updated["revision"])

    async def deactivate(self, actor: Principal, kb_id: str, expected_snapshot: str,
                         expected_revision: int) -> int:
        async with self.uow.transaction() as conn:
            await self.permissions.authorize_in_transaction(conn, actor, kb_id)
            updated = await (await conn.execute(
                "UPDATE harness_rag.active_snapshots SET snapshot_id=NULL,revision=revision+1 "
                "WHERE tenant_id=%s AND application_id=%s AND knowledge_base_id=%s "
                "AND snapshot_id=%s AND revision=%s RETURNING revision",
                (actor.tenant_id, actor.application_id, kb_id, expected_snapshot, expected_revision),
            )).fetchone()
            if updated is None:
                raise RuntimeFault("SNAPSHOT_ACTIVE_CONFLICT", 409)
            await self._audit(conn, actor, kb_id, "snapshot_deactivated", expected_snapshot,
                              {"revision": updated["revision"]})
            return int(updated["revision"])

    async def offline(self, actor: Principal, snapshot_id: str) -> None:
        async with self.uow.transaction() as conn:
            snapshot, job = await self._snapshot(conn, actor, snapshot_id, require_versions=False)
            active = await (await conn.execute("SELECT snapshot_id FROM harness_rag.active_snapshots "
                                               "WHERE snapshot_id=%s FOR UPDATE", (snapshot_id,))).fetchone()
            if active is not None:
                raise RuntimeFault("SNAPSHOT_STILL_ACTIVE", 409)
            if snapshot["state"] == "offline":
                return
            await conn.execute("UPDATE harness_rag.knowledge_snapshots SET state='offline' "
                               "WHERE snapshot_id=%s", (snapshot_id,))
            await conn.execute("UPDATE harness_rag.index_jobs SET state='offline',updated_at=clock_timestamp() "
                               "WHERE job_id=%s", (job["job_id"],))
            await self._audit(conn, actor, job["knowledge_base_id"], "snapshot_offline", snapshot_id, {})

    async def active(self, actor: Principal, kb_id: str) -> dict[str, object]:
        async with self.uow.transaction() as conn:
            await self.permissions.authorize_in_transaction(conn, actor, kb_id)
            row = await (await conn.execute("SELECT snapshot_id,revision FROM harness_rag.active_snapshots "
                                           "WHERE tenant_id=%s AND application_id=%s AND knowledge_base_id=%s",
                                           (actor.tenant_id, actor.application_id, kb_id))).fetchone()
            return {"snapshot_id": row["snapshot_id"], "revision": row["revision"]} if row else {
                "snapshot_id": None, "revision": 0,
            }
