"""正式入库PG权威状态；KB→文档→job锁顺序，不在事务中解析或写文件。"""

from collections.abc import Callable
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, PostgresUnitOfWork, Row
from harness.contracts.hashing import canonical_hash
from harness.contracts.ingestion import Chunk, IngestionJob, IngestionProfile, SourceRef, Upload
from harness.contracts.runtime import Principal, RuntimeFault


def job_view(row: Row) -> IngestionJob:
    return IngestionJob.model_validate({key: row[key] for key in IngestionJob.model_fields})


class PostgresIngestionRepository:
    def __init__(self, dsn: str, current_identity: Callable[[Principal], Principal]) -> None:
        self.uow = PostgresUnitOfWork(dsn)
        self.current_identity = current_identity

    @staticmethod
    def _role(principal: Principal, kb_id: str) -> None:
        if ("knowledge_manager" not in principal.roles
            or f"kb:{kb_id}" not in principal.resource_scopes):
            raise RuntimeFault("KNOWLEDGE_ACCESS_DENIED", 403)

    async def _authorize(self, conn: Connection, principal: Principal, kb_id: str) -> None:
        self._role(self.current_identity(principal), kb_id)
        row = await (await conn.execute(
            "SELECT k.revoked_at FROM harness_rag.knowledge_bases k "
            "JOIN harness_rag.knowledge_grants g USING(tenant_id,application_id,knowledge_base_id) "
            "WHERE k.tenant_id=%s AND k.application_id=%s AND k.knowledge_base_id=%s "
            "AND g.subject_id=%s FOR UPDATE OF k,g",
            (principal.tenant_id, principal.application_id, kb_id, principal.subject_id),
        )).fetchone()
        if row is None or row["revoked_at"] is not None:
            raise RuntimeFault("KNOWLEDGE_ACCESS_DENIED", 403)

    async def authorize_in_transaction(self, conn: Connection, principal: Principal,
                                       kb_id: str) -> None:
        """供可选PG扩展复用同一事务的权威KB授权，不在核心暴露SQL。"""
        await self._authorize(conn, principal, kb_id)

    async def authorize_document_in_transaction(self, conn: Connection, principal: Principal,
                                                kb_id: str, document_id: str) -> None:
        """调用者先锁KB，再按稳定顺序锁文档。"""
        await self._document(conn, principal, kb_id, document_id)

    async def create_knowledge_base(self, principal: Principal, kb_id: str) -> None:
        self._role(self.current_identity(principal), kb_id)
        async with self.uow.transaction() as conn:
            created = await conn.execute(
                "INSERT INTO harness_rag.knowledge_bases(tenant_id,application_id,knowledge_base_id,"
                "created_by) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (principal.tenant_id, principal.application_id, kb_id, principal.subject_id),
            )
            if created.rowcount == 1:
                await conn.execute(
                    "INSERT INTO harness_rag.knowledge_grants VALUES(%s,%s,%s,%s)",
                    (principal.tenant_id, principal.application_id, kb_id, principal.subject_id),
                )
            await self._authorize(conn, principal, kb_id)

    async def _document(self, conn: Connection, principal: Principal, kb_id: str,
                        document_id: str) -> None:
        row = await (await conn.execute(
            "SELECT revoked_at FROM harness_rag.documents WHERE tenant_id=%s AND application_id=%s "
            "AND knowledge_base_id=%s AND document_id=%s FOR UPDATE",
            (principal.tenant_id, principal.application_id, kb_id, document_id),
        )).fetchone()
        if row is None or row["revoked_at"] is not None:
            raise RuntimeFault("KNOWLEDGE_DOCUMENT_ACCESS_DENIED", 403)

    async def _job(self, conn: Connection, principal: Principal, job_id: str,
                   *, allow_revoked_document: bool = False) -> Row:
        # 无锁读取仅用于确定锁定次序，拿齐锁后重读；不交付未授权内容。
        hint = await (await conn.execute(
            "SELECT knowledge_base_id,document_id FROM harness_rag.ingestion_jobs "
            "WHERE job_id=%s AND tenant_id=%s AND application_id=%s AND subject_id=%s",
            (job_id, principal.tenant_id, principal.application_id, principal.subject_id),
        )).fetchone()
        if hint is None:
            raise RuntimeFault("INGESTION_JOB_NOT_FOUND", 404)
        await self._authorize(conn, principal, hint["knowledge_base_id"])
        if not allow_revoked_document:
            await self._document(conn, principal, hint["knowledge_base_id"], hint["document_id"])
        row = await (await conn.execute(
            "SELECT * FROM harness_rag.ingestion_jobs WHERE job_id=%s FOR UPDATE", (job_id,),
        )).fetchone()
        if row is None:
            raise RuntimeFault("INGESTION_JOB_NOT_FOUND", 404)
        return row

    async def begin(self, principal: Principal, kb_id: str, upload: Upload,
                    source: SourceRef, profile: IngestionProfile) -> IngestionJob:
        fingerprint = canonical_hash({"upload": upload.model_dump(mode="json"),
                                      "source": source.model_dump(mode="json"),
                                      "profile": profile.model_dump(mode="json")})
        async with self.uow.transaction() as conn:
            await self._authorize(conn, principal, kb_id)
            lifecycle = await (await conn.execute("SELECT to_regclass('harness_rag.cleanup_steps') AS table_ref")).fetchone()
            if lifecycle["table_ref"] is not None:
                deleting = await (await conn.execute("SELECT 1 FROM harness_rag.cleanup_steps s "
                    "JOIN harness_rag.cleanup_jobs j USING(job_id) WHERE j.tenant_id=%s AND j.application_id=%s "
                    "AND j.knowledge_base_id=%s AND s.kind='source' AND s.state='running' "
                    "AND s.target->'source'->>'object_id'=%s LIMIT 1",
                    (principal.tenant_id, principal.application_id, kb_id, source.object_id))).fetchone()
                if deleting:
                    raise RuntimeFault("INGESTION_SOURCE_CLEANUP_IN_PROGRESS", 409)
            old = await (await conn.execute(
                "SELECT * FROM harness_rag.ingestion_jobs WHERE tenant_id=%s AND application_id=%s "
                "AND knowledge_base_id=%s AND request_id=%s",
                (principal.tenant_id, principal.application_id, kb_id, upload.request_id),
            )).fetchone()
            if old is not None:
                if old["subject_id"] != principal.subject_id or old["fingerprint"] != fingerprint:
                    raise RuntimeFault("INGESTION_REQUEST_CONFLICT", 409)
                await self._document(conn, principal, kb_id, old["document_id"])
                return job_view(old)
            await conn.execute(
                "INSERT INTO harness_rag.documents(tenant_id,application_id,knowledge_base_id,"
                "document_id) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (principal.tenant_id, principal.application_id, kb_id, upload.document_id),
            )
            await self._document(conn, principal, kb_id, upload.document_id)
            row = await (await conn.execute(
                "INSERT INTO harness_rag.ingestion_jobs(job_id,tenant_id,application_id,knowledge_base_id,"
                "document_id,subject_id,request_id,fingerprint,version_id,source,upload,profile,state) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued') RETURNING *",
                (uuid4().hex, principal.tenant_id, principal.application_id, kb_id, upload.document_id,
                 principal.subject_id, upload.request_id, fingerprint, uuid4().hex,
                 Jsonb(source.model_dump(mode="json")), Jsonb(upload.model_dump(mode="json")),
                 Jsonb(profile.model_dump(mode="json"))),
            )).fetchone()
            if row is None:
                raise RuntimeFault("INGESTION_CREATE_FAILED")
            return job_view(row)

    async def read(self, principal: Principal, job_id: str) -> IngestionJob:
        async with self.uow.transaction() as conn:
            return job_view(await self._job(conn, principal, job_id))

    async def advance(self, principal: Principal, job_id: str, expected: str,
                      target: str) -> None:
        if (expected, target) not in {("queued", "storing"), ("storing", "parsing"),
                                      ("parsing", "persisting")}:
            raise RuntimeFault("INGESTION_INVALID_TRANSITION", 409)
        async with self.uow.transaction() as conn:
            row = await self._job(conn, principal, job_id)
            if row["state"] != expected:
                raise RuntimeFault("INGESTION_JOB_BUSY", 409)
            await conn.execute(
                "UPDATE harness_rag.ingestion_jobs SET state=%s,updated_at=clock_timestamp() "
                "WHERE job_id=%s", (target, job_id),
            )

    async def finish(self, principal: Principal, job_id: str,
                     chunks: tuple[Chunk, ...]) -> IngestionJob:
        # 适配器边界重新校验，不能信任第三方parser返回的hash/offset或model_copy。
        chunks = tuple(Chunk.model_validate_json(c.model_dump_json()) for c in chunks)
        async with self.uow.transaction() as conn:
            row = await self._job(conn, principal, job_id)
            profile = IngestionProfile.model_validate(row["profile"])
            if row["state"] != "persisting":
                raise RuntimeFault("INGESTION_INVALID_TRANSITION", 409)
            cursor = 0
            if not chunks or len(chunks) > profile.max_chunks:
                raise RuntimeFault("INGESTION_CHUNKS_INVALID", 422)
            for ordinal, chunk in enumerate(chunks):
                if (chunk.ordinal != ordinal or chunk.start != cursor
                    or chunk.end != cursor + len(chunk.body)
                    or len(chunk.body) > profile.chunk_characters
                    or chunk.body_hash != canonical_hash(chunk.body)):
                    raise RuntimeFault("INGESTION_CHUNKS_INVALID", 422)
                cursor = chunk.end
            if cursor > profile.max_characters:
                raise RuntimeFault("INGESTION_CHUNKS_INVALID", 422)
            await conn.execute(
                "INSERT INTO harness_rag.document_versions(version_id,job_id,source,parser_profile,"
                "content_hash,chunk_count) VALUES(%s,%s,%s,%s,%s,%s)",
                (row["version_id"], job_id, Jsonb(row["source"]), Jsonb(row["profile"]),
                 canonical_hash("".join(c.body for c in chunks)), len(chunks)),
            )
            for chunk in chunks:
                await conn.execute(
                    "INSERT INTO harness_rag.chunks(chunk_id,version_id,ordinal,start_offset,"
                    "end_offset,body,body_hash) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                    (canonical_hash({"version": row["version_id"], "ordinal": chunk.ordinal}),
                     row["version_id"], chunk.ordinal, chunk.start, chunk.end, chunk.body, chunk.body_hash),
                )
            updated = await (await conn.execute(
                "UPDATE harness_rag.ingestion_jobs SET state='succeeded',chunks_written=%s,"
                "updated_at=clock_timestamp() WHERE job_id=%s RETURNING *", (len(chunks), job_id),
            )).fetchone()
            if updated is None:
                raise RuntimeFault("INGESTION_COMMIT_FAILED")
            return job_view(updated)

    async def fail(self, principal: Principal, job_id: str, code: str) -> None:
        async with self.uow.transaction() as conn:
            # 已排空的失败生产者仍可在文档撤权后落账，不能因此恢复正文权限。
            row = await self._job(conn, principal, job_id, allow_revoked_document=True)
            if row["state"] in {"failed", "succeeded"}:
                return
            await conn.execute(
                "UPDATE harness_rag.ingestion_jobs SET state='failed',error_code=%s,"
                "updated_at=clock_timestamp() WHERE job_id=%s", (code, job_id),
            )
