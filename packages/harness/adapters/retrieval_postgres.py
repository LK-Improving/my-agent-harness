"""固定run快照的PG检索权威层；每次回表与交付重查身份、ACL和撤权。"""

import asyncio
from collections.abc import Callable
from datetime import timedelta
from uuid import uuid4

from pydantic import JsonValue
from psycopg.types.json import Jsonb

from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.postgres.connection import Connection, PostgresUnitOfWork, Row
from harness.contracts.execution import ExecutionProfile
from harness.contracts.evaluation_gates import RetrievalChildEvidence, RetrievalModelCallEvidence
from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import IndexBinding
from harness.contracts.release import ReleaseManifest
from harness.contracts.retrieval import (
    Evidence, RetrievalQuery, RetrievalRelease, RetrievalReservation, RetrievalResult, RetrievalTicket, SnapshotPin,
)
from harness.contracts.retrieval_models import EmbeddingRequest, RerankRequest, RetrievalPayload, RetrievalScope
from harness.contracts.runtime import Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


class PostgresRetrievalRepository:
    def __init__(self, dsn: str, execution: PostgresExecutionStore,
                 current_identity: Callable[[Principal], Principal]) -> None:
        self.uow, self.execution, self.current_identity = PostgresUnitOfWork(dsn), execution, current_identity

    async def _kb(self, conn: Connection, actor: Principal, kb_id: str) -> None:
        current = await asyncio.to_thread(self.current_identity, actor)
        if (not current.roles & {"resource_reader", "knowledge_manager"}
            or f"kb:{kb_id}" not in current.resource_scopes):
            raise RuntimeFault("RETRIEVAL_ACCESS_DENIED", 403)
        row = await (await conn.execute(
            "SELECT revoked_at FROM harness_rag.knowledge_bases k WHERE tenant_id=%s AND application_id=%s "
            "AND knowledge_base_id=%s AND (EXISTS(SELECT 1 FROM harness_rag.knowledge_read_grants g "
            "WHERE (g.tenant_id,g.application_id,g.knowledge_base_id)=(k.tenant_id,k.application_id,k.knowledge_base_id) "
            "AND g.subject_id=%s) OR EXISTS(SELECT 1 FROM harness_rag.knowledge_grants g "
            "WHERE (g.tenant_id,g.application_id,g.knowledge_base_id)=(k.tenant_id,k.application_id,k.knowledge_base_id) "
            "AND g.subject_id=%s)) FOR SHARE OF k",
            (actor.tenant_id, actor.application_id, kb_id, actor.subject_id, actor.subject_id),
        )).fetchone()
        if row is None or row["revoked_at"] is not None:
            raise RuntimeFault("RETRIEVAL_ACCESS_DENIED", 403)

    async def _pin(self, conn: Connection, actor: Principal, pin: SnapshotPin) -> IndexBinding:
        row = await (await conn.execute(
            "SELECT s.content_hash,s.state,j.generation_id,j.binding,j.state AS generation_state "
            "FROM harness_rag.knowledge_snapshots s JOIN harness_rag.index_jobs j USING(generation_id) "
            "WHERE s.snapshot_id=%s AND s.tenant_id=%s AND s.application_id=%s AND s.knowledge_base_id=%s "
            "FOR SHARE OF s,j", (pin.snapshot_id, actor.tenant_id, actor.application_id, pin.knowledge_base_id),
        )).fetchone()
        if row is None or row["state"] != "published" or row["generation_state"] != "ready":
            raise RuntimeFault("RETRIEVAL_SNAPSHOT_UNAVAILABLE", 409)
        binding = IndexBinding.model_validate(row["binding"])
        if (row["content_hash"] != pin.snapshot_hash or row["generation_id"] != pin.generation_id
            or binding.content_hash() != pin.index_binding_hash):
            raise RuntimeFault("RETRIEVAL_SNAPSHOT_BINDING_MISMATCH", 409)
        return binding

    async def _run(self, conn: Connection, actor: Principal, owner: str, run_id: str,
                   kb_id: str, configuration: RetrievalRelease | None = None) -> tuple[Row, ReleaseManifest, RetrievalRelease, SnapshotPin, IndexBinding]:
        run, manifest = await self.execution.authorize_extension_in_transaction(conn, actor, owner, run_id)
        installed = manifest.profiles.as_object().get("retrieval")
        if not isinstance(installed, dict):
            raise RuntimeFault("RETRIEVAL_RELEASE_NOT_INSTALLED", 409)
        release = RetrievalRelease.model_validate(installed)
        if configuration is not None and release != configuration:
            raise RuntimeFault("RETRIEVAL_DEPLOYMENT_MISMATCH", 409)
        bindings = [b for b in manifest.spec.knowledge_bindings if b.knowledge_base_id == kb_id]
        pins = [p for p in release.snapshots if p.knowledge_base_id == kb_id]
        if len(bindings) != 1 or len(pins) != 1 or pins[0].snapshot_id != bindings[0].snapshot_id:
            raise RuntimeFault("RETRIEVAL_KNOWLEDGE_BINDING_DENIED", 403)
        profile = next((p for p in release.profiles if p.ref == bindings[0].retrieval_profile_ref), None)
        if profile is None:
            raise RuntimeFault("RETRIEVAL_PROFILE_NOT_INSTALLED", 409)
        await self._kb(conn, actor, kb_id)
        binding = await self._pin(conn, actor, pins[0])
        if binding.model_snapshot != release.models or binding.backend_hash != release.backend_hash:
            raise RuntimeFault("RETRIEVAL_INDEX_MODEL_MISMATCH", 409)
        return run, manifest, release, pins[0], binding

    async def _locked(self, conn: Connection, ticket: RetrievalTicket, *,
                      current_owner: str | None = None, require_deadline: bool = True) -> Row:
        run, manifest, release, pin, binding = await self._run(
            conn, ticket.actor, current_owner or ticket.owner, ticket.scope.owner_id, ticket.pin.knowledge_base_id)
        row = await (await conn.execute("SELECT * FROM harness_rag.retrieval_requests "
                                       "WHERE retrieval_id=%s FOR UPDATE", (ticket.retrieval_id,))).fetchone()
        if (row is None or RetrievalTicket.model_validate(row["ticket"]) != ticket or pin != ticket.pin
            or binding != ticket.index_binding or manifest.content_hash() != ticket.scope.manifest_hash
            or ticket.profile not in release.profiles):
            raise RuntimeFault("RETRIEVAL_TICKET_MISMATCH", 409)
        if require_deadline and run["now"] >= ticket.scope.deadline:
            raise RuntimeFault("RETRIEVAL_DEADLINE_EXCEEDED", 409)
        return row

    async def _stage(self, conn: Connection, ticket: RetrievalTicket, name: str, details: JsonSnapshot) -> None:
        if len(details.canonical_text.encode("utf-8")) > 4096:
            raise RuntimeFault("RETRIEVAL_AUDIT_LIMIT_EXCEEDED", 413)
        await conn.execute("INSERT INTO harness_rag.retrieval_stages(retrieval_id,stage,details) VALUES(%s,%s,%s)",
                           (ticket.retrieval_id, name, Jsonb(details.materialize())))

    async def open(self, actor: Principal, owner: str, run_id: str, kb_id: str,
                   query: RetrievalQuery, configuration: RetrievalRelease) -> RetrievalReservation:
        query = RetrievalQuery.model_validate_json(query.model_dump_json())
        async with self.uow.transaction() as conn:
            run, manifest, release, pin, binding = await self._run(conn, actor, owner, run_id, kb_id, configuration)
            knowledge = next(b for b in manifest.spec.knowledge_bindings if b.knowledge_base_id == kb_id)
            profile = next(p for p in release.profiles if p.ref == knowledge.retrieval_profile_ref)
            if not query.query.strip() or len(query.query) > profile.query_characters:
                raise RuntimeFault("RETRIEVAL_QUERY_LIMIT_EXCEEDED", 422)
            fingerprint = canonical_hash({"query": query.model_dump(mode="json"), "release": release.model_dump(mode="json")})
            old = await (await conn.execute("SELECT * FROM harness_rag.retrieval_requests WHERE run_id=%s "
                                           "AND knowledge_base_id=%s FOR UPDATE", (run_id, kb_id))).fetchone()
            if old is not None:
                if old["fingerprint"] != fingerprint:
                    raise RuntimeFault("RETRIEVAL_RUN_QUERY_CONFLICT", 409)
                ticket = RetrievalTicket.model_validate(old["ticket"])
                await self._locked(conn, ticket)
                return RetrievalReservation(ticket=ticket, state=old["state"], result=(
                    RetrievalResult.model_validate(old["result"]) if old["result"] else None))
            execution_profile = ExecutionProfile.model_validate(manifest.profiles.as_object()["execution"])
            reserved = profile.parent_token_reservation()
            if run["model_reserved_tokens"] + reserved > execution_profile.budget.max_total_tokens:
                raise RuntimeFault("RETRIEVAL_PARENT_BUDGET_EXCEEDED", 429)
            retrieval_id = uuid4().hex
            ticket = RetrievalTicket(
                retrieval_id=retrieval_id, actor=actor, owner=owner, pin=pin, index_binding=binding,
                query=query, profile=profile, scope=RetrievalScope(
                    context_id=retrieval_id, tenant_id=actor.tenant_id, subject_id=actor.subject_id,
                    application_id=actor.application_id, owner_kind="run", owner_id=run_id,
                    knowledge_base_id=kb_id, manifest_hash=manifest.content_hash(),
                    embedding_binding_hash=binding.embedding_binding_hash, rerank_binding_hash=binding.rerank_binding_hash,
                    deadline=min(run["execution_deadline"], run["expires_at"],
                                 run["now"] + timedelta(seconds=profile.timeout_seconds)),
                ),
            )
            await conn.execute("UPDATE harness.runs SET model_reserved_tokens=model_reserved_tokens+%s "
                               "WHERE run_id=%s", (reserved, run_id))
            await conn.execute(
                "INSERT INTO harness_rag.retrieval_requests(retrieval_id,run_id,knowledge_base_id,tenant_id,application_id,"
                "subject_id,request_id,fingerprint,ticket,parent_reserved_tokens,state) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued')",
                (retrieval_id, run_id, kb_id, actor.tenant_id, actor.application_id, actor.subject_id,
                 query.request_id, fingerprint, Jsonb(ticket.model_dump(mode="json")), reserved),
            )
            await self._stage(conn, ticket, "created", JsonSnapshot.capture({"snapshot_hash": pin.snapshot_hash,
                                                                          "parent_reserved_tokens": reserved}))
            return RetrievalReservation(ticket=ticket, state="queued")

    async def claim(self, ticket: RetrievalTicket) -> None:
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if row["state"] != "queued":
                raise RuntimeFault("RETRIEVAL_REQUEST_BUSY", 409)
            await conn.execute("UPDATE harness_rag.retrieval_requests SET state='running',updated_at=clock_timestamp() "
                               "WHERE retrieval_id=%s", (ticket.retrieval_id,))
            await self._stage(conn, ticket, "claimed", JsonSnapshot.capture({}))

    async def check(self, ticket: RetrievalTicket) -> None:
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if row["state"] not in {"running", "succeeded"}:
                raise RuntimeFault("RETRIEVAL_REQUEST_INACTIVE", 409)

    async def _hydrate(self, conn: Connection, ticket: RetrievalTicket, ids: tuple[str, ...]) -> tuple[Evidence, ...]:
        if len(ids) > ticket.profile.fusion_limit or len(set(ids)) != len(ids):
            raise RuntimeFault("RETRIEVAL_CANDIDATES_INVALID", 422)
        actor, kb_id = ticket.actor, ticket.pin.knowledge_base_id
        rows = await (await conn.execute(
            "SELECT c.chunk_id,c.version_id,j.document_id,c.body_hash,c.body,m.body_hash AS member_hash "
            "FROM harness_rag.snapshot_members m JOIN harness_rag.chunks c "
            "ON (c.chunk_id,c.version_id)=(m.chunk_id,m.version_id) "
            "JOIN harness_rag.document_versions v ON v.version_id=c.version_id "
            "JOIN harness_rag.ingestion_jobs j ON j.job_id=v.job_id "
            "JOIN harness_rag.documents d ON (d.tenant_id,d.application_id,d.knowledge_base_id,d.document_id)="
            "(j.tenant_id,j.application_id,j.knowledge_base_id,j.document_id) "
            "LEFT JOIN harness_rag.document_access a ON (a.tenant_id,a.application_id,a.knowledge_base_id,a.document_id)="
            "(d.tenant_id,d.application_id,d.knowledge_base_id,d.document_id) "
            "WHERE m.snapshot_id=%s AND c.chunk_id=ANY(%s) AND d.tenant_id=%s AND d.application_id=%s "
            "AND d.knowledge_base_id=%s AND d.revoked_at IS NULL AND j.state='succeeded' "
            "AND (NOT coalesce(a.restricted,false) OR EXISTS(SELECT 1 FROM harness_rag.document_read_grants g "
            "WHERE (g.tenant_id,g.application_id,g.knowledge_base_id,g.document_id)="
            "(d.tenant_id,d.application_id,d.knowledge_base_id,d.document_id) AND g.subject_id=%s))",
            (ticket.pin.snapshot_id, list(ids), actor.tenant_id, actor.application_id, kb_id, actor.subject_id),
        )).fetchall()
        found = {}
        for row in rows:
            if row["body_hash"] != row["member_hash"] or canonical_hash(row["body"]) != row["body_hash"]:
                raise RuntimeFault("RETRIEVAL_AUTHORITY_HASH_MISMATCH", 409)
            found[row["chunk_id"]] = Evidence(
                chunk_id=row["chunk_id"], version_id=row["version_id"], document_id=row["document_id"],
                snapshot_id=ticket.pin.snapshot_id, generation_id=ticket.pin.generation_id,
                body_hash=row["body_hash"], body=row["body"],
            )
        return tuple(found[chunk_id] for chunk_id in ids if chunk_id in found)

    async def hydrate(self, ticket: RetrievalTicket, ids: tuple[str, ...]) -> tuple[Evidence, ...]:
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if row["state"] not in {"running", "succeeded"}:
                raise RuntimeFault("RETRIEVAL_REQUEST_INACTIVE", 409)
            return await self._hydrate(conn, ticket, ids)

    async def authorize(self, scope: RetrievalScope, payload: RetrievalPayload) -> None:
        async with self.uow.transaction() as conn:
            hint = await (await conn.execute("SELECT ticket FROM harness_rag.retrieval_requests WHERE retrieval_id=%s",
                                            (scope.context_id,))).fetchone()
            if hint is None:
                raise RuntimeFault("RETRIEVAL_MODEL_OWNER_NOT_FOUND", 403)
            ticket = RetrievalTicket.model_validate(hint["ticket"])
            if ticket.scope != scope:
                raise RuntimeFault("RETRIEVAL_MODEL_SCOPE_MISMATCH", 403)
            row = await self._locked(conn, ticket)
            if row["state"] != "running":
                raise RuntimeFault("RETRIEVAL_MODEL_OWNER_INACTIVE", 409)
            if isinstance(payload, EmbeddingRequest):
                if payload.texts != (ticket.query.query,):
                    raise RuntimeFault("RETRIEVAL_MODEL_QUERY_MISMATCH", 403)
            elif isinstance(payload, RerankRequest):
                ids = tuple(e.chunk_id for e in payload.evidence)
                actual = await self._hydrate(conn, ticket, ids)
                if (payload.query != ticket.query.query or len(actual) != len(ids)
                    or tuple(e.body for e in actual) != payload.documents
                    or any((e.version_id, e.body_hash) != (ref.version_id, ref.body_hash)
                           for e, ref in zip(actual, payload.evidence, strict=True))):
                    raise RuntimeFault("RETRIEVAL_MODEL_EVIDENCE_DENIED", 403)

    async def stage(self, ticket: RetrievalTicket, name: str, data: JsonSnapshot) -> None:
        if name not in {"recall", "evidence"}:
            raise RuntimeFault("RETRIEVAL_STAGE_INVALID", 422)
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if row["state"] != "running":
                raise RuntimeFault("RETRIEVAL_REQUEST_INACTIVE", 409)
            await self._stage(conn, ticket, name, data)

    async def _validate_result(self, conn: Connection, ticket: RetrievalTicket, result: RetrievalResult) -> None:
        if result.retrieval_id != ticket.retrieval_id or len(result.evidence) > ticket.profile.top_n:
            raise RuntimeFault("RETRIEVAL_RESULT_BINDING_MISMATCH", 409)
        actual = await self._hydrate(conn, ticket, tuple(e.chunk_id for e in result.evidence))
        if actual != result.evidence or sum(len(e.body) for e in actual) > ticket.profile.evidence_characters:
            raise RuntimeFault("RETRIEVAL_DELIVERY_AUTHORITY_CHANGED", 403)

    async def finish(self, ticket: RetrievalTicket, result: RetrievalResult) -> None:
        result = RetrievalResult.model_validate_json(result.model_dump_json())
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if row["state"] != "running":
                raise RuntimeFault("RETRIEVAL_REQUEST_INACTIVE", 409)
            await self._validate_result(conn, ticket, result)
            stages = await (await conn.execute("SELECT stage FROM harness_rag.retrieval_stages "
                                              "WHERE retrieval_id=%s", (ticket.retrieval_id,))).fetchall()
            if {r["stage"] for r in stages} != {"created", "claimed", "recall", "evidence"}:
                raise RuntimeFault("RETRIEVAL_STAGE_RECEIPTS_INCOMPLETE", 409)
            await conn.execute("UPDATE harness_rag.retrieval_requests SET state='succeeded',result=%s,result_hash=%s,"
                               "updated_at=clock_timestamp() WHERE retrieval_id=%s",
                               (Jsonb(result.model_dump(mode="json")), canonical_hash(result.model_dump(mode="json")),
                                ticket.retrieval_id))
            await self._stage(conn, ticket, "completed", JsonSnapshot.capture({"outcome": result.outcome,
                                                                             "evidence_count": len(result.evidence)}))

    async def deliver(self, ticket: RetrievalTicket, result: RetrievalResult) -> RetrievalResult:
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if (row["state"] != "succeeded" or row["result_hash"] != canonical_hash(result.model_dump(mode="json"))
                or row["result"] != result.model_dump(mode="json")):
                raise RuntimeFault("RETRIEVAL_RESULT_BINDING_MISMATCH", 409)
            await self._validate_result(conn, ticket, result)
            return result

    async def fail(self, ticket: RetrievalTicket, code: str) -> None:
        async with self.uow.transaction() as conn:
            row = await self._locked(conn, ticket)
            if row["state"] != "running":
                return
            await conn.execute("UPDATE harness_rag.retrieval_requests SET state='failed',error_code=%s,"
                               "updated_at=clock_timestamp() WHERE retrieval_id=%s", (code, ticket.retrieval_id))
            await self._stage(conn, ticket, "failed", JsonSnapshot.capture({"code": code}))

    async def authorize_result_in_transaction(self, conn: Connection, actor: Principal, owner: str,
                                               run_id: str, result: RetrievalResult) -> RetrievalResult:
        """只复用已成功结果；新合法owner可重查证据，不接管调用、不改scope/期限或额度。"""
        hint = await (await conn.execute("SELECT ticket FROM harness_rag.retrieval_requests WHERE retrieval_id=%s "
                                        "AND run_id=%s AND tenant_id=%s AND application_id=%s AND subject_id=%s",
                                        (result.retrieval_id, run_id, actor.tenant_id, actor.application_id,
                                         actor.subject_id))).fetchone()
        if hint is None:
            raise RuntimeFault("KNOWLEDGE_RESULT_NOT_FOUND", 404)
        ticket = RetrievalTicket.model_validate(hint["ticket"])
        if ticket.actor != actor or ticket.scope.owner_id != run_id:
            raise RuntimeFault("KNOWLEDGE_RESULT_SCOPE_MISMATCH", 403)
        row = await self._locked(conn, ticket, current_owner=owner, require_deadline=False)
        if (row["state"] != "succeeded" or row["result"] != result.model_dump(mode="json")
            or row["result_hash"] != canonical_hash(result.model_dump(mode="json"))):
            raise RuntimeFault("KNOWLEDGE_RESULT_RECEIPT_MISMATCH", 409)
        await self._validate_result(conn, ticket, result)
        return result

    async def authorize_result(self, actor: Principal, owner: str, run_id: str,
                               result: RetrievalResult) -> RetrievalResult:
        async with self.uow.transaction() as conn:
            return await self.authorize_result_in_transaction(conn, actor, owner, run_id, result)

    async def require_schema(self) -> None:
        async with self.uow.transaction() as conn:
            rows = await (await conn.execute("SELECT version FROM harness_rag.schema_versions ORDER BY version")).fetchall()
            if [row["version"] for row in rows] not in ([1, 2, 3, 4], [1, 2, 3, 4, 5]):
                raise RuntimeFault("RETRIEVAL_EXPLICIT_MIGRATION_REQUIRED", 503)

    async def authorize_public_read(self, conn: Connection, actor: Principal, run: Row, session: Row) -> None:
        """结果/SSE回放的知识权限守卫；终态不需Worker所有权，但每页仍检查当前KB/文档权限。"""
        release_row = await (await conn.execute("SELECT manifest FROM harness.releases WHERE tenant_id=%s "
                                               "AND application_id=%s AND release_id=%s",
                                               (actor.tenant_id, actor.application_id, session["release_id"]))).fetchone()
        if release_row is None:
            raise RuntimeFault("KNOWLEDGE_RELEASE_NOT_FOUND", 404)
        manifest = ReleaseManifest.model_validate(release_row["manifest"])
        if manifest.content_hash() != run["manifest_hash"]:
            raise RuntimeFault("KNOWLEDGE_RELEASE_CHANGED", 409)
        configuration = RetrievalRelease.model_validate(manifest.profiles.as_object().get("retrieval"))
        for pin in configuration.snapshots:
            await self._kb(conn, actor, pin.knowledge_base_id)
            await self._pin(conn, actor, pin)
        records = await (await conn.execute("SELECT ticket,result,result_hash FROM harness_rag.retrieval_requests "
                                           "WHERE run_id=%s AND state='succeeded'", (run["run_id"],))).fetchall()
        for record in records:
            ticket = RetrievalTicket.model_validate(record["ticket"])
            result = RetrievalResult.model_validate(record["result"])
            if ticket.scope.manifest_hash != run["manifest_hash"] or record["result_hash"] != canonical_hash(result.model_dump(mode="json")):
                raise RuntimeFault("KNOWLEDGE_RESULT_RECEIPT_MISMATCH", 409)
            await self._validate_result(conn, ticket, result)

    async def judge_reference_material(self, conn: Connection, actor: Principal, run: Row, session: Row) -> dict[str, JsonValue]:
        """仅在当前知识权限与全部返回证据重新核对后提供检索材料。"""
        await self.authorize_public_read(conn, actor, run, session)
        rows = await (await conn.execute(
            "SELECT knowledge_base_id,result,result_hash FROM harness_rag.retrieval_requests "
            "WHERE run_id=%s AND state='succeeded' ORDER BY knowledge_base_id,retrieval_id LIMIT 5",
            (run["run_id"],),
        )).fetchall()
        release = await self.execution._release(conn, actor, session["release_id"])
        manifest = ReleaseManifest.model_validate(release["manifest"])
        expected = {item.knowledge_base_id for item in manifest.spec.knowledge_bindings}
        if (not rows or len(rows) > 4 or len(rows) != len(expected)
                or {row["knowledge_base_id"] for row in rows} != expected):
            raise RuntimeFault("JUDGE_KNOWLEDGE_EVIDENCE_UNAVAILABLE", 409)
        results: list[JsonValue] = []
        for row in rows:
            result = RetrievalResult.model_validate(row["result"])
            if canonical_hash(result.model_dump(mode="json")) != row["result_hash"]:
                raise RuntimeFault("KNOWLEDGE_RESULT_RECEIPT_MISMATCH", 409)
            results.append(result.model_dump(mode="json"))
        return {"version": "knowledge-judge-material@1", "results": results}

    async def evaluation_retrieval_evidence(
        self, conn: Connection, actor: Principal, run: Row, session: Row,
    ) -> tuple[RetrievalChildEvidence, ...]:
        """只投影当前授权 run 的子账本元数据，不导出文本、向量或完整供应商响应。"""
        release_row = await self.execution._release(conn, actor, session["release_id"])
        manifest = ReleaseManifest.model_validate(release_row["manifest"])
        configuration = RetrievalRelease.model_validate(manifest.profiles.as_object().get("retrieval"))
        run_actor = Principal.model_validate(run["principal"])
        rows = await (await conn.execute(
            "SELECT * FROM harness_rag.retrieval_requests WHERE run_id=%s "
            "ORDER BY retrieval_id LIMIT 5 FOR SHARE", (run["run_id"],),
        )).fetchall()
        if len(rows) > 4:
            raise RuntimeFault("EVALUATION_RETRIEVAL_LIMIT", 422)
        children = []
        for row in rows:
            ticket = RetrievalTicket.model_validate(row["ticket"])
            agent_binding = next((item for item in manifest.spec.knowledge_bindings
                                  if item.knowledge_base_id == row["knowledge_base_id"]), None)
            bound = (ticket.retrieval_id == row["retrieval_id"] == ticket.scope.context_id
                     and ticket.scope.owner_kind == "run" and ticket.scope.owner_id == run["run_id"]
                     and ticket.scope.manifest_hash == run["manifest_hash"] == manifest.content_hash()
                     and ticket.actor == run_actor and ticket.pin.knowledge_base_id == row["knowledge_base_id"]
                     and ticket.scope.knowledge_base_id == row["knowledge_base_id"]
                     and (row["tenant_id"], row["subject_id"], row["application_id"])
                     == (run_actor.tenant_id, run_actor.subject_id, run_actor.application_id)
                     and agent_binding is not None and agent_binding.retrieval_profile_ref == ticket.profile.ref
                     and agent_binding.snapshot_id == ticket.pin.snapshot_id
                     and ticket.pin in configuration.snapshots and ticket.profile in configuration.profiles
                     and ticket.index_binding.model_snapshot == configuration.models
                     and ticket.index_binding.backend_hash == configuration.backend_hash)
            context = await (await conn.execute(
                "SELECT * FROM harness_rag.model_contexts WHERE context_id=%s FOR SHARE", (row["retrieval_id"],),
            )).fetchone()
            calls: list[RetrievalModelCallEvidence] = []
            context_scope = None
            context_limits = None
            context_valid = None
            if context is not None:
                from harness.contracts.retrieval_models import RetrievalLimits

                context_scope = RetrievalScope.model_validate(context["scope"])
                context_limits = RetrievalLimits.model_validate(context["limits"])
                context_valid = (context["context_id"] == ticket.scope.context_id
                    and context["tenant_id"] == ticket.scope.tenant_id
                    and context["application_id"] == ticket.scope.application_id
                    and context["owner_kind"] == ticket.scope.owner_kind
                    and context["owner_id"] == ticket.scope.owner_id
                    and context["knowledge_base_id"] == ticket.scope.knowledge_base_id
                    and context["deadline"] == ticket.scope.deadline)
                raw_calls = await (await conn.execute(
                    "SELECT call_id,attempt_id,request_fingerprint,binding_hash,task_type,input_bytes,state,"
                    "result->>'call_id' AS result_call_id,result->>'attempt_id' AS result_attempt_id,"
                    "result->>'request_fingerprint' AS result_fingerprint,result->>'task_type' AS result_task_type,"
                    "result->'usage' AS usage,failure->>'dispatch_status' AS dispatch_status,"
                    "failure->>'billing_status' AS billing_status FROM harness_rag.model_calls "
                    "WHERE context_id=%s ORDER BY call_id LIMIT 129 FOR SHARE", (row["retrieval_id"],),
                )).fetchall()
                if len(raw_calls) > 128:
                    raise RuntimeFault("EVALUATION_RETRIEVAL_CALL_LIMIT", 422)
                calls = [RetrievalModelCallEvidence.model_validate(item) for item in raw_calls]
            children.append(RetrievalChildEvidence(
                retrieval_id=row["retrieval_id"], knowledge_base_id=row["knowledge_base_id"], state=row["state"],
                ticket_scope=ticket.scope, ticket_binding_valid=bound, profile_limits=ticket.profile.model_limits,
                parent_reserved_tokens=row["parent_reserved_tokens"], context_scope=context_scope,
                context_binding_valid=context_valid, context_limits=context_limits,
                calls_used=context["calls_used"] if context else None,
                input_bytes_used=context["input_bytes_used"] if context else None, calls=tuple(calls),
            ))
        return tuple(children)
