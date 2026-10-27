"""PG不可变产物；复用权威执行授权、会话锁和来源回执，不调用源工具。"""

import asyncio
import hashlib
import json
from typing import Any
from uuid import uuid4

from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.postgres.connection import Connection, Row
from harness.context.artifacts import artifact_text, page_text
from harness.contracts.artifacts import ArtifactPage, ArtifactProfile, ArtifactRead
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.state import ReceiptState, ToolReceipt
from harness.context.projection import project_model_messages


class PostgresArtifacts:
    def __init__(self, execution: PostgresExecutionStore, profile: ArtifactProfile) -> None:
        self.execution, self.profile = execution, profile

    async def _current(self, actor: Principal) -> Principal:
        source = self.execution.identity_source
        return await asyncio.to_thread(source.current, actor) if source is not None else actor

    async def _source(self, conn: Connection, run_id: str, receipt: ToolReceipt) -> bool:
        row = await self.execution._one(conn, """
            SELECT receipt,tool_name FROM harness.read_operations
            WHERE run_id=%s AND operation_id=%s AND state='succeeded'
            UNION ALL
            SELECT receipt,tool_name FROM harness.operations
            WHERE run_id=%s AND operation_id=%s AND state='succeeded' AND output_error IS NULL
            """, (run_id, receipt.operation_id, run_id, receipt.operation_id))
        if canonical_hash(row["receipt"]) != canonical_hash(receipt.model_dump(mode="json")):
            raise RuntimeFault("ARTIFACT_RECEIPT_MISMATCH", 409)
        return row["tool_name"] == "read_artifact"

    async def _authorized(self, conn: Connection, actor: Principal, session_id: str,
                          request: ArtifactRead) -> Row:
        row = await self.execution._one(conn, "SELECT * FROM harness.artifacts WHERE artifact_ref=%s AND session_id=%s",
                                        (request.artifact_ref, session_id))
        source, session = await self.execution._authorized_run(conn, actor, row["run_id"], lock=True)
        if source["session_id"] != session_id:
            raise RuntimeFault("ARTIFACT_SOURCE_SESSION_MISMATCH", 403)
        # 先session→run再artifact；读页与删除意图共享锁序，并重读锁前可能变化的撤权。
        row = await self.execution._one(
            conn, "SELECT * FROM harness.artifacts WHERE artifact_ref=%s AND session_id=%s FOR UPDATE",
            (request.artifact_ref, session_id),
        )
        await self.execution._authorize_public_resource(conn, actor, source, session)
        readable = await self.execution._one(conn, "SELECT %s::timestamptz>clock_timestamp() AS valid", (row["expires_at"],))
        if row.get("purged_at") is not None:
            raise RuntimeFault("ARTIFACT_PURGED", 410)
        if row["revoked_at"] is not None:
            raise RuntimeFault("ARTIFACT_REVOKED", 403)
        if not readable["valid"]:
            raise RuntimeFault("ARTIFACT_EXPIRED", 410)
        proof = await self.execution._one(conn, """
            SELECT receipt FROM harness.read_operations
            WHERE run_id=%s AND operation_id=%s AND state='succeeded'
            UNION ALL
            SELECT receipt FROM harness.operations
            WHERE run_id=%s AND operation_id=%s AND state='succeeded' AND output_error IS NULL
            """, (row["run_id"], row["operation_id"], row["run_id"], row["operation_id"]))
        if canonical_hash(proof["receipt"]) != row["receipt_hash"]:
            raise RuntimeFault("ARTIFACT_SOURCE_RECEIPT_CHANGED", 409)
        if (hashlib.sha256(row["body"].encode("utf-8")).hexdigest() != row["content_hash"]
            or len(row["body"]) != row["total_chars"] or len(row["body"].encode("utf-8")) != row["total_bytes"]):
            raise RuntimeFault("ARTIFACT_CONTENT_CORRUPT", 409)
        release = await self.execution._release(conn, actor, session["release_id"])
        manifest = ReleaseManifest.model_validate(release["manifest"])
        if manifest.profiles.as_object().get("artifacts") != self.profile.model_dump(mode="json"):
            raise RuntimeFault("ARTIFACT_PROFILE_UNAVAILABLE", 503)
        return row

    def _page(self, row: Row, request: ArtifactRead) -> ArtifactPage:
        text, next_offset = page_text(row["body"], request.offset, request.limit, self.profile)
        return ArtifactPage(artifact_ref=row["artifact_ref"], media_type=row["media_type"],
                            content_hash=row["content_hash"], source_run_id=row["run_id"],
                            source_operation_id=row["operation_id"], total_chars=row["total_chars"],
                            total_bytes=row["total_bytes"], expires_at=row["expires_at"],
                            offset=request.offset, next_offset=next_offset, text=text)

    async def read(self, actor: Principal, session_id: str, request: ArtifactRead) -> ArtifactPage:
        actor = await self._current(actor)
        async with self.execution._transaction() as conn:
            row = await self._authorized(conn, actor, session_id, request)
            page = self._page(row, request)
            await self.execution._audit(conn, actor, "artifact.read", row["artifact_ref"])
            return page

    async def project(self, run: ClaimedRun, owner: str, receipt: ToolReceipt) -> JsonSnapshot:
        text, media_type, digest = artifact_text(receipt.data)
        if receipt.status != ReceiptState.SUCCEEDED:
            raise RuntimeFault("ARTIFACT_SUCCESS_RECEIPT_REQUIRED", 409)
        async with self.execution._transaction() as conn:
            source, manifest = await self.execution.authorize_extension_in_transaction(conn, run.principal, owner, run.run_id)
            if source["session_id"] != run.session_id or manifest.content_hash() != run.manifest.content_hash():
                raise RuntimeFault("ARTIFACT_RUN_BINDING_MISMATCH", 409)
            session = await self.execution._session(conn, run.principal, run.session_id)
            await self.execution._authorize_public_resource(conn, run.principal, source, session)
            if manifest.profiles.as_object().get("artifacts") != self.profile.model_dump(mode="json"):
                raise RuntimeFault("ARTIFACT_PROFILE_MISMATCH", 409)
            is_page = await self._source(conn, run.run_id, receipt)
            if is_page:
                page = ArtifactPage.model_validate(receipt.data)
                request = ArtifactRead(artifact_ref=page.artifact_ref, offset=page.offset, limit=max(1, len(page.text)))
                original = await self._authorized(conn, run.principal, run.session_id, request)
                if self._page(original, request) != page:
                    raise RuntimeFault("ARTIFACT_PAGE_CHANGED", 409)
                return JsonSnapshot.capture(page.model_dump(mode="json"))
            if len(text) <= self.profile.projection_chars and len(text.encode("utf-8")) <= self.profile.page_bytes:
                return JsonSnapshot.capture(receipt.data)
            if len(text.encode("utf-8")) > self.profile.max_artifact_bytes:
                raise RuntimeFault("ARTIFACT_TOO_LARGE", 422)
            if not any(binding.name == "read_artifact" and binding.version == "1" for binding in manifest.spec.allowed_tools):
                raise RuntimeFault("ARTIFACT_READBACK_NOT_ALLOWED", 403)
            old = await (await conn.execute("SELECT * FROM harness.artifacts WHERE run_id=%s AND operation_id=%s",
                                           (run.run_id, receipt.operation_id))).fetchone()
            receipt_hash = canonical_hash(receipt.model_dump(mode="json"))
            if old is not None:
                if old["receipt_hash"] != receipt_hash or old["content_hash"] != digest:
                    raise RuntimeFault("ARTIFACT_SOURCE_CONFLICT", 409)
                request = ArtifactRead(artifact_ref=old["artifact_ref"], limit=self.profile.projection_chars)
                row = await self._authorized(conn, run.principal, run.session_id, request)
            else:
                # authorize_extension已按session→run加锁；容量与唯一来源在同一事务中串行检查。
                used = await self.execution._one(conn, "SELECT coalesce(sum(total_bytes),0) AS used FROM harness.artifacts WHERE run_id=%s", (run.run_id,))
                if used["used"] + len(text.encode("utf-8")) > self.profile.max_run_bytes:
                    raise RuntimeFault("ARTIFACT_RUN_CAPACITY", 429)
                request = ArtifactRead(artifact_ref="artifact-" + uuid4().hex, limit=self.profile.projection_chars)
                row = await self.execution._one(conn, """INSERT INTO harness.artifacts
                    (artifact_ref,run_id,session_id,operation_id,receipt_hash,media_type,body,content_hash,total_chars,total_bytes,expires_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,clock_timestamp()+%s*interval '1 second') RETURNING *""",
                    (request.artifact_ref, run.run_id, run.session_id, receipt.operation_id, receipt_hash, media_type,
                     text, digest, len(text), len(text.encode("utf-8")), self.profile.ttl_seconds))
                await self.execution._audit(conn, run.principal, "artifact.saved", row["artifact_ref"])
                await self.execution._event(conn, run.run_id, "artifact.saved", {
                    "artifact_ref": row["artifact_ref"], "session_id": run.session_id,
                    "content_hash": digest, "source_operation_id": receipt.operation_id,
                    "total_chars": row["total_chars"], "total_bytes": row["total_bytes"],
                    "expires_at": row["expires_at"].isoformat(),
                })
            page = self._page(row, request)
            # 此引用只在事务成功退出后交付；原始回执保留给完成验证器。
            projection = JsonSnapshot.capture({"artifact": page.model_dump(mode="json"), "truncated": True})
        return projection

    async def authorize_context(self, run: ClaimedRun, owner: str) -> None:
        async with self.execution._transaction() as conn:
            await self.execution.authorize_extension_in_transaction(conn, run.principal, owner, run.run_id)
            rows = await (await conn.execute("SELECT artifact_ref FROM harness.artifacts WHERE run_id=%s", (run.run_id,))).fetchall()
            pages = await (await conn.execute("SELECT receipt FROM harness.read_operations WHERE run_id=%s AND tool_name='read_artifact' AND state='succeeded'", (run.run_id,))).fetchall()
            refs = {row["artifact_ref"] for row in rows}
            for page in pages:
                refs.add(ArtifactPage.model_validate(page["receipt"]["data"]).artifact_ref)
            for ref in refs:
                await self._authorized(conn, run.principal, run.session_id, ArtifactRead(artifact_ref=ref))

    async def project_stored_messages(self, conn: Connection, actor: Principal, session_id: str,
                                      messages: JsonSnapshot, *, source_run_id: str) -> JsonSnapshot:
        """从已保存回执/产物重建工具输出；不执行工具，不为旧历史制造新ref。"""
        items: list[dict[str, Any]] = [dict(item) for item in project_model_messages(messages.as_object_array()).as_object_array()]
        names: dict[str, str] = {}
        for message in items:
            for call in message.get("tool_calls", []):
                names[call["id"]] = call["function"]["name"]
            if message["role"] != "tool":
                continue
            try:
                wrapper = json.loads(message["content"])
                operation_id = wrapper["operation_id"]
            except (ValueError, TypeError, KeyError):
                raise RuntimeFault("CONTEXT_TOOL_RECEIPT_REQUIRED", 422) from None
            if not isinstance(operation_id, str):
                raise RuntimeFault("CONTEXT_TOOL_RECEIPT_REQUIRED", 422)
            proof = await self.execution._one(conn, """
                SELECT run_id,tool_name,receipt FROM harness.read_operations WHERE operation_id=%s AND state='succeeded'
                UNION ALL SELECT run_id,tool_name,receipt FROM harness.operations
                WHERE operation_id=%s AND state='succeeded' AND output_error IS NULL
                """, (operation_id, operation_id))
            source, session = await self.execution._authorized_run(conn, actor, proof["run_id"], lock=True)
            if (source["session_id"] != session_id or proof["run_id"] != source_run_id
                or proof["tool_name"] != names.get(message["tool_call_id"])):
                raise RuntimeFault("CONTEXT_TOOL_SOURCE_MISMATCH", 403)
            await self.execution._authorize_public_resource(conn, actor, source, session)
            receipt = ToolReceipt.model_validate(proof["receipt"])
            text, _, _ = artifact_text(receipt.data)
            data = receipt.data
            if proof["tool_name"] == "read_artifact":
                page = ArtifactPage.model_validate(data)
                request = ArtifactRead(artifact_ref=page.artifact_ref, offset=page.offset, limit=max(1, len(page.text)))
                row = await self._authorized(conn, actor, session_id, request)
                if self._page(row, request) != page:
                    raise RuntimeFault("ARTIFACT_PAGE_CHANGED", 409)
            elif len(text) > self.profile.projection_chars or len(text.encode("utf-8")) > self.profile.page_bytes:
                ref = await self.execution._one(conn, "SELECT artifact_ref FROM harness.artifacts WHERE run_id=%s AND operation_id=%s",
                                                (proof["run_id"], operation_id))
                request = ArtifactRead(artifact_ref=ref["artifact_ref"], limit=self.profile.projection_chars)
                row = await self._authorized(conn, actor, session_id, request)
                data = {"artifact": self._page(row, request).model_dump(mode="json"), "truncated": True}
            message["content"] = JsonSnapshot.capture({"operation_id": operation_id, "data": data, "status": receipt.status}).canonical_text
        return project_model_messages(items)

    async def project_messages(self, run: ClaimedRun, owner: str, messages: JsonSnapshot,
                               *, source_run_id: str | None = None) -> JsonSnapshot:
        async with self.execution._transaction() as conn:
            await self.execution.authorize_extension_in_transaction(conn, run.principal, owner, run.run_id)
            if source_run_id is not None and source_run_id != run.run_id:
                await self.execution._one(conn, """SELECT t.run_id FROM harness.turns t
                    JOIN harness.runs r USING(run_id) JOIN harness.commits c USING(run_id)
                    WHERE t.run_id=%s AND t.session_id=%s AND r.status='succeeded'""",
                    (source_run_id, run.session_id))
            return await self.project_stored_messages(conn, run.principal, run.session_id, messages,
                                                      source_run_id=source_run_id or run.run_id)
