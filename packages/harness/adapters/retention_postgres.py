"""受限保留任务；session→run→job锁序，副本端口始终在锁外执行。"""

import asyncio
from datetime import timedelta
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, Row
from harness.adapters.runtime_postgres import PostgresRuntimeStore
from harness.contracts.hashing import canonical_hash, checkpoint_thread_id
from harness.contracts.retention import CleanupCopies, CleanupRequest, CleanupView, RetentionPolicy
from harness.contracts.runtime import Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


class PostgresRetention:
    def __init__(self, execution: PostgresRuntimeStore, copies: CleanupCopies | None = None) -> None:
        self.execution, self.copies = execution, copies

    async def _actor(self, actor: Principal) -> Principal:
        source = getattr(self.execution, "identity_source", None)
        return await asyncio.to_thread(source.current, actor) if source is not None else actor

    async def _session(self, conn: Connection, actor: Principal, session_id: str, *, operator: bool = False) -> Row:
        # 管理删除状态不能依赖普通正文可读性或已经到期的评测context。
        session = await self.execution._one(conn, """SELECT *,clock_timestamp() AS now FROM harness.sessions
            WHERE session_id=%s AND tenant_id=%s AND application_id=%s FOR UPDATE""",
            (session_id, actor.tenant_id, actor.application_id))
        if operator:
            if "retention_operator" not in actor.roles or f"session:{session_id}:retention" not in actor.resource_scopes:
                raise RuntimeFault("RETENTION_SCOPE_REQUIRED", 403)
        elif session["subject_id"] != actor.subject_id:
            raise RuntimeFault("NOT_FOUND", 404)
        elif not {"user", "evaluator"} & actor.roles:
            raise RuntimeFault("RETENTION_ROLE_REQUIRED", 403)
        return session

    async def _runs(self, conn: Connection, session: Row) -> list[Row]:
        policy = RetentionPolicy.model_validate(session["retention_policy"])
        return await (await conn.execute("""SELECT * FROM harness.runs WHERE session_id=%s
            ORDER BY run_id LIMIT %s FOR UPDATE""", (session["session_id"], policy.max_runs + 1))).fetchall()

    @staticmethod
    def _view(job: Row, session: Row) -> CleanupView:
        return CleanupView(job_id=job["job_id"], session_id=job["session_id"], kind=job["kind"],
            phase=job["phase"], state=job["state"], attempts=job["attempts"], reasons=tuple(job["reasons"]),
            next_check_at=job["next_check_at"], revoked_at=session["revoked_at"], request_hash=job["request_hash"])

    async def request(self, actor: Principal, session_id: str, request: CleanupRequest) -> CleanupView:
        actor = await self._actor(actor)
        digest = canonical_hash({"session_id": session_id, "request": request.model_dump(mode="json")})
        async with self.execution._transaction() as conn:
            await self.execution._lock_tenant(conn, actor)
            session = await self._session(conn, actor, session_id, operator="retention_operator" in actor.roles)
            old = await (await conn.execute("SELECT * FROM harness.cleanup_jobs WHERE session_id=%s AND request_id=%s",
                                            (session_id, request.request_id))).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    raise RuntimeFault("CLEANUP_REQUEST_CONFLICT", 409)
                return self._view(old, session)
            if session["revision"] != request.expected_revision:
                raise RuntimeFault("SESSION_REVISION_CONFLICT", 409)
            policy = RetentionPolicy.model_validate(session["retention_policy"])
            jobs = await self.execution._one(conn, "SELECT count(*) AS used FROM harness.cleanup_jobs WHERE session_id=%s", (session_id,))
            if jobs["used"] >= policy.max_jobs:
                raise RuntimeFault("CLEANUP_JOB_LIMIT", 429)
            await self.execution._rate_admit(conn, actor)
            if request.kind == "artifact":
                # 先session、再源run、再artifact；正文hash来自实际持久身份。
                ref = await self.execution._one(conn, "SELECT run_id FROM harness.artifacts WHERE session_id=%s AND artifact_ref=%s",
                                                (session_id, request.artifact_ref))
                await conn.execute("SELECT run_id FROM harness.runs WHERE run_id=%s FOR UPDATE", (ref["run_id"],))
                artifact = await self.execution._one(conn, "SELECT * FROM harness.artifacts WHERE artifact_ref=%s FOR UPDATE",
                                                     (request.artifact_ref,))
                if artifact["content_hash"] != request.expected_content_hash:
                    raise RuntimeFault("ARTIFACT_DELETE_CONTENT_MISMATCH", 409)
                if request.reason == "delete":
                    await conn.execute("UPDATE harness.artifacts SET revoked_at=coalesce(revoked_at,clock_timestamp()) WHERE artifact_ref=%s",
                                       (request.artifact_ref,))
            if request.kind == "session" and request.reason == "delete":
                session = await self.execution._one(conn, """UPDATE harness.sessions SET revoked_at=coalesce(revoked_at,clock_timestamp())
                    WHERE session_id=%s RETURNING *,clock_timestamp() AS now""", (session_id,))
                await conn.execute("UPDATE harness.artifacts SET revoked_at=coalesce(revoked_at,clock_timestamp()) WHERE session_id=%s",
                                   (session_id,))
            job = await self.execution._one(conn, """INSERT INTO harness.cleanup_jobs
                (job_id,session_id,request_id,request_hash,request,policy,kind) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                (uuid4().hex, session_id, request.request_id, digest, Jsonb(request.model_dump(mode="json")),
                 Jsonb(session["retention_policy"]), request.kind))
            await self.execution._audit(conn, actor, "retention.requested", job["job_id"])
            return self._view(job, session)

    async def _lock_job(self, conn: Connection, actor: Principal, job_id: str, *, operator: bool = False) -> tuple[Row, list[Row], Row]:
        ref = await self.execution._one(conn, "SELECT session_id FROM harness.cleanup_jobs WHERE job_id=%s", (job_id,))
        session = await self._session(conn, actor, ref["session_id"], operator=operator)
        runs = await self._runs(conn, session)
        job = await self.execution._one(conn, "SELECT * FROM harness.cleanup_jobs WHERE job_id=%s FOR UPDATE", (job_id,))
        return session, runs, job

    async def inspect(self, actor: Principal, job_id: str) -> CleanupView:
        actor = await self._actor(actor)
        async with self.execution._transaction() as conn:
            session, _, job = await self._lock_job(conn, actor, job_id, operator="retention_operator" in actor.roles)
            await self.execution._audit(conn, actor, "retention.inspected", job_id)
            return self._view(job, session)

    async def _reasons(self, conn: Connection, session: Row, runs: list[Row], job: Row) -> tuple[str, ...]:
        policy = RetentionPolicy.model_validate(job["policy"])
        reasons = []
        now = session["now"]
        ids = [r["run_id"] for r in runs]
        if len(runs) > policy.max_runs:
            reasons.append("run_limit_requires_manual_cleanup")
        if session["legal_hold"] or (session["retain_until"] is not None and now < session["retain_until"]):
            reasons.append("retention_hold")
        if session["active_run_id"] is not None or any(r["status"] not in {"succeeded", "partial", "failed", "cancelled", "expired"}
                                                     or r["recovery_required"] for r in runs):
            reasons.append("active_or_recovery_run")
        facts = await self.execution._one(conn, """SELECT
            EXISTS(SELECT 1 FROM harness.worker_registration WHERE active_run_id=ANY(%s)) AS worker,
            EXISTS(SELECT 1 FROM harness.operations WHERE run_id=ANY(%s)) AS writes,
            EXISTS(SELECT 1 FROM harness.operations WHERE run_id=ANY(%s) AND
                (state IN ('prepared','dispatched','unknown') OR reconcile_token IS NOT NULL)) AS unresolved,
            EXISTS(SELECT 1 FROM harness.approvals a JOIN harness.operations o USING(operation_id)
                WHERE o.run_id=ANY(%s) AND a.state='pending') AS approval,
            EXISTS(SELECT 1 FROM harness.model_calls WHERE run_id=ANY(%s) AND state IN ('dispatched','unknown')) AS model,
            EXISTS(SELECT 1 FROM harness.read_operations WHERE run_id=ANY(%s) AND state IN ('dispatched','unknown')) AS reads,
            EXISTS(SELECT 1 FROM harness.evaluation_contexts WHERE context_id=%s AND revoked_at IS NULL
                AND expires_at>clock_timestamp()) AS evaluation,
            EXISTS(SELECT 1 FROM harness.cleanup_jobs WHERE session_id=%s AND job_id<>%s AND state='running') AS competing
            """, (ids, ids, ids, ids, ids, ids, session["evaluation_context_id"], session["session_id"], job["job_id"]))
        reasons.extend(key for key in ("worker", "unresolved", "approval", "model", "reads", "evaluation", "competing") if facts[key])
        if facts["writes"] and job["kind"] in {"session", "artifact", "extensions"}:
            reasons.append("write_audit_retained")
        request = CleanupRequest.model_validate(job["request"])
        if request.expected_revision != session["revision"]:
            reasons.append("session_revision_changed")
        if job["target"] is not None and job["target"]["threads"] != [
            checkpoint_thread_id(session["tenant_id"], session["application_id"], r["run_id"]) for r in runs]:
            reasons.append("cleanup_target_changed")
        newest = max([session["created_at"], *[r["created_at"] for r in runs]])
        latest = await self.execution._one(conn, """SELECT greatest(
            (SELECT max(created_at) FROM harness.commits WHERE run_id=ANY(%s)),
            (SELECT max(created_at) FROM harness.model_calls WHERE run_id=ANY(%s)),
            (SELECT max(created_at) FROM harness.extension_decisions WHERE run_id=ANY(%s)),
            (SELECT max(timestamp) FROM harness.events WHERE run_id=ANY(%s))) AS at""", (ids, ids, ids, ids))
        if latest["at"] is not None:
            newest = max(newest, latest["at"])
        seconds = {"session": policy.private_seconds, "events": policy.event_seconds,
                   "extensions": policy.private_seconds, "trace": policy.trace_seconds, "artifact": 0}[job["kind"]]
        if now < newest + timedelta(seconds=seconds):
            reasons.append("retention_window")
        if policy.external_trace_copies and job["kind"] in {"session", "trace"}:
            reasons.append("external_trace_cleanup_not_installed")
        if job["kind"] == "artifact":
            artifact = await self.execution._one(conn, "SELECT * FROM harness.artifacts WHERE session_id=%s AND artifact_ref=%s FOR UPDATE",
                                                 (session["session_id"], request.artifact_ref))
            if artifact["content_hash"] != request.expected_content_hash:
                reasons.append("artifact_identity_changed")
            if request.reason == "ttl" and now < artifact["expires_at"]:
                reasons.append("artifact_not_expired")
            if artifact["purged_at"] is None and (any(r["owner_id"] for r in runs) or
                await (await conn.execute("""SELECT 1 FROM harness.read_operations WHERE run_id=ANY(%s)
                    UNION ALL SELECT 1 FROM harness.model_calls WHERE run_id=ANY(%s)
                    UNION ALL SELECT 1 FROM harness.turns WHERE session_id=%s LIMIT 1""",
                    (ids, ids, session["session_id"]))).fetchone()):
                reasons.append("artifact_body_copies_retained")
        # 只探测本库可选表是否存在，不加载或连接任何RAG供应商。
        if job["kind"] == "session":
            for table, condition, args in (
                ("retrieval_requests", "run_id=ANY(%s)", (ids,)),
                ("model_contexts", "owner_kind='run' AND owner_id=ANY(%s)", (ids,)),
            ):
                exists = await self.execution._one(conn, "SELECT to_regclass(%s) AS ref", (f"harness_rag.{table}",))
                if exists["ref"] is not None and await (await conn.execute(
                    f"SELECT 1 FROM harness_rag.{table} WHERE {condition} LIMIT 1", args)).fetchone():
                    reasons.append("optional_rag_copies_retained")
        return tuple(dict.fromkeys(reasons))

    async def advance(self, actor: Principal, job_id: str) -> CleanupView:
        actor = await self._actor(actor)
        async with self.execution._transaction() as conn:
            session, runs, job = await self._lock_job(conn, actor, job_id, operator=True)
            if job["state"] in {"completed", "manual"}:
                return self._view(job, session)
            if job["kind"] == "session" and session["purged_at"] is not None:
                # 其他同会话任务已完成传播时直接复用墓碑，不能重新改写最少审计hash。
                return await self._complete(conn, actor, session, job)
            if job["state"] == "running":
                raise RuntimeFault("CLEANUP_RECONCILIATION_REQUIRED", 409)
            reasons = await self._reasons(conn, session, runs, job)
            policy = RetentionPolicy.model_validate(job["policy"])
            if reasons:
                job = await self.execution._one(conn, """UPDATE harness.cleanup_jobs SET state='blocked',reasons=%s,
                    next_check_at=clock_timestamp()+%s*interval '1 second',updated_at=clock_timestamp() WHERE job_id=%s RETURNING *""",
                    (Jsonb(list(reasons)), policy.recheck_seconds, job_id))
                await self.execution._audit(conn, actor, "retention.blocked", job_id)
                return self._view(job, session)
            request = CleanupRequest.model_validate(job["request"])
            spent = await self.execution._one(conn, """SELECT coalesce(sum(attempts),0) AS used
                FROM harness.cleanup_jobs WHERE session_id=%s AND kind=%s
                AND request->>'artifact_ref' IS NOT DISTINCT FROM %s""",
                (session["session_id"], job["kind"], request.artifact_ref))
            # 资源级额度也跨request_id保留，不能换幂等键绕过未决或人工处理边界。
            if spent["used"] >= policy.max_attempts:
                return await self._failure(conn, actor, session, job, "attempt_limit", manual=True)
            if job["kind"] == "session":
                session = await self.execution._one(conn, """UPDATE harness.sessions SET revoked_at=coalesce(revoked_at,clock_timestamp())
                    WHERE session_id=%s RETURNING *,clock_timestamp() AS now""", (session["session_id"],))
            needs_copies = job["kind"] in {"session", "trace"} and job["phase"] == "copies"
            if needs_copies and self.copies is None:
                return await self._failure(conn, actor, session, job, "copy_cleanup_not_installed", manual=False)
            target = JsonSnapshot.capture({"kind": job["kind"], "session_id": session["session_id"],
                "threads": [checkpoint_thread_id(session["tenant_id"], session["application_id"], r["run_id"]) for r in runs]})
            token = uuid4().hex
            job = await self.execution._one(conn, """UPDATE harness.cleanup_jobs SET state='running',token=%s,
                attempts=attempts+1,reasons='[]',next_check_at=NULL,target=%s,
                phase=%s,updated_at=clock_timestamp() WHERE job_id=%s RETURNING *""",
                (token, Jsonb(target.materialize()), "copies" if needs_copies else "payloads", job_id))
            await self.execution._audit(conn, actor, "retention.started", job_id)
        if not needs_copies:
            return await self._payloads(actor, job_id, token)
        # 未持有业务锁。取消/结果不明保留running，显式reconcile先查询实际副本状态。
        assert self.copies is not None
        try:
            await self.copies.erase(target)
            absent = await self.copies.absent(target)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 外部错误不能证明未删除，不自动重放；不把异常正文写进公共视图。
            return await self.inspect(actor, job_id)
        if not absent:
            return await self.inspect(actor, job_id)
        return await self.reconcile(actor, job_id)

    async def reconcile(self, actor: Principal, job_id: str) -> CleanupView:
        actor = await self._actor(actor)
        async with self.execution._transaction() as conn:
            session, _, job = await self._lock_job(conn, actor, job_id, operator=True)
            if job["state"] != "running":
                return self._view(job, session)
            if job["phase"] == "payloads":
                # 同事务完成标记未提交说明PG正文阶段没有成功；新token阻断旧调用继续。
                return await self._failure(conn, actor, session, job, "payload_commit_not_observed",
                    manual=job["attempts"] >= RetentionPolicy.model_validate(job["policy"]).max_attempts)
            token, target = job["token"], JsonSnapshot.capture(job["target"])
        if self.copies is None:
            raise RuntimeFault("COPY_CLEANUP_NOT_INSTALLED", 503)
        absent = await self.copies.absent(target)
        actor = await self._actor(actor)
        async with self.execution._transaction() as conn:
            session, runs, job = await self._lock_job(conn, actor, job_id, operator=True)
            if job["token"] != token or job["state"] != "running":
                raise RuntimeFault("CLEANUP_CAS_CONFLICT", 409)
            reasons = await self._reasons(conn, session, runs, job)
            if reasons or not absent:
                return await self._failure(conn, actor, session, job,
                    "copies_still_present" if not absent else "protection_changed", manual=job["attempts"] >= RetentionPolicy.model_validate(job["policy"]).max_attempts)
            job = await self.execution._one(conn, """UPDATE harness.cleanup_jobs SET state='pending',phase='payloads',
                token=NULL,updated_at=clock_timestamp() WHERE job_id=%s RETURNING *""", (job_id,))
            await self.execution._audit(conn, actor, "retention.copies_confirmed", job_id)
            if job["kind"] == "trace":
                return await self._complete(conn, actor, session, job)
            return self._view(job, session)

    async def _payloads(self, actor: Principal, job_id: str, token: str) -> CleanupView:
        try:
            actor = await self._actor(actor)
            async with self.execution._transaction() as conn:
                session, runs, job = await self._lock_job(conn, actor, job_id, operator=True)
                if job["token"] != token or job["state"] != "running" or job["phase"] != "payloads":
                    raise RuntimeFault("CLEANUP_CAS_CONFLICT", 409)
                reasons = await self._reasons(conn, session, runs, job)
                if reasons:
                    return await self._failure(conn, actor, session, job, "protection_changed", manual=False)
                # 正文、event_floor和完成审计同事务；尝试意图已独立持久化。
                await self._purge(conn, session, runs, job)
                return await self._complete(conn, actor, session, job)
        except asyncio.CancelledError:
            raise
        except Exception:
            # commit确认丢失时先查询任务状态；completed保持幂等，不盲删第二次。
            async with self.execution._transaction() as conn:
                session, _, job = await self._lock_job(conn, actor, job_id, operator=True)
                if job["state"] == "completed":
                    return self._view(job, session)
                if job["state"] != "running" or job["token"] != token:
                    raise RuntimeFault("CLEANUP_CAS_CONFLICT", 409) from None
                return await self._failure(conn, actor, session, job, "payload_transaction_failed",
                    manual=job["attempts"] >= RetentionPolicy.model_validate(job["policy"]).max_attempts)

    async def _failure(self, conn: Connection, actor: Principal, session: Row, job: Row,
                       reason: str, *, manual: bool) -> CleanupView:
        job = await self.execution._one(conn, """UPDATE harness.cleanup_jobs SET state=%s,reasons=%s,
            token=NULL,next_check_at=clock_timestamp()+%s*interval '1 second',updated_at=clock_timestamp() WHERE job_id=%s RETURNING *""",
            ("manual" if manual else "retry", Jsonb([reason]), RetentionPolicy.model_validate(job["policy"]).recheck_seconds, job["job_id"]))
        await self.execution._audit(conn, actor, "retention.retry_or_manual", job["job_id"])
        return self._view(job, session)

    async def _complete(self, conn: Connection, actor: Principal, session: Row, job: Row) -> CleanupView:
        job = await self.execution._one(conn, """UPDATE harness.cleanup_jobs SET state='completed',phase='done',
            token=NULL,next_check_at=NULL,updated_at=clock_timestamp() WHERE job_id=%s RETURNING *""", (job["job_id"],))
        await self.execution._audit(conn, actor, "retention.completed", job["job_id"])
        return self._view(job, session)

    async def _purge(self, conn: Connection, session: Row, runs: list[Row], job: Row) -> None:
        ids = [r["run_id"] for r in runs]
        kind = job["kind"]
        policy = RetentionPolicy.model_validate(job["policy"])
        if kind in {"session", "events"}:
            # 全部事件一起到期才删；不留下无法补齐的中段洞。
            for run in runs:
                if kind == "events" and await (await conn.execute("""SELECT 1 FROM harness.events WHERE run_id=%s AND timestamp>
                    clock_timestamp()-%s*interval '1 second' LIMIT 1""", (run["run_id"], policy.event_seconds))).fetchone():
                    raise RuntimeFault("EVENT_RETENTION_WINDOW", 409)
                await conn.execute("UPDATE harness.runs SET event_floor=greatest(event_floor,event_seq) WHERE run_id=%s", (run["run_id"],))
                await conn.execute("DELETE FROM harness.events WHERE run_id=%s", (run["run_id"],))
        if kind in {"session", "extensions"}:
            await conn.execute("DELETE FROM harness.extension_decisions WHERE run_id=ANY(%s)", (ids,))
        if kind == "session":
            commits = await (await conn.execute("SELECT commit_id,validation FROM harness.commits WHERE run_id=ANY(%s)", (ids,))).fetchall()
            for commit in commits:
                validation = commit["validation"]
                await conn.execute("UPDATE harness.commits SET validation=%s WHERE commit_id=%s", (Jsonb({
                    "original_validation_hash": canonical_hash(validation), "input_hash": validation.get("input_hash"),
                    "verdict": validation.get("verdict"), "checked_at": validation.get("checked_at")}), commit["commit_id"]))
            for table in ("session_summaries", "summary_candidates", "turns"):
                await conn.execute(f"DELETE FROM harness.{table} WHERE session_id=%s", (session["session_id"],))
            for table in ("answer_drafts", "model_calls", "read_operations"):
                await conn.execute(f"DELETE FROM harness.{table} WHERE run_id=ANY(%s)", (ids,))
            await conn.execute("UPDATE harness.runs SET request='{}',result=NULL,suspension=NULL WHERE session_id=%s", (session["session_id"],))
            await conn.execute("""UPDATE harness.sessions SET summary_cursor_hash=NULL,purged_at=coalesce(purged_at,clock_timestamp())
                WHERE session_id=%s""", (session["session_id"],))
        if kind in {"session", "artifact"}:
            await conn.execute("SELECT set_config('harness.cleanup_job',%s,true)", (job["job_id"],))
            request = CleanupRequest.model_validate(job["request"])
            await conn.execute("""UPDATE harness.artifacts SET revoked_at=coalesce(revoked_at,clock_timestamp()),
                body=NULL,purged_at=clock_timestamp() WHERE session_id=%s AND purged_at IS NULL"""
                + (" AND artifact_ref=%s" if kind == "artifact" else ""),
                (session["session_id"], request.artifact_ref) if kind == "artifact" else (session["session_id"],))
