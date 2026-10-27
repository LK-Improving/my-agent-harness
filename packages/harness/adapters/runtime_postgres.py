"""M1 PostgreSQL 业务事务；连接异常绝不触发执行权转移。"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, PostgresUnitOfWork, Row, one
from harness.adapters.postgres.events import PostgresEventJournal
from harness.adapters.postgres.lifecycle import PostgresRunLifecycle
from harness.adapters.postgres.mapping import run_view
from harness.adapters.postgres.queries import PostgresEvidenceQueries
from harness.context.boundary import require_memory_boundary
from harness.contracts.events import RuntimeEvent
from harness.contracts.context import ContextProfile, SummaryCommitProfile
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import (
    AdmissionProfile,
    AnswerCandidate,
    CancelRequest,
    ClaimedRun,
    EvaluationRequest,
    Principal,
    RuntimeFault,
    RunView,
    SessionRequest,
)
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.retention import RetentionPolicy


class PostgresRuntimeStore:
    retention_policy = RetentionPolicy()
    summary_authority: Callable[[Connection, str, ClaimedRun, AnswerCandidate, str], Awaitable[None]] | None = None
    summary_authority_ref: str | None = None
    answer_authority: Callable[[Connection, str, ClaimedRun, AnswerCandidate], Awaitable[None]] | None = None
    resource_authority: Callable[[Connection, Principal, Row, Row], Awaitable[None]] | None = None
    resource_authority_ref: str | None = None
    # None仅供旧独立调用方；实际组合根始终设置冻结的已安装manifest集合。
    installed_manifest_hashes: frozenset[str] | None = None

    def __init__(self, dsn: str, profile: AdmissionProfile) -> None:
        self._uow = PostgresUnitOfWork(dsn)
        self._evidence = PostgresEvidenceQueries()
        self.profile = profile
        self._events = PostgresEventJournal()
        self._lifecycle = PostgresRunLifecycle(self._events)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[Connection]:
        async with self._uow.transaction() as conn:
            yield conn

    _one = staticmethod(one)

    @staticmethod
    def _role(actor: Principal, role: str) -> None:
        if role not in actor.roles:
            raise RuntimeFault("FORBIDDEN", 403)

    async def _audit(self, conn: Connection, actor: Principal, action: str, resource: str) -> None:
        await conn.execute(
            (
                "INSERT INTO "
                "harness.audit(tenant_id,subject_id,action,resource_id) "
                "VALUES(%s,%s,%s,%s)"
            ),
            (actor.tenant_id, actor.subject_id, action, resource),
        )

    async def _admit(self, conn: Connection, actor: Principal) -> None:
        await self._lock_tenant(conn, actor)
        await self._rate_admit(conn, actor)

    async def _lock_tenant(self, conn: Connection, actor: Principal) -> None:
        # 租户行先锁定，跨 API 进程的排队与速率上限在同一数据库事务串行生效。
        await conn.execute(
            "INSERT INTO harness.tenant_admission VALUES(%s) ON CONFLICT DO NOTHING",
            (actor.tenant_id,),
        )
        await conn.execute(
            "SELECT tenant_id FROM harness.tenant_admission WHERE tenant_id=%s FOR UPDATE",
            (actor.tenant_id,),
        )

    async def _rate_admit(self, conn: Connection, actor: Principal) -> None:
        row = await self._one(
            conn,
            """INSERT INTO harness.request_windows
               VALUES(%s,%s,%s,date_trunc('minute',clock_timestamp()),1)
               ON CONFLICT(tenant_id,subject_id,application_id) DO UPDATE SET
               used=CASE WHEN request_windows.window_start=date_trunc('minute',clock_timestamp())
                         THEN request_windows.used+1 ELSE 1 END,
               window_start=date_trunc('minute',clock_timestamp()) RETURNING used""",
            (actor.tenant_id, actor.subject_id, actor.application_id),
        )
        if row["used"] > self.profile.requests_per_subject_minute:
            raise RuntimeFault("RATE_LIMITED", 429)

    async def register_release(self, actor: Principal, release: ReleaseManifest) -> str:
        self._role(actor, "release_admin")
        async with self._transaction() as conn:
            await self._admit(conn, actor)
            cursor = await conn.execute(
                """INSERT INTO harness.releases
                   (tenant_id,application_id,release_id,agent_id,manifest_hash,manifest)
                   VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING release_id""",
                (
                    actor.tenant_id,
                    actor.application_id,
                    release.spec.release_id,
                    release.spec.agent_id,
                    release.content_hash(),
                    Jsonb(release.model_dump(mode="json")),
                ),
            )
            if await cursor.fetchone() is None:
                row = await self._release(conn, actor, release.spec.release_id)
                if row["manifest_hash"] != release.content_hash():
                    raise RuntimeFault("IMMUTABLE_RELEASE_CONFLICT")
            await self._audit(conn, actor, "release.registered", release.spec.release_id)
        return release.content_hash()

    async def _release(self, conn: Connection, actor: Principal, release_id: str) -> Row:
        return await self._one(
            conn,
            (
                "SELECT * FROM harness.releases WHERE tenant_id=%s AND "
                "application_id=%s AND release_id=%s"
            ),
            (actor.tenant_id, actor.application_id, release_id),
        )

    async def create_evaluation(
        self, actor: Principal, request: EvaluationRequest
    ) -> EvaluationContext:
        self._role(actor, "evaluation_admin")
        self._role(actor, "evaluator")
        if not request.resource_scopes <= actor.resource_scopes:
            raise RuntimeFault("SCOPE_ESCALATION", 403)
        async with self._transaction() as conn:
            await self._admit(conn, actor)
            row = await self._release(conn, actor, request.candidate_release_id)
            manifest = ReleaseManifest.model_validate(row["manifest"])
            now = (await self._one(conn, "SELECT clock_timestamp() AS now"))["now"]
            if not now < request.expires_at <= now + timedelta(hours=24):
                raise RuntimeFault("INVALID_EVALUATION_EXPIRY", 422)
            context = EvaluationContext(
                context_id=uuid4().hex,
                tenant_id=actor.tenant_id,
                subject_id=actor.subject_id,
                agent_id=manifest.spec.agent_id,
                candidate_release_id=manifest.spec.release_id,
                manifest_hash=manifest.content_hash(),
                dataset_hash=request.dataset_hash,
                gate_policy_version=request.gate_policy_version,
                environment_profile_hash=manifest.environment_profile_hash,
                resource_scopes=request.resource_scopes,
                expires_at=request.expires_at,
            )
            await conn.execute(
                """INSERT INTO harness.evaluation_contexts
                   (context_id,tenant_id,subject_id,application_id,release_id,context,expires_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s)""",
                (
                    context.context_id,
                    actor.tenant_id,
                    actor.subject_id,
                    actor.application_id,
                    context.candidate_release_id,
                    Jsonb(context.model_dump(mode="json")),
                    context.expires_at,
                ),
            )
            await self._audit(conn, actor, "evaluation.created", context.context_id)
            return context

    async def _evaluation(
        self, conn: Connection, actor: Principal, context_id: str
    ) -> EvaluationContext:
        self._role(actor, "evaluator")
        row = await self._one(
            conn,
            """SELECT *,clock_timestamp() AS now FROM harness.evaluation_contexts
               WHERE context_id=%s AND tenant_id=%s AND subject_id=%s AND application_id=%s""",
            (context_id, actor.tenant_id, actor.subject_id, actor.application_id),
        )
        context = EvaluationContext.model_validate(row["context"])
        if row["revoked_at"] is not None or row["now"] >= row["expires_at"]:
            raise RuntimeFault("EVALUATION_REVOKED_OR_EXPIRED", 403)
        if not context.resource_scopes <= actor.resource_scopes:
            raise RuntimeFault("EVALUATION_SCOPE_REVOKED", 403)
        return context

    async def create_session(self, actor: Principal, request: SessionRequest) -> str:
        async with self._transaction() as conn:
            await self._admit(conn, actor)
            if request.evaluation_context_id:
                context = await self._evaluation(conn, actor, request.evaluation_context_id)
                if context.agent_id != request.agent_id:
                    raise RuntimeFault("EVALUATION_AGENT_MISMATCH", 403)
                release_id = context.candidate_release_id
            else:
                self._role(actor, "user")
                row = await self._one(
                    conn,
                    """SELECT release_id FROM harness.active_releases
                       WHERE tenant_id=%s AND application_id=%s AND agent_id=%s""",
                    (actor.tenant_id, actor.application_id, request.agent_id),
                )
                release_id = row["release_id"]
            session_id = uuid4().hex
            await conn.execute(
                """INSERT INTO harness.sessions
                   (session_id,tenant_id,subject_id,application_id,agent_id,release_id,evaluation_context_id,retention_policy)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    session_id,
                    actor.tenant_id,
                    actor.subject_id,
                    actor.application_id,
                    request.agent_id,
                    release_id,
                    request.evaluation_context_id,
                    Jsonb(self.retention_policy.model_dump(mode="json")),
                ),
            )
            await self._audit(conn, actor, "session.created", session_id)
            return session_id

    async def _session(
        self, conn: Connection, actor: Principal, session_id: str, *, lock: bool = False
    ) -> Row:
        row = await self._one(
            conn,
            """SELECT *,clock_timestamp() AS retention_now FROM harness.sessions WHERE session_id=%s
               AND tenant_id=%s AND subject_id=%s AND application_id=%s"""
            + (" FOR UPDATE" if lock else ""),
            (session_id, actor.tenant_id, actor.subject_id, actor.application_id),
        )
        if row.get("revoked_at") is not None or row.get("purged_at") is not None:
            raise RuntimeFault("SESSION_REVOKED", 403)
        if row.get("created_at") is not None and row["active_run_id"] is None:
            policy = RetentionPolicy.model_validate(row["retention_policy"])
            if row["retention_now"] >= row["created_at"] + timedelta(seconds=policy.session_seconds):
                raise RuntimeFault("SESSION_EXPIRED", 410)
        if row["evaluation_context_id"]:
            await self._evaluation(conn, actor, row["evaluation_context_id"])
        else:
            self._role(actor, "user")
        return row

    _view = staticmethod(run_view)

    async def _event(
        self, conn: Connection, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        await self._events.append(conn, run_id, event_type, payload)

    async def create_run(self, actor: Principal, session_id: str, request: RunRequest) -> RunView:
        async with self._transaction() as conn:
            await self._lock_tenant(conn, actor)
            session = await self._session(conn, actor, session_id, lock=True)
            existing = await (
                await conn.execute(
                    "SELECT * FROM harness.runs WHERE session_id=%s AND request_id=%s",
                    (session_id, request.request_id),
                )
            ).fetchone()
            # 去重先于 revision、活动运行和新请求额度校验；鉴权永远先于去重。
            if existing:
                if existing["request_hash"] != request.business_hash():
                    raise RuntimeFault("REQUEST_ID_CONFLICT")
                await self._authorize_public_resource(conn, actor, existing, session)
                return self._view(existing, session)
            ready = getattr(self, "new_run_ready", None)
            if ready is not None and not await asyncio.to_thread(ready):
                raise RuntimeFault("MODEL_CAPABILITY_OR_CREDENTIAL_NOT_READY", 503)
            await self._rate_admit(conn, actor)
            if request.attachment_versions or request.business_options:
                raise RuntimeFault("INPUT_EXTENSION_NOT_ENABLED", 422)
            if request.expected_revision != session["revision"]:
                raise RuntimeFault("SESSION_REVISION_CONFLICT")
            if session["active_run_id"] is not None:
                raise RuntimeFault("SESSION_BUSY")
            counts = await self._one(
                conn,
                """SELECT count(*) FILTER(WHERE r.status='queued') AS queued,count(*) AS open
                   FROM harness.runs r JOIN harness.sessions s USING(session_id)
                   WHERE s.tenant_id=%s AND r.status IN
                   ('queued','running','waiting_approval','reconciling')""",
                (actor.tenant_id,),
            )
            if (
                counts["queued"] >= self.profile.max_queued_per_tenant
                or counts["open"] >= self.profile.max_open_runs_per_tenant
            ):
                raise RuntimeFault("TENANT_QUEUE_FULL", 429)
            release = await self._release(conn, actor, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            if (
                self.installed_manifest_hashes is not None
                and manifest.content_hash() not in self.installed_manifest_hashes
            ):
                raise RuntimeFault("EXECUTABLE_RELEASE_NOT_INSTALLED", 503)
            if manifest.spec.graph_template != "bounded_tool_agent":
                raise RuntimeFault("CAPABILITY_NOT_ENABLED", 422)
            if session["evaluation_context_id"]:
                context = await self._evaluation(conn, actor, session["evaluation_context_id"])
                if (
                    context.manifest_hash != manifest.content_hash()
                    or context.environment_profile_hash != manifest.environment_profile_hash
                ):
                    raise RuntimeFault("EVALUATION_MANIFEST_MISMATCH", 403)
                actor = actor.model_copy(update={"resource_scopes": context.resource_scopes})
            run_id = uuid4().hex
            row = await self._one(
                conn,
                """INSERT INTO harness.runs(run_id,session_id,request_id,request_hash,request,
                   manifest_hash,environment_profile_hash,principal,queue_deadline,expires_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,
                   clock_timestamp()+%s * interval '1 second',
                   clock_timestamp()+%s * interval '1 second')
                   RETURNING *""",
                (
                    run_id,
                    session_id,
                    request.request_id,
                    request.business_hash(),
                    Jsonb(request.model_dump(mode="json")),
                    manifest.content_hash(),
                    manifest.environment_profile_hash,
                    Jsonb(actor.model_dump(mode="json")),
                    self.profile.queue_timeout_seconds,
                    self.profile.run_timeout_seconds,
                ),
            )
            await conn.execute(
                "UPDATE harness.sessions SET active_run_id=%s WHERE session_id=%s",
                (run_id, session_id),
            )
            await self._event(
                conn,
                run_id,
                "run.accepted",
                {
                    "release_id": session["release_id"],
                    "manifest_hash": manifest.content_hash(),
                    "environment_profile_hash": manifest.environment_profile_hash,
                },
            )
            await self._audit(conn, actor, "run.created", run_id)
            return self._view(row, session)

    async def _authorized_run(
        self, conn: Connection, actor: Principal, run_id: str, *, lock: bool = False
    ) -> tuple[Row, Row]:
        ref = await self._one(
            conn, "SELECT session_id FROM harness.runs WHERE run_id=%s", (run_id,)
        )
        session = await self._session(conn, actor, ref["session_id"], lock=lock)
        row = await self._one(
            conn,
            "SELECT * FROM harness.runs WHERE run_id=%s" + (" FOR UPDATE" if lock else ""),
            (run_id,),
        )
        execution_actor = Principal.model_validate(row["principal"])
        if not execution_actor.resource_scopes <= actor.resource_scopes:
            raise RuntimeFault("RESOURCE_SCOPE_REVOKED", 403)
        return row, session

    async def _authorize_public_resource(self, conn: Connection, actor: Principal, row: Row, session: Row) -> None:
        release = await self._one(conn, "SELECT manifest FROM harness.releases WHERE tenant_id=%s "
                                  "AND application_id=%s AND release_id=%s",
                                  (actor.tenant_id, actor.application_id, session["release_id"]))
        manifest = ReleaseManifest.model_validate(release["manifest"])
        if manifest.content_hash() != row["manifest_hash"]:
            raise RuntimeFault("STORED_MANIFEST_MISMATCH", 409)
        required = manifest.profiles.as_object().get("resource_authority_ref")
        if required is not None:
            if self.resource_authority is None or self.resource_authority_ref != required:
                raise RuntimeFault("RESOURCE_AUTHORITY_UNAVAILABLE", 503)
            await self.resource_authority(conn, actor, row, session)

    async def get_run(self, actor: Principal, run_id: str) -> RunView:
        async with self._transaction() as conn:
            row, session = await self._authorized_run(conn, actor, run_id)
            await self._authorize_public_resource(conn, actor, row, session)
            return self._view(row, session)

    async def cancel_run(self, actor: Principal, run_id: str, request: CancelRequest) -> RunView:
        async with self._transaction() as conn:
            await self._admit(conn, actor)
            row, session = await self._authorized_run(conn, actor, run_id, lock=True)
            old = await (
                await conn.execute(
                    "SELECT 1 FROM harness.cancel_requests WHERE run_id=%s AND request_id=%s",
                    (run_id, request.request_id),
                )
            ).fetchone()
            if old:
                return self._view(row, session).model_copy(update={"result": None})
            if row["revision"] != request.expected_run_revision:
                raise RuntimeFault("RUN_REVISION_CONFLICT")
            if row["status"] not in {"queued", "running", "waiting_approval", "reconciling"}:
                return self._view(row, session).model_copy(update={"result": None})
            await conn.execute(
                "INSERT INTO harness.cancel_requests VALUES(%s,%s)", (run_id, request.request_id)
            )
            await conn.execute(
                "UPDATE harness.runs SET cancel_requested=true,revision=revision+1 WHERE run_id=%s",
                (run_id,),
            )
            await self._event(conn, run_id, "run.cancel_requested", {})
            if row["status"] in {"queued", "waiting_approval", "reconciling"}:
                await self._terminate(conn, row, "cancelled", "CANCELLED")
            updated = await self._one(conn, "SELECT * FROM harness.runs WHERE run_id=%s", (run_id,))
            # 取消保留可用性，但不能成为撤权后读取旧答案的旁路。
            return self._view(updated, session).model_copy(update={"result": None})

    async def events(self, actor: Principal, run_id: str, after: int) -> list[dict[str, Any]]:
        async with self._transaction() as conn:
            row, session = await self._authorized_run(conn, actor, run_id)
            await self._authorize_public_resource(conn, actor, row, session)
            if after < row["event_floor"]:
                raise RuntimeFault("EVENT_CURSOR_EXPIRED", 410)
            if after > row["event_seq"]:
                raise RuntimeFault("EVENT_CURSOR_AHEAD", 409)
            events = await (
                await conn.execute(
                    "SELECT * FROM harness.events WHERE run_id=%s AND seq>%s ORDER BY seq LIMIT %s",
                    (run_id, after, self.profile.event_page_size),
                )
            ).fetchall()
            return [RuntimeEvent.model_validate(item).model_dump(mode="json") for item in events]

    async def validate_event_cursor(self, actor: Principal, run_id: str, after: int) -> None:
        """逐条公开前重查清理水位；已加载批次不能跨越新event_floor继续回放。"""
        async with self._transaction() as conn:
            row, session = await self._authorized_run(conn, actor, run_id)
            await self._authorize_public_resource(conn, actor, row, session)
            if after < row["event_floor"]:
                raise RuntimeFault("EVENT_CURSOR_EXPIRED", 410)

    async def register_worker(self, identity: str) -> None:
        async with self._transaction() as conn:
            row = await (
                await conn.execute(
                    """INSERT INTO harness.worker_registration(singleton,identity,state)
                   VALUES(true,%s,'idle') ON CONFLICT DO NOTHING RETURNING identity""",
                    (identity,),
                )
            ).fetchone()
            if row is None:
                raise RuntimeFault("WORKER_ALREADY_REGISTERED")

    async def heartbeat(self, identity: str) -> None:
        async with self._transaction() as conn:
            await self._one(
                conn,
                (
                    "UPDATE harness.worker_registration SET "
                    "heartbeat_at=clock_timestamp() WHERE singleton AND identity=%s "
                    "RETURNING identity"
                ),
                (identity,),
            )

    async def worker_unready(self, identity: str) -> None:
        async with self._transaction() as conn:
            await conn.execute(
                (
                    "UPDATE harness.worker_registration SET state='draining' WHERE "
                    "singleton AND identity=%s"
                ),
                (identity,),
            )

    async def _worker(self, conn: Connection, identity: str) -> Row:
        return await self._one(
            conn,
            "SELECT * FROM harness.worker_registration WHERE singleton AND identity=%s FOR UPDATE",
            (identity,),
        )

    async def _lock_owned_run(
        self, conn: Connection, identity: str, run_id: str
    ) -> tuple[Row, Row]:
        ref = await self._one(
            conn, "SELECT session_id FROM harness.runs WHERE run_id=%s", (run_id,)
        )
        session = await self._one(
            conn,
            "SELECT * FROM harness.sessions WHERE session_id=%s FOR UPDATE",
            (ref["session_id"],),
        )
        row = await self._one(
            conn,
            "SELECT *,clock_timestamp() AS now FROM harness.runs WHERE run_id=%s FOR UPDATE",
            (run_id,),
        )
        if row["owner_id"] != identity:
            raise RuntimeFault("EXECUTION_OWNER_MISMATCH")
        return session, row

    async def _claim_history(self, conn: Connection, actor: Principal, session: Row,
                              manifest: ReleaseManifest) -> list[Row]:
        if session.get("revoked_at") is not None or session.get("purged_at") is not None:
            raise RuntimeFault("SESSION_REVOKED", 403)
        require_memory_boundary(manifest)
        configuration = manifest.profiles.as_object().get("context")
        if configuration is None:
            return await (await conn.execute(
                "SELECT messages FROM harness.turns WHERE session_id=%s ORDER BY revision", (session["session_id"],))).fetchall()
        profile = ContextProfile.model_validate(configuration)
        if profile.history_mode == "current_only":
            return []
        history = await (await conn.execute("""SELECT t.run_id,t.revision,t.messages FROM harness.turns t
            JOIN harness.runs r USING(run_id) JOIN harness.commits c USING(run_id)
            WHERE t.session_id=%s AND r.status='succeeded' ORDER BY t.revision LIMIT %s""",
            (session["session_id"], profile.max_history_turns + 1))).fetchall()
        if len(history) > profile.max_history_turns:
            raise RuntimeFault("CONTEXT_HISTORY_LIMIT", 429)
        if len(history) != session["revision"] or any(turn["revision"] != i + 1 for i, turn in enumerate(history)):
            raise RuntimeFault("CONTEXT_HISTORY_INCOMPLETE", 409)
        for turn in history:
            source, source_session = await self._authorized_run(conn, actor, turn["run_id"])
            await self._authorize_public_resource(conn, actor, source, source_session)
        return history

    async def claim(self, identity: str) -> ClaimedRun | None:
        async with self._transaction() as conn:
            worker = await self._worker(conn, identity)
            if worker["active_run_id"] or worker["state"] != "idle":
                raise RuntimeFault("WORKER_SLOT_BUSY")
            ref = await (
                await conn.execute(
                    "SELECT run_id,session_id FROM harness.runs WHERE status='queued'"
                    + (
                        " AND manifest_hash=ANY(%s)"
                        if self.installed_manifest_hashes is not None
                        else ""
                    )
                    + " ORDER BY created_at,run_id LIMIT 1",
                    (sorted(self.installed_manifest_hashes),)
                    if self.installed_manifest_hashes is not None
                    else (),
                )
            ).fetchone()
            if ref is None:
                return None
            session = await self._one(
                conn,
                "SELECT * FROM harness.sessions WHERE session_id=%s FOR UPDATE",
                (ref["session_id"],),
            )
            row = await self._one(
                conn,
                "SELECT *,clock_timestamp() AS now FROM harness.runs WHERE run_id=%s FOR UPDATE",
                (ref["run_id"],),
            )
            if row["status"] != "queued":
                return None
            if (
                self.installed_manifest_hashes is not None
                and row["manifest_hash"] not in self.installed_manifest_hashes
            ):
                raise RuntimeFault("EXECUTABLE_RELEASE_NOT_INSTALLED", 503)
            if row["now"] >= min(row["queue_deadline"], row["expires_at"]):
                await self._terminate(conn, row, "expired", "QUEUE_EXPIRED")
                return None
            actor = Principal.model_validate(row["principal"])
            if session["evaluation_context_id"]:
                try:
                    await self._evaluation(conn, actor, session["evaluation_context_id"])
                except RuntimeFault:
                    await self._terminate(conn, row, "failed", "EVALUATION_REVOKED_OR_EXPIRED")
                    return None
            release = await self._release(conn, actor, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            if row["manifest_hash"] != manifest.content_hash():
                raise RuntimeFault("STORED_MANIFEST_MISMATCH")
            remaining_ms = self.profile.execution_timeout_seconds * 1000 - row["execution_used_ms"]
            if remaining_ms <= 0:
                await self._terminate(conn, row, "expired", "EXECUTION_BUDGET_EXHAUSTED")
                return None
            deadline = min(row["expires_at"], row["now"] + timedelta(milliseconds=remaining_ms))
            await conn.execute(
                (
                    "UPDATE harness.runs SET "
                    "status='running',owner_id=%s,execution_started_at=clock_timestamp"
                    "(),execution_deadline=%s,revision=revision+1 WHERE run_id=%s"
                ),
                (identity, deadline, row["run_id"]),
            )
            await conn.execute(
                (
                    "UPDATE harness.worker_registration SET "
                    "state='running',active_run_id=%s WHERE singleton AND identity=%s"
                ),
                (row["run_id"], identity),
            )
            try:
                history = await self._claim_history(conn, actor, session, manifest)
            except RuntimeFault as exc:
                if exc.status >= 500:
                    raise
                # 未执行图即可安全收尾；任务级历史拒绝不应使单Worker退出。
                claimed = await self._one(conn, "SELECT *,clock_timestamp() AS now FROM harness.runs WHERE run_id=%s", (row["run_id"],))
                await self._terminate(conn, claimed, "failed", exc.code)
                await conn.execute("UPDATE harness.worker_registration SET active_run_id=NULL,state='idle' WHERE singleton AND identity=%s",
                                   (identity,))
                return None
            return ClaimedRun(
                run_id=row["run_id"],
                session_id=session["session_id"],
                principal=actor,
                request=RunRequest.model_validate(row["request"]),
                manifest=manifest,
                execution_deadline=deadline,
                history=JsonSnapshot.capture(
                    [message for turn in history for message in turn["messages"]]
                ),
                suspension=JsonSnapshot.capture(row["suspension"])
                if row.get("suspension")
                else None,
            )

    async def started(self, identity: str, run: ClaimedRun, observed_hash: str) -> None:
        async with self._transaction() as conn:
            _, row = await self._lock_owned_run(conn, identity, run.run_id)
            if row["status"] != "running" or observed_hash != row["manifest_hash"]:
                raise RuntimeFault("OBSERVED_MANIFEST_MISMATCH")
            await self._event(
                conn,
                run.run_id,
                "run.started",
                {"observed_manifest_hash": observed_hash, "execution_identity": identity},
            )

    async def continue_run(self, identity: str, run_id: str) -> bool:
        async with self._transaction() as conn:
            session, row = await self._lock_owned_run(conn, identity, run_id)
            if row["status"] != "running" or row["recovery_required"]:
                return False
            if session.get("revoked_at") is not None or session.get("purged_at") is not None:
                return False
            if session["evaluation_context_id"]:
                try:
                    await self._evaluation(
                        conn,
                        Principal.model_validate(row["principal"]),
                        session["evaluation_context_id"],
                    )
                except RuntimeFault:
                    return False
            return not row["cancel_requested"] and row["now"] < min(
                row["execution_deadline"], row["expires_at"]
            )

    async def _terminate(self, conn: Connection, row: Row, status: str, code: str) -> None:
        await self._lifecycle.terminate(conn, row, status, code)

    async def complete(self, identity: str, run: ClaimedRun, candidate: AnswerCandidate) -> None:
        digest = canonical_hash(candidate.validation_input())
        report = candidate.validation
        if (
            report.input_hash != digest
            or report.validator_ref != run.manifest.spec.completion_validator_ref
            or report.verdict not in {"pass", "needs_clarification"}
            or not report.checks
            or not all(report.checks.values())
        ):
            raise RuntimeFault("COMPLETION_VALIDATION_FAILED")
        if candidate.task_outcome == "achieved" and (
            candidate.answer_kind != "answer" or report.verdict != "pass"
        ):
            raise RuntimeFault("COMPLETION_OUTCOME_MISMATCH")
        async with self._transaction() as conn:
            session, row = await self._lock_owned_run(conn, identity, run.run_id)
            if session.get("revoked_at") is not None or session.get("purged_at") is not None:
                raise RuntimeFault("SESSION_REVOKED", 403)
            if row["status"] == "succeeded":
                existing = await self._one(
                    conn,
                    "SELECT candidate_hash FROM harness.commits WHERE run_id=%s",
                    (run.run_id,),
                )
                if existing["candidate_hash"] != digest:
                    raise RuntimeFault("COMMIT_CONFLICT")
                return
            if row["status"] != "running" or row["recovery_required"]:
                raise RuntimeFault("RUN_NOT_COMMITTABLE")
            unresolved = await (
                await conn.execute(
                    (
                        "SELECT 1 FROM harness.operations WHERE run_id=%s AND "
                        "(state<>'succeeded' OR output_error IS NOT NULL)"
                    ),
                    (run.run_id,),
                )
            ).fetchone()
            if unresolved:
                raise RuntimeFault("WRITE_EVIDENCE_NOT_COMPLETE")
            read_errors = await (
                await conn.execute(
                    "SELECT 1 FROM harness.read_operations WHERE run_id=%s AND state<>'succeeded'",
                    (run.run_id,),
                )
            ).fetchone()
            if read_errors:
                raise RuntimeFault("READ_EVIDENCE_NOT_COMPLETE")
            if session["evaluation_context_id"]:
                await self._evaluation(conn, run.principal, session["evaluation_context_id"])
            if row["cancel_requested"] or row["now"] >= row["execution_deadline"]:
                await self._terminate(
                    conn,
                    row,
                    "cancelled" if row["cancel_requested"] else "expired",
                    "CANCELLED" if row["cancel_requested"] else "EXECUTION_EXPIRED",
                )
                return
            if self.answer_authority is not None:
                await self.answer_authority(conn, identity, run, candidate)
            commit_id = uuid4().hex
            result = {
                "answer": candidate.answer,
                "answer_kind": candidate.answer_kind,
                "task_outcome": candidate.task_outcome,
                "task": candidate.task.model_dump(mode="json"),
            }
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.commits(commit_id,run_id,candidate_hash,validation) "
                    "VALUES(%s,%s,%s,%s)"
                ),
                (commit_id, run.run_id, digest, Jsonb(report.model_dump(mode="json"))),
            )
            summary_configuration = run.manifest.profiles.as_object().get("summary_commit")
            if summary_configuration is not None:
                summary_policy = SummaryCommitProfile.model_validate(summary_configuration)
                if self.summary_authority is None or self.summary_authority_ref != summary_policy.version:
                    raise RuntimeFault("SUMMARY_COMMITTER_NOT_INSTALLED", 503)
                await self.summary_authority(conn, identity, run, candidate, commit_id)
            # 当前传输完整回答后再公开一个 delta，不伪装成供应商逐 Token 流。
            await self._event(
                conn,
                run.run_id,
                "answer.started",
                {
                    "draft_id": candidate.draft_id,
                    "producer": "model",
                    "model_call_id": candidate.model_call_id,
                    "model_attempt_id": candidate.model_attempt_id,
                },
            )
            await self._event(
                conn,
                run.run_id,
                "answer.delta",
                {
                    "draft_id": candidate.draft_id,
                    "model_attempt_id": candidate.model_attempt_id,
                    "delta": candidate.answer,
                },
            )
            await conn.execute(
                (
                    "INSERT INTO harness.turns(session_id,revision,run_id,messages) "
                    "VALUES(%s,%s,%s,%s)"
                ),
                (
                    run.session_id,
                    session["revision"] + 1,
                    run.run_id,
                    Jsonb(
                        [
                            {"role": "user", "content": run.request.message},
                            {"role": "assistant", "content": candidate.answer},
                        ]
                    ),
                ),
            )
            await conn.execute(
                (
                    "UPDATE harness.sessions SET "
                    "revision=revision+1,active_run_id=NULL WHERE session_id=%s"
                ),
                (run.session_id,),
            )
            await conn.execute(
                (
                    "UPDATE harness.runs SET "
                    "status='succeeded',revision=revision+1,result=%s,execution_used_m"
                    "s=execution_used_ms+greatest(0,extract(epoch FROM "
                    "clock_timestamp()-execution_started_at)*1000)::bigint,execution_s"
                    "tarted_at=NULL WHERE run_id=%s"
                ),
                (Jsonb(result), run.run_id),
            )
            await self._event(
                conn,
                run.run_id,
                "run.completed",
                {
                    "committed_draft_id": candidate.draft_id,
                    "commit_id": commit_id,
                    "result_hash": canonical_hash(result),
                },
            )

    async def fail(self, identity: str, run_id: str, code: str) -> None:
        async with self._transaction() as conn:
            _, row = await self._lock_owned_run(conn, identity, run_id)
            if row["status"] == "running":
                status = "cancelled" if row["cancel_requested"] else "failed"
                if row["now"] >= min(row["execution_deadline"], row["expires_at"]):
                    status = "expired"
                    code = "EXECUTION_EXPIRED"
                await self._terminate(
                    conn, row, status, "CANCELLED" if row["cancel_requested"] else code
                )

    async def finish_slot(self, identity: str) -> None:
        async with self._transaction() as conn:
            worker = await self._worker(conn, identity)
            if worker["active_run_id"]:
                row = await self._one(
                    conn,
                    "SELECT status FROM harness.runs WHERE run_id=%s",
                    (worker["active_run_id"],),
                )
                if row["status"] in {"running", "queued"}:
                    raise RuntimeFault("EXECUTION_NOT_DRAINED")
            await conn.execute(
                (
                    "UPDATE harness.worker_registration SET "
                    "active_run_id=NULL,state='idle' WHERE singleton AND identity=%s"
                ),
                (identity,),
            )

    async def release_worker(self, identity: str) -> None:
        async with self._transaction() as conn:
            worker = await self._worker(conn, identity)
            if worker["active_run_id"] is not None:
                raise RuntimeFault("EXECUTION_NOT_DRAINED")
            await conn.execute(
                "DELETE FROM harness.worker_registration WHERE singleton AND identity=%s",
                (identity,),
            )

    async def health(self) -> dict[str, Any]:
        async with self._transaction() as conn:
            schema = await self._one(
                conn, "SELECT max(version) AS version FROM harness.schema_versions"
            )
            worker = await (
                await conn.execute(
                    (
                        "SELECT *,clock_timestamp()-heartbeat_at < %s * interval '1 "
                        "second' AS fresh FROM harness.worker_registration WHERE singleton"
                    ),
                    (self.profile.heartbeat_stale_seconds,),
                )
            ).fetchone()
            return {
                "database": True,
                "schema": schema["version"] == 9,
                "worker_registered": worker is not None,
                "worker_fresh": bool(
                    worker and worker["fresh"] and worker["state"] in {"idle", "running"}
                ),
                "worker_heartbeat_at": worker["heartbeat_at"] if worker else None,
            }
