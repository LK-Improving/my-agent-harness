"""事务内评测准入、固定绑定和撤销；不调用模型或工具。"""

from datetime import timedelta
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import JsonValue
from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, Row
from harness.contracts.evaluation_contexts import EvaluationRevocationRequest
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import EvaluationRequest, Principal, RuntimeFault

if TYPE_CHECKING:
    from harness.adapters.runtime_postgres import PostgresRuntimeStore


class PostgresEvaluationContexts:
    def __init__(self, store: "PostgresRuntimeStore") -> None:
        self.store = store

    async def row(self, conn: Connection, actor: Principal, context_id: str,
                  *, exclusive: bool = False) -> Row:
        self.store._role(actor, "evaluator")
        return await self.store._one(conn,
            "SELECT * FROM harness.evaluation_contexts WHERE context_id=%s AND tenant_id=%s "
            "AND subject_id=%s AND application_id=%s" + (" FOR UPDATE" if exclusive else " FOR SHARE"),
            (context_id, actor.tenant_id, actor.subject_id, actor.application_id))

    @staticmethod
    def binding(row: Row) -> EvaluationContext:
        context = EvaluationContext.model_validate(row["context"])
        if (context.context_id != row["context_id"] or context.tenant_id != row["tenant_id"]
                or context.subject_id != row["subject_id"]
                or context.candidate_release_id != row["release_id"]
                or context.expires_at != row["expires_at"]
                or context.application_id not in {None, row["application_id"]}
                or (context.application_id is None and context.admission_version != "legacy-evaluation@1")):
            raise RuntimeFault("EVALUATION_BINDING_CORRUPT", 409)
        return context.model_copy(update={"application_id": row["application_id"],
                                           "revoked_at": row["revoked_at"]})

    @staticmethod
    def require_binding(context: EvaluationContext, session: Row, run: Row | None = None) -> None:
        if (context.candidate_release_id != session["release_id"]
                or context.agent_id != session["agent_id"]):
            raise RuntimeFault("EVALUATION_SESSION_BINDING_MISMATCH", 403)
        if run is not None:
            actor = Principal.model_validate(run["principal"])
            if (run["manifest_hash"] != context.manifest_hash
                    or run["environment_profile_hash"] != context.environment_profile_hash
                    or actor.tenant_id != context.tenant_id or actor.subject_id != context.subject_id
                    or actor.application_id != context.application_id
                    or actor.resource_scopes != context.resource_scopes):
                raise RuntimeFault("EVALUATION_RUN_BINDING_MISMATCH", 403)

    async def authorize(self, conn: Connection, actor: Principal, context_id: str,
                        *, exclusive: bool = False, require_installed: bool = True) -> EvaluationContext:
        row = await self.row(conn, actor, context_id, exclusive=exclusive)
        context = self.binding(row)
        # 时钟必须在行锁取得之后读取，不能使用等待锁前已计算的时间。
        now = (await self.store._one(conn, "SELECT clock_timestamp() AS now"))["now"]
        if context.revoked_at is not None or now >= context.expires_at:
            raise RuntimeFault("EVALUATION_REVOKED_OR_EXPIRED", 403)
        profile = self.store.profile.evaluation
        if (not context.resource_scopes <= actor.resource_scopes
                or not context.resource_scopes <= profile.test_resource_scopes):
            raise RuntimeFault("EVALUATION_SCOPE_REVOKED", 403)
        if (profile.candidate_release_ids is not None
                and context.candidate_release_id not in profile.candidate_release_ids):
            raise RuntimeFault("EVALUATION_CANDIDATE_NOT_ALLOWED", 403)
        release = await self.store._release(conn, actor, context.candidate_release_id)
        manifest = ReleaseManifest.model_validate(release["manifest"])
        if (manifest.content_hash() != context.manifest_hash
                or release["manifest_hash"] != context.manifest_hash
                or manifest.environment_profile_hash != context.environment_profile_hash
                or manifest.spec.agent_id != context.agent_id):
            raise RuntimeFault("EVALUATION_MANIFEST_MISMATCH", 403)
        installed = self.store.installed_manifest_hashes
        if require_installed and installed is not None and context.manifest_hash not in installed:
            raise RuntimeFault("EXECUTABLE_RELEASE_NOT_INSTALLED", 503)
        return context

    async def create(self, actor: Principal, request: EvaluationRequest) -> EvaluationContext:
        self.store._role(actor, "evaluation_admin")
        self.store._role(actor, "evaluator")
        profile = self.store.profile.evaluation
        if (not request.resource_scopes <= actor.resource_scopes
                or not request.resource_scopes <= profile.test_resource_scopes):
            raise RuntimeFault("EVALUATION_TEST_SCOPE_REQUIRED", 403)
        if (profile.candidate_release_ids is not None
                and request.candidate_release_id not in profile.candidate_release_ids):
            raise RuntimeFault("EVALUATION_CANDIDATE_NOT_ALLOWED", 403)
        if (request.limits.max_sessions > profile.limits.max_sessions
                or request.limits.max_runs > profile.limits.max_runs):
            raise RuntimeFault("EVALUATION_LIMIT_ESCALATION", 403)
        async with self.store._transaction() as conn:
            # 租户准入锁串行化同主体创建，避免并发超卖有效context数。
            await self.store._admit(conn, actor)
            release = await self.store._release(conn, actor, request.candidate_release_id)
            manifest = ReleaseManifest.model_validate(release["manifest"])
            if release["manifest_hash"] != manifest.content_hash():
                raise RuntimeFault("STORED_MANIFEST_MISMATCH", 409)
            installed = self.store.installed_manifest_hashes
            if installed is not None and manifest.content_hash() not in installed:
                raise RuntimeFault("EXECUTABLE_RELEASE_NOT_INSTALLED", 503)
            now = (await self.store._one(conn, "SELECT clock_timestamp() AS now"))["now"]
            if not now < request.expires_at <= now + timedelta(seconds=profile.max_lifetime_seconds):
                raise RuntimeFault("INVALID_EVALUATION_EXPIRY", 422)
            count = await self.store._one(conn,
                "SELECT count(*) AS used FROM harness.evaluation_contexts WHERE tenant_id=%s "
                "AND application_id=%s AND subject_id=%s AND revoked_at IS NULL "
                "AND expires_at>clock_timestamp()",
                (actor.tenant_id, actor.application_id, actor.subject_id))
            if count["used"] >= profile.max_valid_contexts_per_subject:
                raise RuntimeFault("EVALUATION_CONTEXT_LIMIT", 429)
            context = EvaluationContext(context_id=uuid4().hex, tenant_id=actor.tenant_id,
                subject_id=actor.subject_id, application_id=actor.application_id,
                agent_id=manifest.spec.agent_id, candidate_release_id=manifest.spec.release_id,
                manifest_hash=manifest.content_hash(), dataset_hash=request.dataset_hash,
                gate_policy_version=request.gate_policy_version,
                environment_profile_hash=manifest.environment_profile_hash,
                resource_scopes=request.resource_scopes, expires_at=request.expires_at,
                admission_version=profile.version, limits=request.limits)
            await conn.execute("INSERT INTO harness.evaluation_contexts "
                "(context_id,tenant_id,subject_id,application_id,release_id,context,expires_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (context.context_id, actor.tenant_id, actor.subject_id, actor.application_id,
                 context.candidate_release_id, Jsonb(context.model_dump(mode="json")), context.expires_at))
            await self.store._audit(conn, actor, "evaluation.created", context.context_id)
            return context

    async def consume(self, conn: Connection, context: EvaluationContext,
                      kind: Literal["session", "run"]) -> None:
        # 调用者已持有context独占锁；计数与资源创建同事务，失败回滚，不在恢复时重计。
        if kind == "session":
            sql = ("UPDATE harness.evaluation_contexts SET session_count=session_count+1 "
                   "WHERE context_id=%s AND session_count<%s RETURNING context_id")
            limit = min(context.limits.max_sessions, self.store.profile.evaluation.limits.max_sessions)
        else:
            sql = ("UPDATE harness.evaluation_contexts SET run_count=run_count+1 "
                   "WHERE context_id=%s AND run_count<%s RETURNING context_id")
            limit = min(context.limits.max_runs, self.store.profile.evaluation.limits.max_runs)
        if await (await conn.execute(sql, (context.context_id, limit))).fetchone() is None:
            raise RuntimeFault("EVALUATION_RESOURCE_LIMIT", 429)

    async def view(self, conn: Connection, actor: Principal, context_id: str) -> dict[str, JsonValue]:
        row = await self.row(conn, actor, context_id)
        context = self.binding(row)
        now = (await self.store._one(conn, "SELECT clock_timestamp() AS now"))["now"]
        status = "revoked" if context.revoked_at else "expired" if now >= context.expires_at else "active"
        return {"context": context.model_dump(mode="json"), "status": status,
                "session_count": row["session_count"], "run_count": row["run_count"]}

    async def inspect(self, actor: Principal, context_id: str) -> dict[str, JsonValue]:
        async with self.store._transaction() as conn:
            return await self.view(conn, actor, context_id)

    async def revoke(self, actor: Principal, context_id: str,
                     request: EvaluationRevocationRequest) -> dict[str, JsonValue]:
        async with self.store._transaction() as conn:
            await self.store._admit(conn, actor)
            row = await self.row(conn, actor, context_id, exclusive=True)
            context = self.binding(row)
            if request.expected_manifest_hash != context.manifest_hash:
                raise RuntimeFault("EVALUATION_MANIFEST_MISMATCH", 409)
            digest = canonical_hash(request.model_dump(mode="json", exclude={"request_id"}))
            old = await (await conn.execute("SELECT request_hash FROM harness.evaluation_revocations "
                "WHERE context_id=%s AND request_id=%s", (context_id, request.request_id))).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    raise RuntimeFault("EVALUATION_REVOCATION_CONFLICT", 409)
                return await self.view(conn, actor, context_id)
            await conn.execute("INSERT INTO harness.evaluation_revocations "
                "(context_id,request_id,request_hash) VALUES(%s,%s,%s)",
                (context_id, request.request_id, digest))
            await conn.execute("UPDATE harness.evaluation_contexts SET "
                "revoked_at=COALESCE(revoked_at,clock_timestamp()) WHERE context_id=%s", (context_id,))
            await self.store._audit(conn, actor, "evaluation.revoked", context_id)
            return await self.view(conn, actor, context_id)
