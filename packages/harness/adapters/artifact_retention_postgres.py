"""PG产物删除意图与保留判定；清理通过独立持久任务显式推进。"""

from psycopg.types.json import Jsonb
from pydantic import ValidationError

from harness.adapters.artifacts_postgres import PostgresArtifacts
from harness.adapters.postgres.connection import Connection, Row
from harness.context.artifact_retention import retention_view
from harness.contracts.artifacts import (
    ArtifactDeletionRequest,
    ArtifactRead,
    ArtifactRetentionFacts,
    ArtifactRetentionPolicy,
    ArtifactRetentionView,
)
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import Principal, RuntimeFault


class PostgresArtifactRetention:
    def __init__(self, artifacts: PostgresArtifacts, policy: ArtifactRetentionPolicy, *, executor_installed: bool = False) -> None:
        self.artifacts, self.execution, self.policy = artifacts, artifacts.execution, policy
        self.executor_installed = executor_installed

    @staticmethod
    def _validate_ref(artifact_ref: str) -> None:
        try:
            ArtifactRead(artifact_ref=artifact_ref)
        except ValidationError:
            raise RuntimeFault("ARTIFACT_REF_INVALID", 422) from None

    async def _lock(self, conn: Connection, actor: Principal, session_id: str,
                    artifact_ref: str) -> Row:
        # 不接受跨主体管理捷径；评测会话仍执行原授权/有效期校验。
        await self.execution._session(conn, actor, session_id, lock=True)
        ref = await self.execution._one(
            conn, "SELECT run_id FROM harness.artifacts WHERE artifact_ref=%s AND session_id=%s",
            (artifact_ref, session_id),
        )
        await self.execution._one(
            conn, "SELECT run_id FROM harness.runs WHERE run_id=%s AND session_id=%s FOR UPDATE",
            (ref["run_id"], session_id),
        )
        return await self.execution._one(
            conn, """SELECT artifact_ref,session_id,run_id,operation_id,content_hash,expires_at,revoked_at,purged_at
                FROM harness.artifacts WHERE artifact_ref=%s AND session_id=%s FOR UPDATE""",
            (artifact_ref, session_id),
        )

    async def _view(self, conn: Connection, actor: Principal, row: Row) -> ArtifactRetentionView:
        facts = await self.execution._one(conn, """
            WITH target AS (SELECT %s::text AS run_id,%s::text AS session_id,%s::text AS ref,%s::text AS operation_id)
            SELECT clock_timestamp() AS checked_at,
                r.status NOT IN ('succeeded','partial','failed','cancelled','expired') AS source_run_active,
                s.active_run_id IS NOT NULL AS session_active,
                EXISTS(SELECT 1 FROM harness.operations o JOIN harness.runs x USING(run_id)
                    WHERE x.session_id=t.session_id AND o.state IN ('prepared','dispatched','unknown')) AS unresolved_write,
                EXISTS(SELECT 1 FROM harness.approvals a JOIN harness.operations o USING(operation_id)
                    JOIN harness.runs x ON x.run_id=o.run_id
                    WHERE x.session_id=t.session_id AND a.state='pending') AS pending_approval,
                EXISTS(SELECT 1 FROM harness.evaluation_contexts e WHERE e.context_id=s.evaluation_context_id
                    AND e.revoked_at IS NULL AND e.expires_at>clock_timestamp()) AS active_evaluation,
                (EXISTS(SELECT 1 FROM harness.read_operations o WHERE o.run_id=t.run_id
                    AND o.operation_id=t.operation_id AND o.receipt IS NOT NULL)
                 OR EXISTS(SELECT 1 FROM harness.operations o WHERE o.run_id=t.run_id
                    AND o.operation_id=t.operation_id AND o.receipt IS NOT NULL)) AS source_receipt_retained,
                EXISTS(SELECT 1 FROM harness.read_operations o WHERE o.tool_name='read_artifact'
                    AND o.arguments->>'artifact_ref'=t.ref) AS readback_reference_retained,
                EXISTS(SELECT 1 FROM harness.model_calls m JOIN harness.runs x USING(run_id)
                    WHERE x.session_id=t.session_id) AS model_request_retained,
                EXISTS(SELECT 1 FROM harness.turns h WHERE h.session_id=t.session_id) AS session_history_retained,
                r.owner_id IS NOT NULL AS checkpoint_scope_retained,
                (EXISTS(SELECT 1 FROM harness.artifact_deletion_requests d WHERE d.artifact_ref=t.ref)
                 OR EXISTS(SELECT 1 FROM harness.cleanup_jobs j WHERE j.session_id=t.session_id
                    AND j.request->>'reason'='delete' AND (j.kind='session' OR j.request->>'artifact_ref'=t.ref))) AS deletion_requested
            FROM target t JOIN harness.runs r ON r.run_id=t.run_id
                JOIN harness.sessions s ON s.session_id=t.session_id
            """, (row["run_id"], row["session_id"], row["artifact_ref"], row["operation_id"]))
        denial = None
        try:
            await self.artifacts._authorized(conn, actor, row["session_id"], ArtifactRead(artifact_ref=row["artifact_ref"]))
        except RuntimeFault as exc:
            # 管理视图仅披露可读性；依赖、内容损坏或profile故障仍须阻断。
            if exc.status not in {403, 404, 410}:
                raise
            denial = exc.code
        view = retention_view(
            artifact_ref=row["artifact_ref"], facts=ArtifactRetentionFacts.model_validate({
                name: facts[name] for name in ArtifactRetentionFacts.model_fields
            }), policy=self.policy, checked_at=facts["checked_at"], expires_at=row["expires_at"],
            revoked_at=row["revoked_at"], deletion_requested=facts["deletion_requested"], read_denial_code=denial,
            executor_installed=self.executor_installed, purged_at=row.get("purged_at"),
        )
        await self.execution._audit(conn, actor, "artifact.retention_checked", row["artifact_ref"])
        return view

    async def inspect(self, actor: Principal, session_id: str, artifact_ref: str) -> ArtifactRetentionView:
        self._validate_ref(artifact_ref)
        actor = await self.artifacts._current(actor)
        async with self.execution._transaction() as conn:
            row = await self._lock(conn, actor, session_id, artifact_ref)
            return await self._view(conn, actor, row)

    async def request_deletion(self, actor: Principal, session_id: str, artifact_ref: str,
                               request: ArtifactDeletionRequest) -> ArtifactRetentionView:
        self._validate_ref(artifact_ref)
        actor = await self.artifacts._current(actor)
        digest = canonical_hash({"artifact_ref": artifact_ref, "request": request.model_dump(mode="json")})
        async with self.execution._transaction() as conn:
            row = await self._lock(conn, actor, session_id, artifact_ref)
            old = await (await conn.execute(
                "SELECT request_hash FROM harness.artifact_deletion_requests WHERE session_id=%s AND request_id=%s",
                (session_id, request.request_id),
            )).fetchone()
            if old is not None and old["request_hash"] != digest:
                raise RuntimeFault("ARTIFACT_DELETE_REQUEST_CONFLICT", 409)
            if row["content_hash"] != request.expected_content_hash:
                raise RuntimeFault("ARTIFACT_DELETE_CONTENT_MISMATCH", 409)
            if old is None:
                await conn.execute("""INSERT INTO harness.artifact_deletion_requests
                    (session_id,request_id,artifact_ref,request_hash,expected_content_hash,actor)
                    VALUES(%s,%s,%s,%s,%s,%s)""",
                    (session_id, request.request_id, artifact_ref, digest, request.expected_content_hash,
                     Jsonb(actor.model_dump(mode="json"))),
                )
                # 即使活动证据阻断清理，也先撤权；不更新原始到期或正文。
                row = await self.execution._one(conn, """UPDATE harness.artifacts SET
                    revoked_at=coalesce(revoked_at,clock_timestamp()) WHERE artifact_ref=%s
                    RETURNING artifact_ref,session_id,run_id,operation_id,content_hash,expires_at,revoked_at""",
                    (artifact_ref,),
                )
                await self.execution._audit(conn, actor, "artifact.deletion_requested", artifact_ref)
            else:
                if row["revoked_at"] is None:
                    raise RuntimeFault("ARTIFACT_DELETE_STATE_CORRUPT", 409)
                await self.execution._audit(conn, actor, "artifact.deletion_replayed", artifact_ref)
        # 意图/撤权已独立提交；后续保留判定故障不能重新开放读取。
        return await self.inspect(actor, session_id, artifact_ref)
