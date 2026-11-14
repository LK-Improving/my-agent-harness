"""M2 写入账本；外部调用不占用本模块的数据库事务。"""

from datetime import timedelta
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, Row
from harness.adapters.postgres.events import PostgresEventJournal
from harness.adapters.postgres.lifecycle import PostgresRunLifecycle
from harness.adapters.postgres.mapping import write_operation
from harness.adapters.runtime_postgres import PostgresRuntimeStore
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AdmissionProfile, ClaimedRun, Principal, RuntimeFault
from harness.contracts.specs import ToolSpec
from harness.contracts.state import ReceiptState, ToolReceipt
from harness.contracts.writes import (
    ApprovalDecisionRequest,
    DispatchPermit,
    ExternalWriteResult,
    ReconcileRequest,
    ReconciliationPermit,
    ReconciliationResult,
    SuspensionProof,
    WriteOperation,
    WritePolicy,
    WriteProposal,
)


class PostgresWriteStore(PostgresRuntimeStore):
    def __init__(
        self, dsn: str, profile: AdmissionProfile, writes: WritePolicy | None = None
    ) -> None:
        super().__init__(dsn, profile)
        self.writes = writes or WritePolicy()
        self._events = PostgresEventJournal(writes=True)
        self._lifecycle = PostgresRunLifecycle(self._events)

    _operation = staticmethod(write_operation)

    async def _op_lock(self, conn: Connection, operation_id: str) -> tuple[Row, Row, Row]:
        ref = await self._one(
            conn, "SELECT run_id FROM harness.operations WHERE operation_id=%s", (operation_id,)
        )
        run_ref = await self._one(
            conn, "SELECT session_id FROM harness.runs WHERE run_id=%s", (ref["run_id"],)
        )
        session = await self._one(
            conn,
            "SELECT * FROM harness.sessions WHERE session_id=%s FOR UPDATE",
            (run_ref["session_id"],),
        )
        run = await self._one(
            conn,
            "SELECT *,clock_timestamp() AS now FROM harness.runs WHERE run_id=%s FOR UPDATE",
            (ref["run_id"],),
        )
        operation = await self._one(
            conn,
            "SELECT * FROM harness.operations WHERE operation_id=%s FOR UPDATE",
            (operation_id,),
        )
        return session, run, operation

    def _resource_authority(self, actor: Principal, op: Row, role: str) -> None:
        self._role(actor, role)
        subject = Principal.model_validate(op["principal"])
        if actor.tenant_id != subject.tenant_id or actor.application_id != subject.application_id:
            raise RuntimeFault("NOT_FOUND", 404)
        if not subject.resource_scopes <= actor.resource_scopes:
            raise RuntimeFault("RESOURCE_SCOPE_REVOKED", 403)

    async def _check_execution(
        self, conn: Connection, actor: Principal, owner: str, run: Row, session: Row
    ) -> None:
        await self._session(conn, actor, session["session_id"])
        if session["evaluation_context_id"]:
            context = await self._evaluation(conn, actor, session["evaluation_context_id"])
            self._evaluations.require_binding(context, session, run)
        if run["status"] != "running" or run["owner_id"] != owner or run["recovery_required"]:
            raise RuntimeFault("WRITE_EXECUTION_NOT_OWNED")
        if run["cancel_requested"] or run["now"] >= min(
            run["expires_at"], run["execution_deadline"]
        ):
            raise RuntimeFault("WRITE_EXECUTION_STOPPED")

    async def prepare_write(
        self, actor: Principal, owner: str, proposal: WriteProposal
    ) -> WriteOperation:
        async with self._transaction() as conn:
            session, run = await self._lock_owned_run(conn, owner, proposal.run_id)
            await self._check_execution(conn, actor, owner, run, session)
            release = await self._release(conn, actor, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            tool = ToolSpec.model_validate(proposal.tool.as_object())
            binding = next(
                (
                    item
                    for item in manifest.spec.allowed_tools
                    if item.name == tool.name and item.version == tool.version
                ),
                None,
            )
            if (
                binding is None
                or manifest.tool_schema_hashes.as_object().get(f"{tool.name}@{tool.version}")
                != proposal.tool.content_hash
            ):
                raise RuntimeFault("TOOL_MANIFEST_MISMATCH", 403)
            if (
                tool.effect != "write"
                or tool.cost.billable
                or not tool.idempotency
                or not tool.idempotency.provider_supported
            ):
                raise RuntimeFault("WRITE_CAPABILITY_NOT_ENABLED", 422)
            old = await (
                await conn.execute(
                    "SELECT * FROM harness.operations WHERE run_id=%s AND logical_call_id=%s",
                    (run["run_id"], proposal.logical_call_id),
                )
            ).fetchone()
            if old:
                if (old["tool_hash"], old["args_hash"], old["action_key"], old["plan_hash"]) != (
                    proposal.tool.content_hash,
                    proposal.arguments.content_hash,
                    proposal.action_key,
                    proposal.plan_hash,
                ):
                    raise RuntimeFault("LOGICAL_CALL_CONFLICT")
                return self._operation(old)
            count = await self._one(
                conn,
                (
                    "SELECT count(*) AS count FROM harness.operations WHERE run_id=%s "
                    "AND tool->>'name'=%s"
                ),
                (run["run_id"], tool.name),
            )
            if count["count"] >= binding.max_operations_per_run:
                raise RuntimeFault("TOOL_OPERATION_LIMIT", 429)
            operation_id, approval_id = uuid4().hex, uuid4().hex
            expires = min(
                run["expires_at"], run["now"] + timedelta(seconds=self.writes.approval_ttl_seconds)
            )
            digest = canonical_hash(
                {
                    "operation_id": operation_id,
                    "tool_hash": proposal.tool.content_hash,
                    "args_hash": proposal.arguments.content_hash,
                    "plan_hash": proposal.plan_hash,
                    "principal": run["principal"],
                    "manifest_hash": run["manifest_hash"],
                    "policy": self.writes.model_dump(mode="json"),
                    "expires_at": expires.isoformat(),
                }
            )
            row = await self._one(
                conn,
                """INSERT INTO
                harness.operations(operation_id,run_id,logical_call_id,principal,tool,tool_hash,
                   arguments,args_hash,action_key,plan_hash,binding_hash,approval_id,state)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'prepared') RETURNING *""",
                (
                    operation_id,
                    run["run_id"],
                    proposal.logical_call_id,
                    Jsonb(run["principal"]),
                    Jsonb(proposal.tool.as_object()),
                    proposal.tool.content_hash,
                    Jsonb(proposal.arguments.as_object()),
                    proposal.arguments.content_hash,
                    proposal.action_key,
                    proposal.plan_hash,
                    digest,
                    approval_id,
                ),
            )
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.approvals(approval_id,operation_id,binding_hash,expires_a"
                    "t) VALUES(%s,%s,%s,%s)"
                ),
                (approval_id, operation_id, digest, expires),
            )
            await self._event(
                conn,
                run["run_id"],
                "approval.required",
                {"approval_id": approval_id, "operation_id": operation_id, "binding_hash": digest},
            )
            await self._audit(conn, actor, "write.prepared", operation_id)
            return self._operation(row)

    async def begin_dispatch(
        self, actor: Principal, owner: str, operation_id: str
    ) -> DispatchPermit:
        async with self._transaction() as conn:
            session, run, op = await self._op_lock(conn, operation_id)
            await self._check_execution(conn, actor, owner, run, session)
            fixed_actor = Principal.model_validate(op["principal"])
            if fixed_actor != actor:
                raise RuntimeFault("WRITE_PRINCIPAL_MISMATCH", 403)
            if op["state"] in {"dispatched", "unknown"}:
                raise RuntimeFault("WRITE_RECONCILIATION_REQUIRED")
            if op["state"] not in {"prepared", "not_executed"}:
                raise RuntimeFault("WRITE_NOT_DISPATCHABLE")
            approval = await self._one(
                conn,
                "SELECT * FROM harness.approvals WHERE approval_id=%s FOR UPDATE",
                (op["approval_id"],),
            )
            if (
                approval["binding_hash"] != op["binding_hash"]
                or run["now"] >= approval["expires_at"]
            ):
                raise RuntimeFault("APPROVAL_EXPIRED_OR_CHANGED")
            if approval["state"] == "pending":
                raise RuntimeFault("APPROVAL_PENDING")
            if approval["state"] != "approved":
                raise RuntimeFault("APPROVAL_REJECTED", 403)
            tool = ToolSpec.model_validate(op["tool"])
            if op["attempts"] >= tool.retry.max_attempts_per_operation:
                raise RuntimeFault("TOOL_ATTEMPT_LIMIT", 429)
            release = await self._release(conn, actor, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            if run["tool_dispatches"] >= manifest.spec.execution_limits.max_tool_dispatches:
                raise RuntimeFault("TOOL_DISPATCH_LIMIT", 429)
            if op["first_dispatch_at"] is not None:
                retention = tool.idempotency.retention_seconds if tool.idempotency else None
                if (
                    tool.retry.strategy != "same_key_after_reconcile"
                    or retention is None
                    or run["now"] >= op["first_dispatch_at"] + timedelta(seconds=retention)
                ):
                    raise RuntimeFault("PROVIDER_IDEMPOTENCY_EXPIRED")
            other = await (
                await conn.execute(
                    (
                        "SELECT 1 FROM harness.operations WHERE run_id=%s AND "
                        "operation_id<>%s AND state IN ('dispatched','unknown')"
                    ),
                    (run["run_id"], operation_id),
                )
            ).fetchone()
            if other:
                raise RuntimeFault("WRITE_RECONCILIATION_REQUIRED")
            await conn.execute(
                (
                    "INSERT INTO harness.business_action_guards VALUES(%s,%s,%s,%s) "
                    "ON CONFLICT DO NOTHING"
                ),
                (actor.tenant_id, actor.application_id, op["action_key"], operation_id),
            )
            guard = await self._one(
                conn,
                (
                    "SELECT operation_id FROM harness.business_action_guards WHERE "
                    "tenant_id=%s AND application_id=%s AND action_key=%s"
                ),
                (actor.tenant_id, actor.application_id, op["action_key"]),
            )
            if guard["operation_id"] != operation_id:
                raise RuntimeFault("BUSINESS_ACTION_BLOCKED")
            attempt_id = uuid4().hex
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.tool_attempts(attempt_id,operation_id,ordinal,state) "
                    "VALUES(%s,%s,%s,'dispatched')"
                ),
                (attempt_id, operation_id, op["attempts"] + 1),
            )
            updated = await self._one(
                conn,
                (
                    "UPDATE harness.operations SET "
                    "state='dispatched',attempts=attempts+1,first_dispatch_at=coalesce"
                    "(first_dispatch_at,clock_timestamp()) WHERE operation_id=%s "
                    "RETURNING *"
                ),
                (operation_id,),
            )
            await conn.execute(
                "UPDATE harness.runs SET tool_dispatches=tool_dispatches+1 WHERE run_id=%s",
                (run["run_id"],),
            )
            await self._audit(conn, actor, "write.dispatched", attempt_id)
            return DispatchPermit(
                operation=self._operation(updated),
                attempt_id=attempt_id,
                deadline_at=min(
                    run["execution_deadline"], run["now"] + timedelta(milliseconds=tool.timeout_ms)
                ),
            )

    async def record_write_result(
        self, permit: DispatchPermit, result: ExternalWriteResult, output_error: str | None
    ) -> ToolReceipt:
        async with self._transaction() as conn:
            _, _, op = await self._op_lock(conn, permit.operation.operation_id)
            attempt = await self._one(
                conn,
                (
                    "SELECT * FROM harness.tool_attempts WHERE attempt_id=%s AND "
                    "operation_id=%s FOR UPDATE"
                ),
                (permit.attempt_id, op["operation_id"]),
            )
            receipt = ToolReceipt(
                operation_id=op["operation_id"],
                attempt_id=permit.attempt_id,
                status=ReceiptState.SUCCEEDED,
                args_hash=op["args_hash"],
                external_reference=result.external_reference,
                data=result.data,
                error_code=output_error,
                cost_status="not_billable",
            )
            if op["state"] == "succeeded":
                if op["receipt"] != receipt.model_dump(mode="json"):
                    raise RuntimeFault("WRITE_RECEIPT_CONFLICT")
                return receipt
            if op["state"] not in {"dispatched", "unknown"} or attempt["ordinal"] != op["attempts"]:
                raise RuntimeFault("STALE_TOOL_ATTEMPT")
            await conn.execute(
                (
                    "UPDATE harness.tool_attempts SET "
                    "state='succeeded',finished_at=clock_timestamp() WHERE "
                    "attempt_id=%s"
                ),
                (permit.attempt_id,),
            )
            await conn.execute(
                (
                    "UPDATE harness.operations SET "
                    "state='succeeded',receipt=%s,output_error=%s WHERE "
                    "operation_id=%s"
                ),
                (Jsonb(receipt.model_dump(mode="json")), output_error, op["operation_id"]),
            )
            await self._event(
                conn,
                op["run_id"],
                "tool.receipt",
                {
                    "operation_id": op["operation_id"],
                    "attempt_id": permit.attempt_id,
                    "status": "succeeded",
                    "output_error": output_error,
                },
            )
            return receipt

    async def mark_write_unknown(self, permit: DispatchPermit) -> None:
        async with self._transaction() as conn:
            _, _, op = await self._op_lock(conn, permit.operation.operation_id)
            if op["state"] == "succeeded":
                return
            attempt = await self._one(
                conn,
                "SELECT * FROM harness.tool_attempts WHERE attempt_id=%s FOR UPDATE",
                (permit.attempt_id,),
            )
            if attempt["ordinal"] != op["attempts"]:
                raise RuntimeFault("STALE_TOOL_ATTEMPT")
            await conn.execute(
                "UPDATE harness.operations SET state='unknown' WHERE operation_id=%s",
                (op["operation_id"],),
            )
            await conn.execute(
                (
                    "UPDATE harness.tool_attempts SET "
                    "state='unknown',requires_reconciliation=true WHERE attempt_id=%s "
                    "AND state='dispatched'"
                ),
                (permit.attempt_id,),
            )
            await self._event(
                conn,
                op["run_id"],
                "tool.unknown",
                {"operation_id": op["operation_id"], "attempt_id": permit.attempt_id},
            )

    async def decide_approval(
        self, actor: Principal, approval_id: str, request: ApprovalDecisionRequest
    ) -> dict[str, Any]:
        self._role(actor, "approver")
        async with self._transaction() as conn:
            await self._lock_tenant(conn, actor)
            ref = await self._one(
                conn,
                "SELECT operation_id FROM harness.approvals WHERE approval_id=%s",
                (approval_id,),
            )
            _, run, op = await self._op_lock(conn, ref["operation_id"])
            self._resource_authority(actor, op, "approver")
            if actor.subject_id == Principal.model_validate(op["principal"]).subject_id:
                raise RuntimeFault("SELF_APPROVAL_FORBIDDEN", 403)
            approval = await self._one(
                conn,
                "SELECT * FROM harness.approvals WHERE approval_id=%s FOR UPDATE",
                (approval_id,),
            )
            digest = canonical_hash(
                {"binding_hash": request.binding_hash, "decision": request.decision}
            )
            if approval["decision_request_id"] == request.request_id:
                if (
                    approval["decision_hash"] != digest
                    or approval["decided_by"] != actor.subject_id
                ):
                    raise RuntimeFault("APPROVAL_REQUEST_CONFLICT")
                return {
                    "approval_id": approval_id,
                    "state": approval["state"],
                    "revision": approval["revision"],
                }
            if approval["state"] != "pending" or approval["revision"] != request.expected_revision:
                raise RuntimeFault("APPROVAL_REVISION_CONFLICT")
            if (
                request.binding_hash != op["binding_hash"]
                or request.binding_hash != approval["binding_hash"]
            ):
                raise RuntimeFault("APPROVAL_BINDING_MISMATCH")
            if (
                run["cancel_requested"]
                or run["now"] >= min(approval["expires_at"], run["expires_at"])
                or run["status"] not in {"running", "waiting_approval"}
            ):
                raise RuntimeFault("APPROVAL_NO_LONGER_VALID")
            state = "approved" if request.decision == "approve" else "rejected"
            await self._rate_admit(conn, actor)
            await conn.execute(
                (
                    "UPDATE harness.approvals SET "
                    "state=%s,revision=revision+1,decided_by=%s,decision_request_id=%s"
                    ",decision_hash=%s WHERE approval_id=%s"
                ),
                (state, actor.subject_id, request.request_id, digest, approval_id),
            )
            if state == "rejected":
                await conn.execute(
                    (
                        "UPDATE harness.operations SET state='rejected' WHERE "
                        "operation_id=%s AND state='prepared'"
                    ),
                    (op["operation_id"],),
                )
            # 早到批准仅留账；真正排空并绑定checkpoint之后才能排回队列。
            # 排回队列只由Worker领取前的调度事务处理，API不拥有恢复权。
            await self._audit(conn, actor, "approval." + state, approval_id)
            await self._event(
                conn,
                run["run_id"],
                "approval.decided",
                {"approval_id": approval_id, "decision": state},
            )
            return {
                "approval_id": approval_id,
                "state": state,
                "revision": approval["revision"] + 1,
            }

    async def _queue_suspended(self, conn: Connection, run: Row, op: Row, decision: str) -> None:
        if run["suspension"]["operation_id"] != op["operation_id"]:
            raise RuntimeFault("SUSPENSION_OPERATION_MISMATCH")
        if decision == "rejected":
            await self._terminate(conn, run, "failed", "APPROVAL_REJECTED")
        else:
            await conn.execute(
                (
                    "UPDATE harness.runs SET "
                    "status='queued',revision=revision+1,queue_deadline=least(expires_"
                    "at,clock_timestamp()+%s * interval '1 second') WHERE run_id=%s"
                ),
                (self.profile.queue_timeout_seconds, run["run_id"]),
            )
            await self._event(
                conn, run["run_id"], "run.resume_queued", {"operation_id": op["operation_id"]}
            )

    async def list_approvals(
        self, actor: Principal, after: str, limit: int
    ) -> list[dict[str, Any]]:
        self._role(actor, "approver")
        if not 1 <= limit <= 100:
            raise RuntimeFault("INVALID_PAGE_SIZE", 422)
        async with self._transaction() as conn:
            rows = await (
                await conn.execute(
                    """SELECT a.*,o.tool,o.arguments,o.principal FROM harness.approvals a
                   JOIN harness.operations o USING(operation_id)
                   JOIN harness.runs r ON r.run_id=o.run_id
                   WHERE o.principal->>'tenant_id'=%s AND o.principal->>'application_id'=%s
                   AND o.principal->'resource_scopes' <@ %s AND a.approval_id>%s
                   AND a.state='pending' AND a.expires_at>clock_timestamp()
                   AND r.status IN ('running','waiting_approval') AND NOT r.cancel_requested
                   AND r.expires_at>clock_timestamp()
                   ORDER BY a.approval_id LIMIT %s""",
                    (
                        actor.tenant_id,
                        actor.application_id,
                        Jsonb(sorted(actor.resource_scopes)),
                        after,
                        limit,
                    ),
                )
            ).fetchall()
            result = []
            for row in rows:
                tool = ToolSpec.model_validate(row["tool"])
                args = dict(row["arguments"])
                # 配置含复杂敏感路径时保守隐藏全部参数，不静默漏掉嵌套字段。
                if any("." in field or "[" in field for field in tool.sensitive_fields):
                    args = {"redacted": True}
                else:
                    for field in tool.sensitive_fields:
                        if field in args:
                            args[field] = "[REDACTED]"
                result.append(
                    {
                        "approval_id": row["approval_id"],
                        "operation_id": row["operation_id"],
                        "binding_hash": row["binding_hash"],
                        "revision": row["revision"],
                        "state": row["state"],
                        "expires_at": row["expires_at"].isoformat(),
                        "tool": tool.name,
                        "arguments": args,
                    }
                )
            return result

    async def park_write(self, owner: str, proof: SuspensionProof) -> None:
        async with self._transaction() as conn:
            _, run, op = await self._op_lock(conn, proof.operation_id)
            if (
                run["run_id"] != proof.run_id
                or run["owner_id"] != owner
                or run["status"] != "running"
                or run["manifest_hash"] != proof.manifest_hash
            ):
                raise RuntimeFault("SUSPENSION_PROOF_MISMATCH")
            status = (
                "reconciling" if op["state"] in {"dispatched", "unknown"} else "waiting_approval"
            )
            await conn.execute(
                (
                    "UPDATE harness.runs SET "
                    "status=%s,suspension=%s,execution_used_ms=execution_used_ms+great"
                    "est(0,extract(epoch FROM "
                    "clock_timestamp()-execution_started_at)*1000)::bigint,execution_s"
                    "tarted_at=NULL,revision=revision+1 WHERE run_id=%s"
                ),
                (status, Jsonb(proof.model_dump(mode="json")), run["run_id"]),
            )
            await self._event(
                conn, run["run_id"], "run." + status, {"operation_id": op["operation_id"]}
            )

    async def begin_reconciliation(
        self, actor: Principal, run_id: str, request: ReconcileRequest
    ) -> ReconciliationPermit:
        self._role(actor, "reconciler")
        async with self._transaction() as conn:
            await self._lock_tenant(conn, actor)
            _, run, op = await self._op_lock(conn, request.operation_id)
            self._resource_authority(actor, op, "reconciler")
            if run["run_id"] != run_id:
                raise RuntimeFault("NOT_FOUND", 404)
            old = await (
                await conn.execute(
                    (
                        "SELECT * FROM harness.reconciliation_records WHERE run_id=%s AND "
                        "request_id=%s"
                    ),
                    (run_id, request.request_id),
                )
            ).fetchone()
            if old:
                if old["operation_id"] != op["operation_id"] or old["actor"] != actor.subject_id:
                    raise RuntimeFault("RECONCILIATION_REQUEST_CONFLICT")
                if old["result"] is None:
                    raise RuntimeFault("RECONCILIATION_BUSY_OR_EXHAUSTED", 429)
                return ReconciliationPermit(
                    token=old["token"],
                    operation=self._operation(op),
                    deadline_at=run["now"],
                    recorded_result=ReconciliationResult.model_validate(old["result"]),
                )
            if run["status"] != "reconciling" or op["state"] not in {"dispatched", "unknown"}:
                raise RuntimeFault("OPERATION_NOT_RECONCILABLE")
            active = await (
                await conn.execute(
                    "SELECT 1 FROM harness.worker_registration WHERE active_run_id=%s", (run_id,)
                )
            ).fetchone()
            if active:
                raise RuntimeFault("EXECUTION_NOT_DRAINED")
            if (
                op["reconcile_token"] is not None
                or op["reconcile_count"] >= self.writes.max_reconciliations_per_operation
            ):
                raise RuntimeFault("RECONCILIATION_BUSY_OR_EXHAUSTED", 429)
            token = uuid4().hex
            await self._rate_admit(conn, actor)
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.reconciliation_records(token,operation_id,run_id,request_"
                    "id,actor) VALUES(%s,%s,%s,%s,%s)"
                ),
                (token, op["operation_id"], run_id, request.request_id, actor.subject_id),
            )
            await conn.execute(
                (
                    "UPDATE harness.operations SET "
                    "reconcile_token=%s,reconcile_count=reconcile_count+1 WHERE "
                    "operation_id=%s"
                ),
                (token, op["operation_id"]),
            )
            await self._audit(conn, actor, "reconciliation.started", token)
            return ReconciliationPermit(
                token=token,
                operation=self._operation(op),
                deadline_at=run["now"]
                + timedelta(seconds=self.writes.reconciliation_timeout_seconds),
            )

    async def finish_reconciliation(
        self, actor: Principal, permit: ReconciliationPermit, result: ReconciliationResult
    ) -> WriteOperation:
        async with self._transaction() as conn:
            _, run, op = await self._op_lock(conn, permit.operation.operation_id)
            self._resource_authority(actor, op, "reconciler")
            if (
                op["reconcile_token"] != permit.token
                or result.operation_id != op["operation_id"]
                or result.args_hash != op["args_hash"]
            ):
                raise RuntimeFault("RECONCILIATION_BINDING_MISMATCH")
            state = {
                "confirmed_succeeded": "succeeded",
                "confirmed_not_executed": "not_executed",
                "still_unknown": "unknown",
            }[result.conclusion]
            receipt = None
            attempt = await self._one(
                conn,
                (
                    "SELECT * FROM harness.tool_attempts WHERE operation_id=%s ORDER "
                    "BY ordinal DESC LIMIT 1 FOR UPDATE"
                ),
                (op["operation_id"],),
            )
            if result.result is not None:
                receipt = ToolReceipt(
                    operation_id=op["operation_id"],
                    attempt_id=attempt["attempt_id"],
                    status=ReceiptState.SUCCEEDED,
                    args_hash=op["args_hash"],
                    external_reference=result.result.external_reference,
                    data=result.result.data,
                    error_code=result.output_error,
                    cost_status="not_billable",
                )
            await conn.execute(
                (
                    "UPDATE harness.operations SET "
                    "state=%s,receipt=%s,output_error=%s,reconcile_token=NULL WHERE "
                    "operation_id=%s"
                ),
                (
                    state,
                    Jsonb(receipt.model_dump(mode="json")) if receipt else None,
                    result.output_error,
                    op["operation_id"],
                ),
            )
            await conn.execute(
                (
                    "UPDATE harness.tool_attempts SET "
                    "state=%s,finished_at=clock_timestamp() WHERE attempt_id=%s"
                ),
                (state, attempt["attempt_id"]),
            )
            await conn.execute(
                "UPDATE harness.reconciliation_records SET result=%s WHERE token=%s",
                (Jsonb(result.model_dump(mode="json")), permit.token),
            )
            await self._event(
                conn,
                run["run_id"],
                "tool.reconciled",
                {"operation_id": op["operation_id"], "conclusion": result.conclusion},
            )
            if state != "unknown":
                unresolved = await (
                    await conn.execute(
                        (
                            "SELECT 1 FROM harness.operations WHERE run_id=%s AND state IN "
                            "('unknown','dispatched')"
                        ),
                        (run["run_id"],),
                    )
                ).fetchone()
                if not unresolved:
                    if run["cancel_requested"] or run["now"] >= run["expires_at"]:
                        await self._terminate(
                            conn,
                            run,
                            "cancelled" if run["cancel_requested"] else "expired",
                            "CANCELLED_AFTER_RECONCILIATION"
                            if run["cancel_requested"]
                            else "EXPIRED_AFTER_RECONCILIATION",
                        )
                    elif result.output_error:
                        await self._terminate(conn, run, "failed", result.output_error)
            await self._audit(conn, actor, "reconciliation.finished", permit.token)
            row = await self._one(
                conn,
                "SELECT * FROM harness.operations WHERE operation_id=%s",
                (op["operation_id"],),
            )
            return self._operation(row)

    async def _enqueue_ready_write(self) -> None:
        async with self._transaction() as conn:
            ref = await (
                await conn.execute(
                    """SELECT r.run_id,o.operation_id,o.principal FROM harness.runs r
                   JOIN harness.operations o ON o.run_id=r.run_id
                   JOIN harness.approvals a ON a.approval_id=o.approval_id
                   WHERE r.status IN ('waiting_approval','reconciling') AND r.suspension IS NOT NULL
                   AND r.suspension->>'operation_id'=o.operation_id AND NOT r.recovery_required
                   AND o.state NOT IN ('unknown','dispatched')
                   AND (a.state<>'pending' OR a.expires_at<=clock_timestamp()
                        OR r.expires_at<=clock_timestamp() OR r.cancel_requested)
                   """
                    + (
                        " AND r.manifest_hash=ANY(%s)"
                        if self.installed_manifest_hashes is not None
                        else ""
                    )
                    + " ORDER BY r.created_at,r.run_id LIMIT 1",
                    (sorted(self.installed_manifest_hashes),)
                    if self.installed_manifest_hashes is not None
                    else (),
                )
            ).fetchone()
            if ref is None:
                return
            actor = Principal.model_validate(ref["principal"])
            # 先tenant，再session/run/operation；恢复不消耗新的open配额，也不能超卖queued。
            await conn.execute(
                "SELECT tenant_id FROM harness.tenant_admission WHERE tenant_id=%s FOR UPDATE",
                (actor.tenant_id,),
            )
            _, run, op = await self._op_lock(conn, ref["operation_id"])
            if run["status"] not in {"waiting_approval", "reconciling"} or run["recovery_required"]:
                return
            approval = await self._one(
                conn,
                "SELECT * FROM harness.approvals WHERE approval_id=%s FOR UPDATE",
                (op["approval_id"],),
            )
            if run["cancel_requested"] or run["now"] >= run["expires_at"]:
                await self._terminate(
                    conn,
                    run,
                    "cancelled" if run["cancel_requested"] else "expired",
                    "SUSPENDED_RUN_STOPPED",
                )
                return
            if approval["state"] == "rejected" or run["now"] >= approval["expires_at"]:
                await self._terminate(conn, run, "failed", "APPROVAL_REJECTED_OR_EXPIRED")
                return
            if approval["state"] == "pending":
                return
            unresolved = await (
                await conn.execute(
                    (
                        "SELECT 1 FROM harness.operations WHERE run_id=%s AND state IN "
                        "('dispatched','unknown')"
                    ),
                    (run["run_id"],),
                )
            ).fetchone()
            if unresolved:
                return
            counts = await self._one(
                conn,
                (
                    "SELECT count(*) AS count FROM harness.runs r JOIN "
                    "harness.sessions s USING(session_id) WHERE s.tenant_id=%s AND "
                    "r.status='queued'"
                ),
                (actor.tenant_id,),
            )
            if counts["count"] >= self.profile.max_queued_per_tenant:
                return
            await self._queue_suspended(conn, run, op, approval["state"])

    async def claim(self, identity: str) -> ClaimedRun | None:
        # 只有已登记且空槽的执行者可以调度恢复，不能通过API调用本方法。
        async with self._transaction() as conn:
            worker = await self._worker(conn, identity)
            if worker["active_run_id"] or worker["state"] != "idle":
                raise RuntimeFault("WORKER_SLOT_BUSY")
        await self._enqueue_ready_write()
        return await super().claim(identity)

    async def write_receipts(self, actor: Principal, run_id: str) -> tuple[ToolReceipt, ...]:
        async with self._transaction() as conn:
            await self._authorized_run(conn, actor, run_id)
            return await self._evidence.write_receipts(conn, run_id)
