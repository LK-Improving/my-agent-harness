"""本机受控恢复事务；外部终止/checkpoint证明必须先完成，事务中不调用模型。"""

from psycopg.types.json import Jsonb

from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.contracts.execution import ProcessIdentity, TerminationProof
from harness.contracts.hashing import RunRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


class PostgresRecoveryStore(PostgresExecutionStore):
    async def release_for_run(self, actor: Principal, run_id: str) -> ReleaseManifest:
        """只读运维检查；不要求运行已终止，不改变租约、状态或计数。"""
        self._operator(actor)
        async with self._transaction() as conn:
            row = await self._one(conn, "SELECT * FROM harness.runs WHERE run_id=%s", (run_id,))
            principal = Principal.model_validate(row["principal"])
            if (principal.tenant_id, principal.application_id) != (
                actor.tenant_id,
                actor.application_id,
            ):
                raise RuntimeFault("NOT_FOUND", 404)
            session = await self._one(
                conn,
                "SELECT release_id FROM harness.sessions WHERE session_id=%s",
                (row["session_id"],),
            )
            release = await self._release(conn, principal, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            if manifest.content_hash() != row["manifest_hash"]:
                raise RuntimeFault("STORED_MANIFEST_MISMATCH")
            return manifest

    def _operator(self, actor: Principal) -> None:
        self._role(actor, "recovery_operator")
        if "harness:recovery:local" not in actor.resource_scopes:
            raise RuntimeFault("RECOVERY_SCOPE_REQUIRED", 403)

    async def close_reconciliation(self, actor: Principal, operation_id: str) -> None:
        self._operator(actor)
        async with self._transaction() as conn:
            _, _, op = await self._op_lock(conn, operation_id)
            original = Principal.model_validate(op["principal"])
            if (original.tenant_id, original.application_id) != (
                actor.tenant_id,
                actor.application_id,
            ):
                raise RuntimeFault("NOT_FOUND", 404)
            token = op["reconcile_token"]
            if token is None:
                return
            # 仅关闭只读核对尝试，不改变未知业务写结论，不释放guard或退款计数。
            result = {
                "operation_id": operation_id,
                "args_hash": op["args_hash"],
                "conclusion": "still_unknown",
                "evidence_ref": "local-operator-closed-read-attempt",
                "authoritative_not_executed": False,
                "result": None,
            }
            await conn.execute(
                (
                    "UPDATE harness.reconciliation_records SET result=%s WHERE "
                    "token=%s AND result IS NULL"
                ),
                (Jsonb(result), token),
            )
            await conn.execute(
                (
                    "UPDATE harness.operations SET reconcile_token=NULL WHERE "
                    "operation_id=%s AND reconcile_token=%s"
                ),
                (operation_id, token),
            )
            await self._audit(conn, actor, "reconciliation.read_attempt_closed", token)

    async def registered_process(self, actor: Principal, identity: str) -> ProcessIdentity:
        self._operator(actor)
        async with self._transaction() as conn:
            row = await self._one(
                conn,
                "SELECT process FROM harness.worker_registration WHERE singleton AND identity=%s",
                (identity,),
            )
            if not row["process"]:
                raise RuntimeFault("WORKER_PROCESS_METADATA_MISSING")
            return ProcessIdentity.model_validate(row["process"])

    async def retire_worker(self, actor: Principal, identity: str, proof: TerminationProof) -> None:
        self._operator(actor)
        async with self._transaction() as conn:
            registration = await self._worker(conn, identity)
            if registration["process"] != proof.process.model_dump(mode="json"):
                raise RuntimeFault("TERMINATION_PROOF_BINDING_MISMATCH")
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.terminated_workers(identity,process,proof,actor) "
                    "VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING"
                ),
                (
                    identity,
                    Jsonb(registration["process"]),
                    Jsonb(proof.model_dump(mode="json")),
                    Jsonb(actor.model_dump(mode="json")),
                ),
            )
            refs = await (
                await conn.execute(
                    (
                        "SELECT run_id FROM harness.runs WHERE owner_id=%s AND status IN "
                        "('running','queued','waiting_approval','reconciling') ORDER BY "
                        "session_id,run_id"
                    ),
                    (identity,),
                )
            ).fetchall()
            for ref in refs:
                _, run = await self._lock_owned_run(conn, identity, ref["run_id"])
                await conn.execute(
                    (
                        "UPDATE harness.tool_attempts SET "
                        "state='unknown',requires_reconciliation=true WHERE operation_id "
                        "IN (SELECT operation_id FROM harness.operations WHERE run_id=%s) "
                        "AND state='dispatched'"
                    ),
                    (run["run_id"],),
                )
                await conn.execute(
                    (
                        "UPDATE harness.operations SET state='unknown' WHERE run_id=%s "
                        "AND state='dispatched'"
                    ),
                    (run["run_id"],),
                )
                await conn.execute(
                    (
                        "UPDATE harness.model_calls SET state='unknown' WHERE run_id=%s "
                        "AND state='dispatched'"
                    ),
                    (run["run_id"],),
                )
                await conn.execute(
                    (
                        "UPDATE harness.read_operations SET state='unknown' WHERE "
                        "run_id=%s AND state='dispatched'"
                    ),
                    (run["run_id"],),
                )
                unresolved = await (
                    await conn.execute(
                        "SELECT 1 FROM harness.operations WHERE run_id=%s AND state='unknown'",
                        (run["run_id"],),
                    )
                ).fetchone()
                await conn.execute(
                    (
                        "UPDATE harness.runs SET recovery_required=true,status=CASE WHEN "
                        "%s THEN 'reconciling' ELSE status "
                        "END,execution_used_ms=execution_used_ms+CASE WHEN "
                        "execution_started_at IS NULL THEN 0 ELSE "
                        "greatest(0,extract(epoch FROM "
                        "clock_timestamp()-execution_started_at)*1000)::bigint "
                        "END,execution_started_at=NULL,error_code='WORKER_TERMINATION_CONF"
                        "IRMED',revision=revision+1 WHERE run_id=%s"
                    ),
                    (bool(unresolved), run["run_id"]),
                )
                await self._event(
                    conn, run["run_id"], "run.recovery_required", {"worker_identity": identity}
                )
            await self._audit(conn, actor, "worker.termination_verified", identity)
            await conn.execute(
                "DELETE FROM harness.worker_registration WHERE singleton AND identity=%s",
                (identity,),
            )

    async def recovery_run(self, actor: Principal, run_id: str) -> tuple[ClaimedRun, int, str]:
        self._operator(actor)
        async with self._transaction() as conn:
            row = await self._one(conn, "SELECT * FROM harness.runs WHERE run_id=%s", (run_id,))
            session = await self._one(
                conn, "SELECT * FROM harness.sessions WHERE session_id=%s", (row["session_id"],)
            )
            principal = Principal.model_validate(row["principal"])
            if (
                principal.tenant_id != actor.tenant_id
                or principal.application_id != actor.application_id
            ):
                raise RuntimeFault("NOT_FOUND", 404)
            await self._one(
                conn,
                "SELECT identity FROM harness.terminated_workers WHERE identity=%s",
                (row["owner_id"],),
            )
            release = await self._release(conn, principal, session["release_id"])
            turns = await (
                await conn.execute(
                    "SELECT messages FROM harness.turns WHERE session_id=%s ORDER BY revision",
                    (row["session_id"],),
                )
            ).fetchall()
            return (
                ClaimedRun(
                    run_id=run_id,
                    session_id=row["session_id"],
                    principal=principal,
                    request=RunRequest.model_validate(row["request"]),
                    manifest=ReleaseManifest.model_validate(release["manifest"]),
                    execution_deadline=row["execution_deadline"] or row["expires_at"],
                    history=JsonSnapshot.capture(
                        [message for turn in turns for message in turn["messages"]]
                    ),
                ),
                row["revision"],
                row["owner_id"],
            )

    async def recover_run(
        self,
        actor: Principal,
        run: ClaimedRun,
        expected_revision: int,
        checkpoint: JsonSnapshot | None,
        *,
        abort: bool = False,
    ) -> str:
        self._operator(actor)
        if (actor.tenant_id, actor.application_id) != (
            run.principal.tenant_id,
            run.principal.application_id,
        ):
            raise RuntimeFault("NOT_FOUND", 404)
        async with self._transaction() as conn:
            await conn.execute(
                "SELECT tenant_id FROM harness.tenant_admission WHERE tenant_id=%s FOR UPDATE",
                (run.principal.tenant_id,),
            )
            registration = await (
                await conn.execute(
                    "SELECT * FROM harness.worker_registration WHERE singleton FOR UPDATE"
                )
            ).fetchone()
            if registration and registration["active_run_id"]:
                raise RuntimeFault("WORKER_MUST_BE_IDLE_FOR_RECOVERY")
            row = await self._one(
                conn, "SELECT owner_id FROM harness.runs WHERE run_id=%s", (run.run_id,)
            )
            session, row = await self._lock_owned_run(conn, row["owner_id"], run.run_id)
            if (
                Principal.model_validate(row["principal"]) != run.principal
                or session["session_id"] != run.session_id
                or RunRequest.model_validate(row["request"]).business_hash()
                != run.request.business_hash()
            ):
                raise RuntimeFault("RECOVERY_BINDING_MISMATCH")
            await self._one(
                conn,
                "SELECT identity FROM harness.terminated_workers WHERE identity=%s",
                (row["owner_id"],),
            )
            if row["revision"] != expected_revision:
                raise RuntimeFault("RECOVERY_REVISION_CONFLICT")
            if row["status"] in {"succeeded", "partial", "failed", "cancelled", "expired"}:
                return str(row["status"])
            if not row["recovery_required"] or run.manifest.content_hash() != row["manifest_hash"]:
                raise RuntimeFault("RECOVERY_BINDING_MISMATCH")
            if abort or row["cancel_requested"] or row["now"] >= row["expires_at"]:
                await self._terminate(
                    conn,
                    row,
                    "cancelled" if row["cancel_requested"] else "failed",
                    "CONTROLLED_RECOVERY_ABORTED",
                )
                await self._audit(conn, actor, "run.recovery_aborted", run.run_id)
                updated = await self._one(
                    conn, "SELECT status FROM harness.runs WHERE run_id=%s", (run.run_id,)
                )
                return str(updated["status"])
            if (
                self.installed_manifest_hashes is None
                or row["manifest_hash"] not in self.installed_manifest_hashes
            ):
                raise RuntimeFault("EXECUTABLE_RELEASE_NOT_INSTALLED", 503)
            await self._session(conn, run.principal, session["session_id"])
            unknown = await (
                await conn.execute(
                    (
                        "SELECT 1 FROM harness.operations WHERE run_id=%s AND state IN "
                        "('dispatched','unknown')"
                    ),
                    (run.run_id,),
                )
            ).fetchone()
            if unknown:
                raise RuntimeFault("RECONCILE_BEFORE_RECOVERY")
            ambiguous = await (
                await conn.execute(
                    (
                        "SELECT 1 FROM harness.model_calls WHERE run_id=%s AND "
                        "state<>'succeeded' UNION ALL SELECT 1 FROM "
                        "harness.read_operations WHERE run_id=%s AND state<>'succeeded' "
                        "LIMIT 1"
                    ),
                    (run.run_id, run.run_id),
                )
            ).fetchone()
            if ambiguous:
                raise RuntimeFault("UNRESOLVED_ATTEMPT_ABORT_REQUIRED")
            if row["execution_used_ms"] >= self.profile.execution_timeout_seconds * 1000:
                raise RuntimeFault("EXECUTION_BUDGET_EXHAUSTED", 429)
            if checkpoint is None:
                attempts = await self._one(
                    conn,
                    (
                        "SELECT (SELECT count(*) FROM harness.model_calls WHERE "
                        "run_id=%s)+(SELECT count(*) FROM harness.operations WHERE "
                        "run_id=%s)+(SELECT count(*) FROM harness.read_operations WHERE "
                        "run_id=%s) AS count"
                    ),
                    (run.run_id, run.run_id, run.run_id),
                )
                if attempts["count"]:
                    raise RuntimeFault("CHECKPOINT_MISSING_WITH_EXECUTION_EVIDENCE")
            elif checkpoint.as_object().get("manifest_hash") != row["manifest_hash"]:
                raise RuntimeFault("RECOVERY_CHECKPOINT_BINDING_MISMATCH")
            counts = await self._one(
                conn,
                (
                    "SELECT count(*) AS count FROM harness.runs r JOIN "
                    "harness.sessions s USING(session_id) WHERE s.tenant_id=%s AND "
                    "r.status='queued'"
                ),
                (run.principal.tenant_id,),
            )
            if counts["count"] >= self.profile.max_queued_per_tenant:
                raise RuntimeFault("TENANT_QUEUE_FULL", 429)
            await conn.execute(
                (
                    "UPDATE harness.runs SET "
                    "recovery_required=false,suspension=%s,status='queued',revision=re"
                    "vision+1,queue_deadline=least(expires_at,clock_timestamp()+%s * "
                    "interval '1 second') WHERE run_id=%s"
                ),
                (
                    Jsonb(checkpoint.as_object()) if checkpoint else None,
                    self.profile.queue_timeout_seconds,
                    run.run_id,
                ),
            )
            await self._audit(conn, actor, "run.recovery_queued", run.run_id)
            await self._event(conn, run.run_id, "run.recovery_queued", {})
            return "queued"
