"""模型/只读尝试账本；未知尝试不自动再派发。"""

import asyncio
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, Row
from harness.adapters.postgres.events import PostgresEventJournal
from harness.adapters.postgres.lifecycle import PostgresRunLifecycle
from harness.adapters.writes_postgres import PostgresWriteStore
from harness.context.requests import envelope_tokens
from harness.context.boundary import require_context_capacity, require_memory_boundary
from harness.contracts.context import ModelCallPurpose
from harness.contracts.execution import (
    ExecutionProfile,
    ModelReservation,
    ProcessIdentity,
    ReadReservation,
)
from harness.contracts.extensions import ExtensionDecision, ExtensionInvocation, ExtensionSpec
from harness.contracts.hashing import canonical_hash
from harness.contracts.models import (
    ModelBudgetQuote,
    ModelParameters,
    ModelResult,
    PreparedModelRequest,
    model_input_hash,
)
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import (
    AdmissionProfile,
    AnswerCandidate,
    ClaimedRun,
    Principal,
    RuntimeFault,
)
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ExecutionContext, ToolSpec
from harness.contracts.state import ReceiptState, ToolReceipt
from harness.contracts.writes import DispatchPermit, WritePolicy


class PostgresExecutionStore(PostgresWriteStore):
    delegated_read_hashes: frozenset[str] = frozenset()
    process_identity: ProcessIdentity | None = None
    identity_source: Any = None
    new_run_ready: Callable[[], bool] | None = None

    def __init__(
        self, dsn: str, profile: AdmissionProfile, writes: WritePolicy | None = None
    ) -> None:
        super().__init__(dsn, profile, writes)
        self._events = PostgresEventJournal(writes=True, drafts=True)
        self._lifecycle = PostgresRunLifecycle(self._events)

    async def record_extension(
        self,
        owner: str,
        run: ClaimedRun,
        spec: ExtensionSpec,
        invocation: ExtensionInvocation,
        decision: ExtensionDecision,
    ) -> None:
        expected = ExecutionContext(
            **run.principal.model_dump(mode="python"),
            run_id=run.run_id,
            release_id=run.manifest.spec.release_id,
            trace_id=run.run_id,
        )
        if (
            invocation.context != expected
            or invocation.configuration != spec.configuration
            or invocation.manifest_hash != run.manifest.content_hash()
            or invocation.stage != spec.stage
            or decision.invocation_fingerprint != invocation.fingerprint()
            or decision.extension_id != spec.extension_id
            or decision.version != spec.version
        ):
            raise RuntimeFault("EXTENSION_AUDIT_BINDING_MISMATCH")
        installed = run.manifest.profiles.as_object().get("extensions")
        registrations = installed.get("registrations") if isinstance(installed, dict) else None
        if not isinstance(registrations, list) or spec.model_dump(mode="json") not in registrations:
            raise RuntimeFault("EXTENSION_MANIFEST_MISMATCH")
        digest = canonical_hash(decision.model_dump(mode="json"))
        async with self._transaction() as conn:
            session, row = await self._lock_owned_run(conn, owner, run.run_id)
            await self._check_execution(conn, run.principal, owner, row, session)
            if row["manifest_hash"] != run.manifest.content_hash():
                raise RuntimeFault("EXTENSION_AUDIT_BINDING_MISMATCH")
            await conn.execute(
                """INSERT INTO harness.extension_decisions
                (run_id,extension_id,extension_version,invocation_fingerprint,stage,spec,invocation,decision,decision_hash)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (
                    run.run_id,
                    spec.extension_id,
                    spec.version,
                    invocation.fingerprint(),
                    spec.stage,
                    Jsonb(spec.model_dump(mode="json")),
                    Jsonb(invocation.model_dump(mode="json")),
                    Jsonb(decision.model_dump(mode="json")),
                    digest,
                ),
            )
            recorded = await self._one(
                conn,
                """SELECT spec,decision_hash FROM harness.extension_decisions WHERE
                run_id=%s AND extension_id=%s AND extension_version=%s AND
                invocation_fingerprint=%s""",
                (run.run_id, spec.extension_id, spec.version, invocation.fingerprint()),
            )
            if recorded["decision_hash"] != digest or recorded["spec"] != spec.model_dump(
                mode="json"
            ):
                raise RuntimeFault("EXTENSION_REPLAY_DECISION_CHANGED")
            await self._audit(conn, run.principal, "extension.decision", invocation.fingerprint())

    async def _check_execution(
        self, conn: Connection, actor: Principal, owner: str, run: Row, session: Row
    ) -> None:
        if self.identity_source is not None:
            current = await asyncio.to_thread(self.identity_source.current, actor)
            if (
                not actor.roles <= current.roles
                or not actor.resource_scopes <= current.resource_scopes
            ):
                raise RuntimeFault("EXECUTION_AUTHORITY_REVOKED", 403)
        await super()._check_execution(conn, actor, owner, run, session)

    async def authorize_extension_in_transaction(
        self, conn: Connection, actor: Principal, owner: str, run_id: str
    ) -> tuple[Row, ReleaseManifest]:
        """可选扩展复用session→run锁与执行授权；不新增领域分支或外部调用。"""
        session, row = await self._lock_owned_run(conn, owner, run_id)
        await self._check_execution(conn, actor, owner, row, session)
        if Principal.model_validate(row["principal"]) != actor:
            raise RuntimeFault("EXTENSION_PRINCIPAL_MISMATCH", 403)
        release = await self._release(conn, actor, session["release_id"])
        manifest = ReleaseManifest.model_validate(release["manifest"])
        if row["manifest_hash"] != manifest.content_hash():
            raise RuntimeFault("EXTENSION_MANIFEST_MISMATCH", 409)
        return row, manifest

    async def begin_dispatch(
        self, actor: Principal, owner: str, operation_id: str
    ) -> DispatchPermit:
        if self.identity_source is not None:
            async with self._transaction() as conn:
                row = await self._one(
                    conn,
                    (
                        "SELECT o.principal,a.decided_by FROM harness.operations o JOIN "
                        "harness.approvals a USING(approval_id) WHERE o.operation_id=%s"
                    ),
                    (operation_id,),
                )
            if row["decided_by"] is not None:
                original = Principal.model_validate(row["principal"])
                approver = await asyncio.to_thread(
                    self.identity_source.current,
                    original.model_copy(update={"subject_id": row["decided_by"]}),
                )
                self._resource_authority(approver, row, "approver")
        return await super().begin_dispatch(actor, owner, operation_id)

    async def admit_read(self, actor: Principal) -> None:
        async with self._transaction() as conn:
            row = await self._one(
                conn,
                """INSERT INTO harness.read_windows VALUES(%s,%s,%s,
                date_trunc('minute',clock_timestamp()),1)
                ON CONFLICT(tenant_id,subject_id,application_id) DO UPDATE SET
                used=CASE WHEN read_windows.window_start=date_trunc('minute',clock_timestamp())
                THEN read_windows.used+1 ELSE 1 END,
                window_start=date_trunc('minute',clock_timestamp()) RETURNING used""",
                (actor.tenant_id, actor.subject_id, actor.application_id),
            )
            if row["used"] > self.profile.reads_per_subject_minute:
                raise RuntimeFault("READ_RATE_LIMITED", 429)

    async def acquire_sse(self, actor: Principal) -> str:
        async with self._transaction() as conn:
            await conn.execute(
                "SELECT tenant_id FROM harness.tenant_admission WHERE tenant_id=%s FOR UPDATE",
                (actor.tenant_id,),
            )
            await conn.execute(
                "DELETE FROM harness.sse_connections WHERE expires_at<=clock_timestamp()"
            )
            row = await self._one(
                conn,
                (
                    "SELECT count(*) AS count FROM harness.sse_connections WHERE "
                    "tenant_id=%s AND subject_id=%s AND application_id=%s"
                ),
                (actor.tenant_id, actor.subject_id, actor.application_id),
            )
            if row["count"] >= self.profile.max_sse_per_subject:
                raise RuntimeFault("SSE_CONNECTION_LIMIT", 429)
            token = uuid4().hex
            await conn.execute(
                (
                    "INSERT INTO harness.sse_connections "
                    "VALUES(%s,%s,%s,%s,clock_timestamp()+%s * interval '1 second')"
                ),
                (
                    token,
                    actor.tenant_id,
                    actor.subject_id,
                    actor.application_id,
                    self.profile.sse_lease_seconds,
                ),
            )
            return token

    async def renew_sse(self, actor: Principal, token: str) -> None:
        async with self._transaction() as conn:
            await self._one(
                conn,
                (
                    "UPDATE harness.sse_connections SET "
                    "expires_at=clock_timestamp()+%s * interval '1 second' WHERE "
                    "token=%s AND tenant_id=%s AND subject_id=%s AND "
                    "application_id=%s AND expires_at>clock_timestamp() RETURNING "
                    "token"
                ),
                (
                    self.profile.sse_lease_seconds,
                    token,
                    actor.tenant_id,
                    actor.subject_id,
                    actor.application_id,
                ),
            )

    async def release_sse(self, token: str) -> None:
        async with self._transaction() as conn:
            await conn.execute("DELETE FROM harness.sse_connections WHERE token=%s", (token,))

    async def register_worker(self, identity: str) -> None:
        async with self._transaction() as conn:
            row = await (
                await conn.execute(
                    (
                        "INSERT INTO "
                        "harness.worker_registration(singleton,identity,state,process) "
                        "VALUES(true,%s,'idle',%s) ON CONFLICT DO NOTHING RETURNING "
                        "identity"
                    ),
                    (
                        identity,
                        Jsonb(self.process_identity.model_dump(mode="json"))
                        if self.process_identity
                        else None,
                    ),
                )
            ).fetchone()
            if row is None:
                raise RuntimeFault("WORKER_ALREADY_REGISTERED")

    async def reserve_model(
        self,
        run: ClaimedRun,
        owner: str,
        messages: JsonSnapshot,
        tools: JsonSnapshot,
        profile: ExecutionProfile,
    ) -> ModelReservation:
        return await self._reserve_model(run, owner, messages, tools, profile)

    async def reserve_purpose_model(self, run: ClaimedRun, owner: str, messages: JsonSnapshot,
                                     tools: JsonSnapshot, profile: ExecutionProfile,
                                     purpose: ModelCallPurpose) -> ModelReservation:
        if tools.as_object_array():
            raise RuntimeFault("CONTEXT_SUMMARY_TOOLS_FORBIDDEN", 403)
        return await self._reserve_model(run, owner, messages, tools, profile, purpose=purpose)

    async def authorize_model_purpose(self, run: ClaimedRun, owner: str, messages: JsonSnapshot,
                                      purpose: ModelCallPurpose) -> None:
        from harness.adapters.context_postgres import PostgresContextSource

        async with self._transaction() as conn:
            await PostgresContextSource(self).verify_summary(conn, run, owner, messages, purpose)

    async def _reserve_model(self, run: ClaimedRun, owner: str, messages: JsonSnapshot,
                             tools: JsonSnapshot, profile: ExecutionProfile,
                             *, purpose: ModelCallPurpose | None = None) -> ModelReservation:
        require_memory_boundary(run.manifest)
        parameters = ModelParameters()
        input_hash = model_input_hash(
            profile.route, messages, tools, parameters, profile.budget.max_output_tokens
        )
        accounting_hash = canonical_hash({"purpose": purpose.model_dump(mode="json"), "input_hash": input_hash}) if purpose else input_hash
        async with self._transaction() as conn:
            session, row = await self._lock_owned_run(conn, owner, run.run_id)
            await self._check_execution(conn, run.principal, owner, row, session)
            if run.manifest.model_profile.as_object() != profile.route.model_dump(
                mode="json"
            ) or run.manifest.profiles.as_object().get("execution") != profile.model_dump(
                mode="json"
            ):
                raise RuntimeFault("MODEL_PROFILE_MANIFEST_MISMATCH")
            if purpose is not None:
                from harness.adapters.context_postgres import PostgresContextSource

                await PostgresContextSource(self).verify_summary(conn, run, owner, messages, purpose)
                bound = await (await conn.execute("SELECT input_hash FROM harness.model_calls WHERE run_id=%s AND purpose_ref=%s",
                                                  (run.run_id, purpose.fingerprint()))).fetchone()
                if bound and bound["input_hash"] != accounting_hash:
                    raise RuntimeFault("MODEL_PURPOSE_INPUT_CHANGED", 409)
            old = await (
                await conn.execute(
                    "SELECT * FROM harness.model_calls WHERE run_id=%s AND input_hash=%s",
                    (run.run_id, accounting_hash),
                )
            ).fetchone()
            if old:
                if old["state"] != "succeeded":
                    raise RuntimeFault("MODEL_ATTEMPT_UNRESOLVED")
                result = ModelResult.model_validate(old["result"])
                request = PreparedModelRequest.model_validate(old["request"])
                if (result.usage.input_tokens or 0) > request.budget.input_tokens or (
                    result.usage.output_tokens or 0
                ) > request.max_output_tokens:
                    raise RuntimeFault("MODEL_USAGE_BOUND_EXCEEDED", 429)
                return ModelReservation(request=request, result=result)
            if purpose is not None:
                used = await self._one(conn, "SELECT count(*) AS n FROM harness.model_calls WHERE run_id=%s AND purpose='context_summary'", (run.run_id,))
                if used["n"] >= 1:
                    raise RuntimeFault("CONTEXT_SUMMARY_OPERATION_LIMIT", 429)
            draft = await (
                await conn.execute(
                    "SELECT 1 FROM harness.answer_drafts WHERE run_id=%s", (run.run_id,)
                )
            ).fetchone()
            if draft:
                raise RuntimeFault("MODEL_REGENERATION_AFTER_PUBLISH_DISABLED")
            count = envelope_tokens(
                profile.route.model,
                messages.as_object_array(),
                tools.as_object_array(),
                profile.budget.max_output_tokens,
                parameters,
            )
            remaining = profile.budget.max_total_tokens - row["model_reserved_tokens"]
            require_context_capacity(count, profile)
            if (
                count + profile.budget.max_output_tokens > remaining
                or row["model_steps"] >= run.manifest.spec.execution_limits.max_model_steps
            ):
                raise RuntimeFault("MODEL_BUDGET_EXHAUSTED", 429)
            context = ExecutionContext(
                **run.principal.model_dump(mode="python"),
                run_id=run.run_id,
                release_id=run.manifest.spec.release_id,
                trace_id=run.run_id,
            )
            request = PreparedModelRequest(
                context=context,
                manifest_hash=run.manifest.content_hash(),
                environment_profile_hash=run.manifest.environment_profile_hash,
                model_call_id=uuid4().hex,
                model_attempt_id=uuid4().hex,
                draft_id=uuid4().hex,
                route=profile.route,
                messages=messages,
                tools=tools,
                parameters=parameters,
                max_output_tokens=profile.budget.max_output_tokens,
                budget=ModelBudgetQuote(
                    quote_id=uuid4().hex,
                    policy_ref=profile.budget.version,
                    measured_input_hash=input_hash,
                    input_tokens=count,
                    output_reserved_tokens=profile.budget.max_output_tokens,
                    safety_margin_tokens=profile.safety_margin_tokens,
                    remaining_total_tokens=remaining,
                    expires_at=min(row["execution_deadline"], row["now"] + timedelta(seconds=30)),
                ),
            )
            request.require_dispatchable(row["now"])
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.model_calls(model_call_id,run_id,input_hash,request_finge"
                    "rprint,request,state) VALUES(%s,%s,%s,%s,%s,'dispatched')"
                ),
                (
                    request.model_call_id,
                    run.run_id,
                    accounting_hash,
                    request.fingerprint(),
                    Jsonb(request.model_dump(mode="json")),
                ),
            )
            if purpose is not None:
                await conn.execute("UPDATE harness.model_calls SET purpose='context_summary',purpose_ref=%s,purpose_binding=%s WHERE model_call_id=%s",
                                   (purpose.fingerprint(), Jsonb(purpose.model_dump(mode="json")), request.model_call_id))
                await self._audit(conn, run.principal, "model.context_summary_reserved", purpose.fingerprint())
            await conn.execute(
                (
                    "UPDATE harness.runs SET "
                    "model_steps=model_steps+1,model_reserved_tokens=model_reserved_to"
                    "kens+%s WHERE run_id=%s"
                ),
                (count + request.max_output_tokens, run.run_id),
            )
            await self._audit(
                conn, run.principal, "model.dispatch_intent", request.model_attempt_id
            )
            return ModelReservation(request=request)

    async def save_model_result(self, request: PreparedModelRequest, result: ModelResult) -> None:
        async with self._transaction() as conn:
            ref = await self._one(
                conn, "SELECT owner_id FROM harness.runs WHERE run_id=%s", (request.context.run_id,)
            )
            await self._lock_owned_run(conn, ref["owner_id"], request.context.run_id)
            call = await self._one(
                conn,
                "SELECT * FROM harness.model_calls WHERE model_call_id=%s FOR UPDATE",
                (request.model_call_id,),
            )
            if call["request_fingerprint"] != request.fingerprint():
                raise RuntimeFault("MODEL_REQUEST_BINDING_MISMATCH")
            if call["state"] == "succeeded":
                if call["result"] != result.model_dump(mode="json"):
                    raise RuntimeFault("MODEL_RESULT_CONFLICT")
                return
            await conn.execute(
                "UPDATE harness.model_calls SET state='succeeded',result=%s WHERE model_call_id=%s",
                (Jsonb(result.model_dump(mode="json")), request.model_call_id),
            )

    async def fail_model(self, request: PreparedModelRequest, failure: JsonSnapshot) -> None:
        async with self._transaction() as conn:
            await conn.execute(
                (
                    "UPDATE harness.model_calls SET state='unknown',failure=%s WHERE "
                    "model_call_id=%s AND request_fingerprint=%s AND "
                    "state='dispatched'"
                ),
                (Jsonb(failure.as_object()), request.model_call_id, request.fingerprint()),
            )

    async def reserve_read(
        self,
        run: ClaimedRun,
        owner: str,
        logical_id: str,
        spec: JsonSnapshot,
        arguments: JsonSnapshot,
    ) -> ReadReservation:
        tool = ToolSpec.model_validate(spec.as_object())
        async with self._transaction() as conn:
            session, row = await self._lock_owned_run(conn, owner, run.run_id)
            await self._check_execution(conn, run.principal, owner, row, session)
            binding = next(
                (
                    item
                    for item in run.manifest.spec.allowed_tools
                    if item.name == tool.name and item.version == tool.version
                ),
                None,
            )
            if (
                tool.effect not in {"read", "pure"}
                or (tool.cost.billable and (spec.content_hash not in self.delegated_read_hashes
                    or run.manifest.profiles.as_object().get("read_accounting", {}) != {
                        digest: "child-model-ledger@1" for digest in self.delegated_read_hashes
                    }))
                or binding is None
                or run.manifest.tool_schema_hashes.as_object().get(f"{tool.name}@{tool.version}")
                != spec.content_hash
            ):
                raise RuntimeFault("READ_TOOL_MANIFEST_MISMATCH", 403)
            old = await (
                await conn.execute(
                    (
                        "SELECT * FROM harness.read_operations WHERE run_id=%s AND "
                        "logical_call_id=%s FOR UPDATE"
                    ),
                    (run.run_id, logical_id),
                )
            ).fetchone()
            if old:
                if (
                    old["spec_hash"] != spec.content_hash
                    or old["args_hash"] != arguments.content_hash
                ):
                    raise RuntimeFault("LOGICAL_CALL_CONFLICT")
                if old["state"] != "succeeded":
                    raise RuntimeFault("READ_ATTEMPT_UNRESOLVED")
                return ReadReservation(
                    operation_id=old["operation_id"],
                    attempt_id=old["attempt_id"],
                    receipt=ToolReceipt.model_validate(old["receipt"]),
                )
            count = await self._one(
                conn,
                (
                    "SELECT count(*) AS count FROM harness.read_operations WHERE "
                    "run_id=%s AND tool_name=%s"
                ),
                (run.run_id, tool.name),
            )
            if (
                count["count"] >= binding.max_operations_per_run
                or row["tool_dispatches"] >= run.manifest.spec.execution_limits.max_tool_dispatches
            ):
                raise RuntimeFault("READ_TOOL_LIMIT", 429)
            operation_id, attempt_id = uuid4().hex, uuid4().hex
            await conn.execute(
                (
                    "INSERT INTO "
                    "harness.read_operations(operation_id,run_id,logical_call_id,tool_"
                    "name,spec_hash,args_hash,arguments,attempt_id,state) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'dispatched')"
                ),
                (
                    operation_id,
                    run.run_id,
                    logical_id,
                    tool.name,
                    spec.content_hash,
                    arguments.content_hash,
                    Jsonb(arguments.as_object()),
                    attempt_id,
                ),
            )
            await conn.execute(
                "UPDATE harness.runs SET tool_dispatches=tool_dispatches+1 WHERE run_id=%s",
                (run.run_id,),
            )
            await self._audit(conn, run.principal, "read.dispatch_intent", attempt_id)
            return ReadReservation(operation_id=operation_id, attempt_id=attempt_id)

    async def save_read(
        self, reservation: ReadReservation, data: JsonSnapshot | None, error: str | None
    ) -> ToolReceipt:
        async with self._transaction() as conn:
            ref = await self._one(
                conn,
                "SELECT run_id FROM harness.read_operations WHERE operation_id=%s",
                (reservation.operation_id,),
            )
            owner = await self._one(
                conn, "SELECT owner_id FROM harness.runs WHERE run_id=%s", (ref["run_id"],)
            )
            await self._lock_owned_run(conn, owner["owner_id"], ref["run_id"])
            row = await self._one(
                conn,
                "SELECT * FROM harness.read_operations WHERE operation_id=%s FOR UPDATE",
                (reservation.operation_id,),
            )
            if row["attempt_id"] != reservation.attempt_id:
                raise RuntimeFault("READ_ATTEMPT_MISMATCH")
            receipt = ToolReceipt(
                operation_id=row["operation_id"],
                attempt_id=row["attempt_id"],
                status=ReceiptState.FAILED if error else ReceiptState.SUCCEEDED,
                args_hash=row["args_hash"],
                data=data.materialize() if data else None,
                error_code=error,
                cost_status="unknown" if row["spec_hash"] in self.delegated_read_hashes else "not_billable",
            )
            await conn.execute(
                "UPDATE harness.read_operations SET state=%s,receipt=%s WHERE operation_id=%s",
                (
                    "failed" if error else "succeeded",
                    Jsonb(receipt.model_dump(mode="json")),
                    reservation.operation_id,
                ),
            )
            return receipt

    async def operation_state(
        self, actor: Principal, operation_id: str
    ) -> tuple[JsonSnapshot, JsonSnapshot]:
        async with self._transaction() as conn:
            op = await self._one(
                conn, "SELECT * FROM harness.operations WHERE operation_id=%s", (operation_id,)
            )
            await self._authorized_run(conn, actor, op["run_id"])
            return await self._evidence.operation_state(conn, op)

    async def read_receipts(self, actor: Principal, run_id: str) -> tuple[ToolReceipt, ...]:
        async with self._transaction() as conn:
            await self._authorized_run(conn, actor, run_id)
            return await self._evidence.read_receipts(conn, run_id)

    async def candidate_for_run(self, actor: Principal, run_id: str) -> AnswerCandidate | None:
        async with self._transaction() as conn:
            await self._authorized_run(conn, actor, run_id)
            return await self._evidence.candidate(conn, run_id)

    async def stage_answer(self, owner: str, run: ClaimedRun, candidate: AnswerCandidate) -> None:
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
            session, row = await self._lock_owned_run(conn, owner, run.run_id)
            if row["status"] == "succeeded":
                commit = await self._one(
                    conn,
                    "SELECT candidate_hash FROM harness.commits WHERE run_id=%s",
                    (run.run_id,),
                )
                if commit["candidate_hash"] != digest:
                    raise RuntimeFault("COMMIT_CONFLICT")
                return
            await self._check_execution(conn, run.principal, owner, row, session)
            call = await self._one(
                conn,
                "SELECT * FROM harness.model_calls WHERE run_id=%s AND model_call_id=%s",
                (run.run_id, candidate.model_call_id),
            )
            if self.answer_authority is not None:
                await self.answer_authority(conn, owner, run, candidate)
            prepared = PreparedModelRequest.model_validate(call["request"])
            if call.get("purpose", "answer") != "answer":
                raise RuntimeFault("SUMMARY_CANNOT_PUBLISH_ANSWER", 403)
            if (
                call["state"] != "succeeded"
                or candidate.model_attempt_id != prepared.model_attempt_id
                or candidate.draft_id != prepared.draft_id
            ):
                raise RuntimeFault("ANSWER_ATTEMPT_BINDING_MISMATCH")
            blocked = await (
                await conn.execute(
                    (
                        "SELECT 1 FROM harness.operations WHERE run_id=%s AND "
                        "(state<>'succeeded' OR output_error IS NOT NULL) UNION ALL "
                        "SELECT 1 FROM harness.read_operations WHERE run_id=%s AND "
                        "state<>'succeeded' LIMIT 1"
                    ),
                    (run.run_id, run.run_id),
                )
            ).fetchone()
            if blocked:
                raise RuntimeFault("TOOL_EVIDENCE_NOT_COMPLETE")
            old = await (
                await conn.execute(
                    "SELECT * FROM harness.answer_drafts WHERE run_id=%s FOR UPDATE", (run.run_id,)
                )
            ).fetchone()
            if old:
                if (
                    old["candidate_hash"] != digest
                    or old["draft_id"] != candidate.draft_id
                    or old["state"] == "invalid"
                ):
                    raise RuntimeFault("ANSWER_DRAFT_CONFLICT")
                offset = old["published_offset"]
            else:
                await conn.execute(
                    (
                        "INSERT INTO "
                        "harness.answer_drafts(run_id,draft_id,model_attempt_id,candidate_"
                        "hash,candidate,state) VALUES(%s,%s,%s,%s,%s,'published')"
                    ),
                    (
                        run.run_id,
                        candidate.draft_id,
                        candidate.model_attempt_id,
                        digest,
                        Jsonb(candidate.model_dump(mode="json")),
                    ),
                )
                await self._events.append_published(
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
                offset = 0
        while offset < len(candidate.answer):
            async with self._transaction() as conn:
                session, row = await self._lock_owned_run(conn, owner, run.run_id)
                if row["status"] == "succeeded":
                    commit = await self._one(
                        conn,
                        "SELECT candidate_hash FROM harness.commits WHERE run_id=%s",
                        (run.run_id,),
                    )
                    if commit["candidate_hash"] != digest:
                        raise RuntimeFault("COMMIT_CONFLICT")
                    return
                await self._check_execution(conn, run.principal, owner, row, session)
                draft = await self._one(
                    conn,
                    "SELECT * FROM harness.answer_drafts WHERE run_id=%s FOR UPDATE",
                    (run.run_id,),
                )
                if self.answer_authority is not None:
                    await self.answer_authority(conn, owner, run, candidate)
                if (
                    draft["state"] != "published"
                    or draft["model_attempt_id"] != candidate.model_attempt_id
                    or draft["candidate_hash"] != digest
                    or draft["published_offset"] < offset
                ):
                    raise RuntimeFault("LATE_ANSWER_DELTA_REJECTED")
                if draft["published_offset"] > offset:
                    offset = draft["published_offset"]
                    continue
                delta = candidate.answer[offset : offset + 800]
                await self._events.append_published(
                    conn,
                    run.run_id,
                    "answer.delta",
                    {
                        "draft_id": candidate.draft_id,
                        "model_attempt_id": candidate.model_attempt_id,
                        "delta": delta,
                    },
                )
                offset += len(delta)
                await conn.execute(
                    "UPDATE harness.answer_drafts SET published_offset=%s WHERE run_id=%s",
                    (offset, run.run_id),
                )

    async def complete(self, identity: str, run: ClaimedRun, candidate: AnswerCandidate) -> None:
        # 业务commit已存在时直接走唯一提交校验，不重复发布草稿。
        async with self._transaction() as conn:
            committed = await (
                await conn.execute("SELECT 1 FROM harness.commits WHERE run_id=%s", (run.run_id,))
            ).fetchone()
        if not committed:
            await self.stage_answer(identity, run, candidate)
        await super().complete(identity, run, candidate)

    async def health(self) -> dict[str, Any]:
        health = await super().health()
        async with self._transaction() as conn:
            row = await self._one(
                conn,
                (
                    "SELECT to_regclass('harness_checkpoint.checkpoints') IS NOT NULL "
                    "AND to_regclass('harness_checkpoint.checkpoint_migrations') IS "
                    "NOT NULL AS installed"
                ),
            )
            health["checkpoint"] = row["installed"]
            if row["installed"]:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                version = await self._one(
                    conn, "SELECT max(v) AS version FROM harness_checkpoint.checkpoint_migrations"
                )
                health["checkpoint"] = version["version"] == len(AsyncPostgresSaver.MIGRATIONS) - 1
        return health
