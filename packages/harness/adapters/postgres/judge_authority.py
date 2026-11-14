"""Judge 运行对目标评测和参考材料的事务内授权复查。"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from harness.adapters.postgres.connection import Connection, Row
from harness.contracts.evaluation import EvaluationFreeze, GatePolicy, JudgePolicy
from harness.contracts.evaluation_gates import JudgeInput
from harness.contracts.hashing import RunRequest, canonical_hash, canonical_json
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.evaluation.assets import cases, decode

if TYPE_CHECKING:
    from harness.adapters.execution_postgres import PostgresExecutionStore


class PostgresJudgeAuthority:
    def __init__(
        self, store: "PostgresExecutionStore", frozen: EvaluationFreeze, blobs: dict[str, bytes],
        material_authority: Callable[[Connection, Principal, Row, Row], Awaitable[object]],
        material_authority_ref: str,
    ) -> None:
        self.store = store
        self.frozen = frozen
        self.blobs = dict(blobs)
        self.material_authority = material_authority
        self.material_authority_ref = material_authority_ref
        self.judge_policy = JudgePolicy.model_validate(decode(blobs["judge.json"]))
        self.gate_policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
        self.rows = {row.case_id: row for split in ("development", "holdout")
                     for row in cases(blobs[f"{split}.jsonl"])}

    def ready_for(self, manifest: ReleaseManifest) -> bool:
        policy = self.judge_policy
        profile = manifest.profiles.as_object().get("judge")
        return bool(
            policy.required and policy.implementation and policy.prompt_sha256 and policy.model_binding_ref
            and policy.judge_manifest_hash and isinstance(profile, dict)
            and profile.get("implementation") == policy.implementation
            and manifest.content_hash() == policy.judge_manifest_hash
            and manifest.prompt_hash == policy.prompt_sha256
            and manifest.spec.model_profile_ref == policy.model_binding_ref
            and not manifest.spec.allowed_tools and not manifest.spec.knowledge_bindings
            and manifest.spec.execution_limits.max_model_steps == 1
            and manifest.profiles.as_object().get("resource_authority_ref") == "judge-target@1"
            and len(set(policy.dimensions)) == len(policy.dimensions) <= 64
        )

    def input_for(self, run: ClaimedRun) -> JudgeInput:
        try:
            request = JudgeInput.model_validate(decode(run.request.message.encode("utf-8")))
        except (TypeError, ValueError):
            raise RuntimeFault("JUDGE_INPUT_INVALID", 422) from None
        row = self.rows.get(request.case_id)
        policy = self.judge_policy
        if (row is None or not self.ready_for(run.manifest)
                or run.request.attachment_versions or run.request.business_options
                or canonical_json(request.model_dump(mode="json")) != run.request.message
                or request.freeze_hash != self.frozen.content_hash() or request.input_hash != row.input_hash()
                or request.message != row.message or request.attachment_versions != row.attachment_versions
                or request.business_options != row.business_options or request.expected != row.expected
                or request.dimensions != policy.dimensions):
            raise RuntimeFault("JUDGE_POLICY_OR_INPUT_MISMATCH", 403)
        return request

    async def authorize_in_transaction(self, conn: Connection, run: ClaimedRun, request: JudgeInput) -> None:
        if self.store.identity_source is None:
            raise RuntimeFault("JUDGE_IDENTITY_SOURCE_UNAVAILABLE", 503)
        current = await asyncio.to_thread(self.store.identity_source.current, run.principal)
        if (not run.principal.roles <= current.roles
                or not run.principal.resource_scopes <= current.resource_scopes):
            raise RuntimeFault("EXECUTION_AUTHORITY_REVOKED", 403)
        actor = run.principal
        judge, judge_session = await self.store._authorized_run(conn, actor, run.run_id, lock=True)
        try:
            stored_judge_request = RunRequest.model_validate(judge["request"])
            stored_judge_actor = Principal.model_validate(judge["principal"])
        except (TypeError, ValueError):
            raise RuntimeFault("JUDGE_RUN_BINDING_INVALID", 409) from None
        if judge_session["evaluation_context_id"] is None:
            raise RuntimeFault("EVALUATION_RUN_REQUIRED", 403)
        judge_context = await self.store._evaluation(conn, actor, judge_session["evaluation_context_id"])
        if (judge["manifest_hash"] != run.manifest.content_hash()
                or stored_judge_request != run.request or stored_judge_actor != run.principal
                or stored_judge_request.business_hash() != judge["request_hash"]
                or judge_context.dataset_hash != self.frozen.dataset_hash
                or judge_context.gate_policy_version != self.gate_policy.version):
            raise RuntimeFault("JUDGE_CONTEXT_BINDING_MISMATCH", 403)
        target, session = await self.store._authorized_run(
            conn, actor, request.target_run_id, lock=True, require_installed=False,
        )
        if session["evaluation_context_id"] is None:
            raise RuntimeFault("EVALUATION_RUN_REQUIRED", 403)
        context = await self.store._evaluation(conn, actor, session["evaluation_context_id"],
                                               require_installed=False)
        release = await self.store._release(conn, actor, session["release_id"])
        manifest = ReleaseManifest.model_validate(release["manifest"])
        source_ref = manifest.profiles.as_object().get("resource_authority_ref")
        try:
            stored_target_request = RunRequest.model_validate(target["request"])
        except (TypeError, ValueError):
            raise RuntimeFault("JUDGE_TARGET_BINDING_MISMATCH", 403) from None
        if (source_ref != self.material_authority_ref or request.material_source_ref != source_ref
                or manifest.content_hash() != target["manifest_hash"]
                or request.target_manifest_hash != target["manifest_hash"]
                or context.dataset_hash != self.frozen.dataset_hash
                or context.gate_policy_version != self.gate_policy.version
                or context.resource_scopes != judge_context.resource_scopes
                or target["request_hash"] != request.input_hash
                or stored_target_request.business_hash() != target["request_hash"] or target["status"] != "succeeded"
                or target["event_seq"] != request.event_seq or target["result"] is None
                or canonical_hash(target["result"]) != request.target_result_hash
                or canonical_hash(request.target_result) != request.target_result_hash):
            raise RuntimeFault("JUDGE_TARGET_BINDING_MISMATCH", 403)
        material = await self.material_authority(conn, actor, target, session)
        if canonical_hash(material) != canonical_hash(request.reference_material):
            raise RuntimeFault("JUDGE_MATERIAL_CHANGED", 403)

    async def authorize(self, run: ClaimedRun) -> None:
        request = self.input_for(run)
        async with self.store._transaction() as conn:
            await self.authorize_in_transaction(conn, run, request)
            await self.store._audit(conn, run.principal, "evaluation.judge_target_authorized", canonical_hash({
                "judge_run_id": run.run_id, "target_run_id": request.target_run_id,
                "freeze_hash": request.freeze_hash, "result_hash": request.target_result_hash,
                "material_hash": canonical_hash(request.reference_material), "event_seq": request.event_seq,
            }))

    async def authorize_public_read(self, conn: Connection, actor: Principal, row: Row, session: Row) -> None:
        release = await self.store._release(conn, actor, session["release_id"])
        run = ClaimedRun(
            run_id=row["run_id"], session_id=session["session_id"],
            principal=Principal.model_validate(row["principal"]), request=RunRequest.model_validate(row["request"]),
            manifest=ReleaseManifest.model_validate(release["manifest"]),
            execution_deadline=row["execution_deadline"] or row["expires_at"], history=JsonSnapshot.capture([]),
        )
        request = self.input_for(run)
        await self.authorize_in_transaction(conn, run, request)
