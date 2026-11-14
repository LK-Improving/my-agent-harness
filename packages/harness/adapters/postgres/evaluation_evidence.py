"""当前授权范围内的评测账本及受限 Judge 材料读取。"""

from typing import TYPE_CHECKING

from harness.contracts.evaluation_gates import (
    DispatchEvidence, JudgeModelEnvelope, JudgeModelResponse, JudgeSourceMaterial, ModelCallEvidence,
    OperationEvidence, RunGateEvidence,
)
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.models import ModelResult, PreparedModelRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AnswerCandidate, Principal, RuntimeFault
from harness.adapters.postgres.connection import Row
from harness.evaluation.assets import decode

if TYPE_CHECKING:
    from harness.adapters.runtime_postgres import PostgresRuntimeStore


class PostgresEvaluationEvidence:
    def __init__(self, store: "PostgresRuntimeStore") -> None:
        self.store = store

    async def judge_material(self, actor: Principal, run_id: str) -> JudgeSourceMaterial:
        """由已安装的资源权威读取参考材料，撤权或审计失败时阻断。"""
        self.store._role(actor, "evaluator")
        async with self.store._transaction() as conn:
            run, session = await self.store._authorized_run(conn, actor, run_id, lock=True)
            if session["evaluation_context_id"] is None:
                raise RuntimeFault("EVALUATION_RUN_REQUIRED", 403)
            await self.store._authorize_public_resource(conn, actor, run, session)
            context = await self.store._evaluation(conn, actor, session["evaluation_context_id"])
            release = await self.store._release(conn, actor, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            source_ref = manifest.profiles.as_object().get("resource_authority_ref")
            if (not isinstance(source_ref, str) or self.store.judge_material_authority is None
                    or self.store.judge_material_authority_ref != source_ref):
                raise RuntimeFault("JUDGE_MATERIAL_AUTHORITY_UNAVAILABLE", 503)
            if run["status"] != "succeeded" or run["result"] is None:
                raise RuntimeFault("JUDGE_TARGET_NOT_SUCCEEDED", 409)
            material = await self.store.judge_material_authority(conn, actor, run, session)
            try:
                material_size = len(canonical_json(material).encode("utf-8"))
            except (TypeError, ValueError):
                raise RuntimeFault("JUDGE_MATERIAL_INVALID", 503) from None
            if material_size > 1048576:
                raise RuntimeFault("JUDGE_MATERIAL_LIMIT_EXCEEDED", 413)
            response = JudgeSourceMaterial(
                run_id=run_id, context_id=context.context_id, dataset_hash=context.dataset_hash,
                manifest_hash=run["manifest_hash"], input_hash=run["request_hash"],
                result_hash=canonical_hash(run["result"]), event_seq=run["event_seq"],
                source_ref=source_ref, material=material, material_hash=canonical_hash(material),
            )
            await self.store._audit(conn, actor, "evaluation.judge_material_read", canonical_hash({
                "run_id": run_id, "material_hash": response.material_hash, "event_seq": response.event_seq,
            }))
            return response

    async def judge_response(self, actor: Principal, run_id: str, call_id: str) -> JudgeModelResponse:
        """只导出已提交 Judge 草稿对应的模型响应，并记录读取审计。"""
        self.store._role(actor, "evaluator")
        async with self.store._transaction() as conn:
            run, session = await self.store._authorized_run(conn, actor, run_id, lock=True)
            if session["evaluation_context_id"] is None:
                raise RuntimeFault("EVALUATION_RUN_REQUIRED", 403)
            await self.store._authorize_public_resource(conn, actor, run, session)
            release = await self.store._release(conn, actor, session["release_id"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            profile = manifest.profiles.as_object().get("judge")
            if (not isinstance(profile, dict) or not isinstance(profile.get("implementation"), str)
                    or not profile["implementation"] or manifest.spec.allowed_tools or manifest.spec.knowledge_bindings):
                raise RuntimeFault("JUDGE_RUN_REQUIRED", 403)
            if run["status"] != "succeeded" or not isinstance(run["result"], dict):
                raise RuntimeFault("JUDGE_RESPONSE_NOT_AVAILABLE", 409)
            row = await self.store._one(conn,
                "SELECT m.state AS call_state,m.purpose AS call_purpose,m.request_fingerprint,m.request,"
                "octet_length(m.result->>'content') AS response_bytes,"
                "d.state AS draft_state,d.draft_id,d.candidate,d.candidate_hash "
                "FROM harness.model_calls m JOIN harness.answer_drafts d USING(run_id) "
                "WHERE m.run_id=%s AND m.model_call_id=%s FOR SHARE OF m,d", (run_id, call_id))
            if (row["call_state"] != "succeeded" or row["call_purpose"] != "answer"
                    or row["draft_state"] != "committed"
                    or row["response_bytes"] is None or row["response_bytes"] > 1048576):
                raise RuntimeFault("JUDGE_RESPONSE_NOT_AVAILABLE", 409)
            raw = await self.store._one(conn,
                "SELECT result FROM harness.model_calls WHERE run_id=%s AND model_call_id=%s", (run_id, call_id))
            try:
                candidate = AnswerCandidate.model_validate(row["candidate"])
                request = PreparedModelRequest.model_validate(row["request"])
                result = ModelResult.model_validate(raw["result"])
                if not result.content:
                    raise ValueError("模型响应为空")
                envelope = JudgeModelEnvelope.model_validate(decode(result.content.encode("utf-8")))
            except (TypeError, ValueError):
                raise RuntimeFault("JUDGE_RESPONSE_INVALID", 409) from None
            principal = Principal.model_validate(run["principal"])
            if (request.context.run_id != run_id or request.context.tenant_id != principal.tenant_id
                    or request.context.subject_id != principal.subject_id
                    or request.context.application_id != principal.application_id
                    or request.context.roles != principal.roles
                    or request.context.resource_scopes != principal.resource_scopes
                    or request.context.release_id != manifest.spec.release_id
                    or request.manifest_hash != run["manifest_hash"]
                    or request.environment_profile_hash != run["environment_profile_hash"]
                    or request.tools.as_object_array() or result.tool_calls.as_object_array()
                    or candidate.model_call_id != call_id or candidate.model_attempt_id != request.model_attempt_id
                    or candidate.draft_id != row["draft_id"] or candidate.draft_id != request.draft_id
                    or candidate.model_call_id != request.model_call_id or result.model_call_id != call_id
                    or result.model_attempt_id != request.model_attempt_id
                    or result.request_fingerprint != row["request_fingerprint"]
                    or request.fingerprint() != row["request_fingerprint"]
                    or candidate.answer != envelope.answer or run["result"].get("answer") != candidate.answer
                    or envelope.evidence_refs or candidate.task.evidence_refs
                    or candidate.task.source_turn_id != run_id
                    or candidate.answer_kind != envelope.answer_kind or candidate.task_outcome != envelope.task_outcome
                    or run["result"].get("answer_kind") != envelope.answer_kind
                    or run["result"].get("task_outcome") != envelope.task_outcome
                    or canonical_hash(candidate.validation_input()) != row["candidate_hash"]):
                raise RuntimeFault("JUDGE_RESPONSE_BINDING_MISMATCH", 409)
            response = JudgeModelResponse(run_id=run_id, model_call_id=call_id, content=result.content,
                                          content_hash=canonical_hash(result.content))
            await self.store._audit(conn, actor, "evaluation.judge_response_read", canonical_hash({
                "run_id": run_id, "model_call_id": call_id, "content_hash": response.content_hash,
            }))
            return response

    async def inspect(self, actor: Principal, run_id: str) -> RunGateEvidence:
        self.store._role(actor, "evaluator")
        async with self.store._transaction() as conn:
            run, session = await self.store._authorized_run(conn, actor, run_id, lock=True)
            if session["evaluation_context_id"] is None:
                raise RuntimeFault("EVALUATION_RUN_REQUIRED", 403)
            await self.store._authorize_public_resource(conn, actor, run, session)
            context = await self.store._evaluation(conn, actor, session["evaluation_context_id"])
            release = await self.store._release(conn, actor, session["release_id"])
            profiles = release["manifest"]["profiles"]
            execution = profiles.get("execution", {})
            budget = execution.get("budget", {})
            missing: list[str] = []

            async def rows(sql: str, limit: int = 1000) -> list[Row]:
                result = await (await conn.execute(sql, (run_id, limit + 1))).fetchall()
                if len(result) > limit:
                    raise RuntimeFault("EVALUATION_EVIDENCE_LIMIT", 422)
                return result

            operations = []
            writes = await rows("SELECT o.operation_id,o.tool,o.principal,o.args_hash,o.action_key,o.attempts,"
                "o.approval_id,o.binding_hash,o.state,jsonb_build_object('args_hash',o.receipt->'args_hash',"
                "'external_reference',o.receipt->'external_reference') AS receipt,"
                "a.binding_hash AS approved_binding,a.state AS approval_state,"
                "a.expires_at AS approval_expires,a.decided_by FROM harness.operations o "
                "LEFT JOIN harness.approvals a USING(approval_id) WHERE o.run_id=%s ORDER BY operation_id LIMIT %s")
            attempt_rows = await rows("SELECT t.* FROM harness.tool_attempts t JOIN harness.operations o "
                "USING(operation_id) WHERE o.run_id=%s ORDER BY t.operation_id,t.ordinal LIMIT %s")
            for op in writes:
                receipt = op["receipt"] or {}
                tool = op["tool"]
                operations.append(OperationEvidence(operation_id=op["operation_id"],
                    tool_name=tool["name"], tool_hash=canonical_hash(tool), effect="write", state=op["state"],
                    resource_scopes=tuple(sorted(op["principal"]["resource_scopes"])), args_hash=op["args_hash"],
                    receipt_args_hash=receipt.get("args_hash"), action_key=op["action_key"],
                    recorded_attempts=op["attempts"], max_attempts=tool["retry"]["max_attempts_per_operation"],
                    attempts=tuple(DispatchEvidence(attempt_id=t["attempt_id"], ordinal=t["ordinal"],
                        state=t["state"], dispatched_at=t["dispatched_at"])
                        for t in attempt_rows if t["operation_id"] == op["operation_id"]),
                    approval_id=op["approval_id"], binding_hash=op["binding_hash"],
                    approval_binding_hash=op["approved_binding"], approval_state=op["approval_state"],
                    approval_expires_at=op["approval_expires"], approver=op["decided_by"],
                    external_reference_hash=canonical_hash(receipt["external_reference"])
                        if receipt.get("external_reference") else None))
            reads = await rows("SELECT operation_id,tool_name,spec_hash,args_hash,state,"
                "receipt->>'args_hash' AS receipt_args_hash FROM harness.read_operations "
                "WHERE run_id=%s ORDER BY operation_id LIMIT %s")
            for op in reads:
                operations.append(OperationEvidence(operation_id=op["operation_id"], tool_name=op["tool_name"],
                    tool_hash=op["spec_hash"], effect="read", state=op["state"],
                    resource_scopes=tuple(sorted(run["principal"]["resource_scopes"])), args_hash=op["args_hash"],
                    receipt_args_hash=op["receipt_args_hash"], recorded_attempts=1, max_attempts=1))
            calls = await rows("SELECT model_call_id,state,request->'budget' AS budget,"
                "request->'route' AS route,result->'usage' AS usage,result->>'content' AS response_content "
                "FROM harness.model_calls WHERE run_id=%s ORDER BY model_call_id LIMIT %s")
            model_calls = tuple(ModelCallEvidence(call_id=call["model_call_id"], state=call["state"],
                provider=(call["route"] or {}).get("provider"), model=(call["route"] or {}).get("model"),
                price_version_ref=(call["route"] or {}).get("price_version_ref"),
                input_tokens=call["budget"]["input_tokens"],
                reserved_output_tokens=call["budget"]["output_reserved_tokens"],
                remaining_total_tokens=call["budget"]["remaining_total_tokens"],
                reported_input_tokens=(call["usage"] or {}).get("input_tokens"),
                output_tokens=(call["usage"] or {}).get("output_tokens"),
                response_content_hash=canonical_hash(call["response_content"])
                    if call["response_content"] is not None else None) for call in calls)
            extension_tokens = 0
            retrieval_children = None
            rag = await self.store._one(conn, "SELECT to_regclass('harness_rag.retrieval_requests') IS NOT NULL AS present")
            if rag["present"]:
                reservation = await self.store._one(conn, "SELECT COALESCE(sum(parent_reserved_tokens),0) AS tokens "
                    "FROM harness_rag.retrieval_requests WHERE run_id=%s", (run_id,))
                extension_tokens = int(reservation["tokens"])
            manifest = ReleaseManifest.model_validate(release["manifest"])
            if rag["present"] and self.store.retrieval_evidence_authority is not None:
                retrieval_children = await self.store.retrieval_evidence_authority(conn, actor, run, session)
            elif extension_tokens or manifest.spec.knowledge_bindings:
                # 未安装扩展权威或旧schema时，不从父预留推断子调用状态。
                missing.append("retrieval_child_ledger")
            events = await rows("SELECT * FROM harness.events WHERE run_id=%s ORDER BY seq LIMIT %s", 10000)
            commit = await (await conn.execute("SELECT validation FROM harness.commits WHERE run_id=%s", (run_id,))).fetchone()
            validation = commit["validation"] if commit else None
            now = await self.store._one(conn, "SELECT clock_timestamp() AS now")
            return RunGateEvidence(run_id=run_id, session_id=session["session_id"], context_id=context.context_id,
                dataset_hash=context.dataset_hash, gate_policy_version=context.gate_policy_version,
                request_hash=run["request_hash"],
                manifest_hash=run["manifest_hash"], environment_profile_hash=run["environment_profile_hash"],
                snapshot_at=now["now"], status=run["status"], requester_id=run["principal"]["subject_id"],
                event_seq=run["event_seq"],
                result_hash=canonical_hash(run["result"]) if run["result"] is not None else None,
                run_scopes=tuple(sorted(run["principal"]["resource_scopes"])),
                context_scopes=tuple(sorted(context.resource_scopes)), operations=tuple(operations),
                model_calls=model_calls, model_steps=run["model_steps"], reserved_tokens=run["model_reserved_tokens"],
                tool_dispatches=run["tool_dispatches"], execution_used_ms=run["execution_used_ms"],
                max_input_tokens=budget.get("max_input_tokens"), max_output_tokens=budget.get("max_output_tokens"),
                max_total_tokens=budget.get("max_total_tokens"), max_execution_ms=budget.get("execution_budget_ms"),
                reserved_extension_tokens=extension_tokens,
                validation_checks=validation.get("checks") if validation else None,
                validation_verdict=validation.get("verdict") if validation else None,
                events=tuple(RuntimeEvent.model_validate(e) for e in events if e["type"] == "approval.decided"),
                event_hashes=tuple(canonical_hash(RuntimeEvent.model_validate(e).model_dump(mode="json"))
                                   for e in events), missing=tuple(missing), retrieval_children=retrieval_children)
