"""M5-03 真实PG证据读取与独立写沙箱门禁定义；本轮未执行。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from psycopg.types.json import Jsonb

from harness.bootstrap.api import create_app
from harness.adapters.evaluation_files import read_freeze
from harness.adapters.postgres.judge_authority import PostgresJudgeAuthority
from harness.contracts.evaluation_contexts import EvaluationRevocationRequest
from harness.contracts.evaluation_gates import JudgeInput
from harness.contracts.hashing import RunRequest, canonical_hash, canonical_json
from harness.contracts.models import ModelResult, ModelUsage, PreparedModelRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, EvaluationRequest, RuntimeFault, SessionRequest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.state import TaskState, ValidationReport
from harness.contracts.writes import ApprovalDecisionRequest
from harness.evaluation.assets import cases, freeze_assets
from tests.integration.test_m2_writes import write_case as write_case
from tests.unit.test_m0_boundaries import prepared as prepared_model

pytestmark = pytest.mark.integration


async def test_evidence_endpoint_binds_current_evaluator_and_contains_no_request_body(prepared):
    store, actor, _, session = prepared
    run = await store.create_run(actor, session, RunRequest(request_id="gate-evidence", message="PRIVATE_SENTINEL", expected_revision=0))
    app = create_app(store, lambda token: actor)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/runs/{run.run_id}/evaluation-evidence")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body["run_id"] == run.run_id and body["event_seq"] == 1
        events = await store.events(actor, run.run_id, 0)
        assert body["event_hashes"] == [canonical_hash(event) for event in events]
        assert "PRIVATE_SENTINEL" not in response.text
        assert "messages" not in body and "arguments" not in body and body["events"] == []
    stranger = actor.model_copy(update={"application_id": "another-app"})
    with pytest.raises(RuntimeFault) as denied:
        await store.evaluation_evidence(stranger, run.run_id)
    assert denied.value.status == 404
    ordinary = actor.model_copy(update={"roles": frozenset({"user"})})
    with pytest.raises(RuntimeFault) as denied:
        await store.evaluation_evidence(ordinary, run.run_id)
    assert denied.value.status == 403


async def test_revoked_context_has_no_evaluation_evidence_bypass(prepared):
    store, actor, manifest, session = prepared
    run = await store.create_run(actor, session, RunRequest(request_id="revoke-evidence", message="查询", expected_revision=0))
    evidence = await store.evaluation_evidence(actor, run.run_id)
    await store.revoke_evaluation(actor, evidence.context_id, EvaluationRevocationRequest(
        request_id="revoke", expected_manifest_hash=manifest.content_hash()))
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await store.evaluation_evidence(actor, run.run_id)


async def test_evidence_distinguishes_model_reservation_from_reported_usage(prepared):
    store, actor, _, session = prepared
    run = await store.create_run(actor, session, RunRequest(
        request_id="model-usage-evidence", message="查询", expected_revision=0))
    request = {"budget": {"input_tokens": 100, "output_reserved_tokens": 40, "remaining_total_tokens": 200},
               "route": {"provider": "synthetic", "model": "synthetic-model", "price_version_ref": "price@1"}}
    result = {"usage": {"input_tokens": 31, "output_tokens": 9}, "content": "synthetic-response"}
    async with store._transaction() as conn:
        await conn.execute(
            "INSERT INTO harness.model_calls(model_call_id,run_id,input_hash,request_fingerprint,request,state,result) "
            "VALUES(%s,%s,%s,%s,%s,'succeeded',%s)",
            ("synthetic-call", run.run_id, "a" * 64, "b" * 64, Jsonb(request), Jsonb(result)),
        )
    evidence = await store.evaluation_evidence(actor, run.run_id)
    assert len(evidence.model_calls) == 1
    call = evidence.model_calls[0]
    assert (call.input_tokens, call.reserved_output_tokens) == (100, 40)
    assert (call.reported_input_tokens, call.output_tokens) == (31, 9)
    assert (call.provider, call.model, call.price_version_ref) == ("synthetic", "synthetic-model", "price@1")
    assert call.response_content_hash == canonical_hash("synthetic-response")


async def test_judge_model_response_is_authorized_bound_and_audited(prepared, monkeypatch):
    store, actor, original, _ = prepared
    spec = original.spec.model_copy(update={"agent_id": "judge", "release_id": "judge-" + actor.tenant_id,
                                            "allowed_tools": (), "knowledge_bindings": ()})
    manifest = ReleaseManifest.model_validate({**original.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
        "tool_schema_hashes": {}, "profiles": {"judge": {"implementation": "synthetic-judge@1"}}})
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(actor, EvaluationRequest(
        candidate_release_id=spec.release_id, dataset_hash="6" * 64, gate_policy_version="test@1",
        resource_scopes=actor.resource_scopes, expires_at=datetime.now(UTC) + timedelta(hours=1)))
    sid = await store.create_session(actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id))
    run = await store.create_run(actor, sid, RunRequest(request_id="judge-response", message="合成评分", expected_revision=0))
    base = prepared_model().model_dump(mode="json")
    base["context"].update({"tenant_id": actor.tenant_id, "subject_id": actor.subject_id,
                            "application_id": actor.application_id, "run_id": run.run_id, "release_id": spec.release_id,
                            "roles": sorted(actor.roles), "resource_scopes": sorted(actor.resource_scopes)})
    base.update({"manifest_hash": manifest.content_hash(), "environment_profile_hash": manifest.environment_profile_hash,
                 "model_call_id": "call-" + run.run_id, "model_attempt_id": "attempt-" + run.run_id,
                 "draft_id": "draft-" + run.run_id})
    request = PreparedModelRequest.model_validate(base)
    answer = '{"version":"evaluation-judge-output@1","scores":{"quality":0.8}}'
    content = canonical_json({"answer": answer, "answer_kind": "answer", "task_outcome": "achieved"})
    model_result = ModelResult(model="synthetic", content=content, finish_reason="stop", request_id=None,
        usage=ModelUsage(input_tokens=31, output_tokens=9), model_call_id=request.model_call_id,
        model_attempt_id=request.model_attempt_id, request_fingerprint=request.fingerprint())
    validation = ValidationReport(validator_ref=spec.completion_validator_ref, input_hash="0" * 64,
                                  verdict="pass", checks={"valid": True}, checked_at=datetime.now(UTC))
    candidate = AnswerCandidate(draft_id=request.draft_id, model_call_id=request.model_call_id,
        model_attempt_id=request.model_attempt_id, answer=answer, answer_kind="answer", task_outcome="achieved",
        task=TaskState(objective="合成评分", source_turn_id=run.run_id), validation=validation)
    candidate = candidate.model_copy(update={"validation": validation.model_copy(
        update={"input_hash": canonical_hash(candidate.validation_input())})})
    final = {"answer": answer, "answer_kind": "answer", "task_outcome": "achieved",
             "task": candidate.task.model_dump(mode="json")}
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.runs SET status='succeeded',result=%s WHERE run_id=%s", (Jsonb(final), run.run_id))
        await conn.execute(
            "INSERT INTO harness.model_calls(model_call_id,run_id,input_hash,request_fingerprint,request,state,result) "
            "VALUES(%s,%s,%s,%s,%s,'succeeded',%s)",
            (request.model_call_id, run.run_id, "c" * 64, request.fingerprint(),
             Jsonb(request.model_dump(mode="json")), Jsonb(model_result.model_dump(mode="json"))))
        await conn.execute(
            "INSERT INTO harness.answer_drafts(run_id,draft_id,model_attempt_id,candidate_hash,candidate,published_offset,state) "
            "VALUES(%s,%s,%s,%s,%s,%s,'committed')",
            (run.run_id, request.draft_id, request.model_attempt_id, canonical_hash(candidate.validation_input()),
             Jsonb(candidate.model_dump(mode="json")), len(answer)))
    app = create_app(store, lambda token: actor)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/runs/{run.run_id}/evaluation-judge-responses/{request.model_call_id}")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["content"] == content and response.json()["content_hash"] == canonical_hash(content)
    stranger = actor.model_copy(update={"subject_id": "stranger"})
    with pytest.raises(RuntimeFault):
        await store.judge_model_response(stranger, run.run_id, request.model_call_id)
    ordinary = actor.model_copy(update={"roles": frozenset({"user"})})
    with pytest.raises(RuntimeFault, match="FORBIDDEN"):
        await store.judge_model_response(ordinary, run.run_id, request.model_call_id)
    original_audit = store._audit

    async def reject_audit(*args):
        raise RuntimeFault("SYNTHETIC_AUDIT_FAILURE")

    monkeypatch.setattr(store, "_audit", reject_audit)
    with pytest.raises(RuntimeFault, match="SYNTHETIC_AUDIT_FAILURE"):
        await store.judge_model_response(actor, run.run_id, request.model_call_id)
    monkeypatch.setattr(store, "_audit", original_audit)
    async with store._transaction() as conn:
        audit = await store._one(conn, "SELECT count(*) AS count FROM harness.audit WHERE tenant_id=%s "
                                 "AND action='evaluation.judge_response_read'", (actor.tenant_id,))
        await conn.execute("UPDATE harness.model_calls SET result=jsonb_set(result,'{content}',%s) "
                           "WHERE model_call_id=%s", (Jsonb('{"answer":"different","answer_kind":"answer","task_outcome":"achieved"}'), request.model_call_id))
    assert audit["count"] == 1
    with pytest.raises(RuntimeFault, match="JUDGE_RESPONSE_BINDING_MISMATCH"):
        await store.judge_model_response(actor, run.run_id, request.model_call_id)
    await store.revoke_evaluation(actor, context.context_id, EvaluationRevocationRequest(
        request_id="judge-revoke", expected_manifest_hash=manifest.content_hash()))
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await store.judge_model_response(actor, run.run_id, request.model_call_id)


async def test_judge_response_endpoint_rejects_non_judge_run(prepared):
    store, actor, _, session = prepared
    run = await store.create_run(actor, session, RunRequest(request_id="not-judge", message="查询", expected_revision=0))
    with pytest.raises(RuntimeFault, match="JUDGE_RUN_REQUIRED"):
        await store.judge_model_response(actor, run.run_id, "unknown-call")


async def test_judge_material_endpoint_rechecks_authority_and_audits(prepared):
    store, actor, original, _ = prepared
    spec = original.spec.model_copy(update={"release_id": "material-" + actor.tenant_id})
    manifest = ReleaseManifest.model_validate({**original.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
        "profiles": {"resource_authority_ref": "synthetic-authority@1"}})
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(actor, EvaluationRequest(
        candidate_release_id=spec.release_id, dataset_hash="6" * 64, gate_policy_version="test@1",
        resource_scopes=actor.resource_scopes, expires_at=datetime.now(UTC) + timedelta(hours=1)))
    sid = await store.create_session(actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id))
    run = await store.create_run(actor, sid, RunRequest(request_id="judge-material", message="合成查询", expected_revision=0))
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.runs SET status='succeeded',result=%s WHERE run_id=%s",
                           (Jsonb({"answer": "合成答案"}), run.run_id))
    allowed = {"value": True}

    async def resource_authority(conn, subject, source_run, source_session):
        if not allowed["value"]:
            raise RuntimeFault("SYNTHETIC_RESOURCE_REVOKED", 403)

    async def material_authority(conn, subject, source_run, source_session):
        return {"reference": "合成材料"}

    store.resource_authority_ref = "synthetic-authority@1"
    store.resource_authority = resource_authority
    store.judge_material_authority_ref = "synthetic-authority@1"
    store.judge_material_authority = material_authority
    app = create_app(store, lambda token: actor)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/runs/{run.run_id}/evaluation-judge-material")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["material"] == {"reference": "合成材料"}
    assert body["material_hash"] == canonical_hash(body["material"])
    async with store._transaction() as conn:
        audit = await store._one(conn, "SELECT count(*) AS count FROM harness.audit WHERE tenant_id=%s "
                                 "AND action='evaluation.judge_material_read'", (actor.tenant_id,))
    assert audit["count"] == 1
    allowed["value"] = False
    with pytest.raises(RuntimeFault, match="SYNTHETIC_RESOURCE_REVOKED"):
        await store.judge_source_material(actor, run.run_id)


async def test_judge_rechecks_target_authority_after_switching_installed_release(prepared, monkeypatch):
    store, actor, original, _ = prepared
    target_spec = original.spec.model_copy(update={"release_id": "target-" + actor.tenant_id})
    target = ReleaseManifest.model_validate({**original.model_dump(mode="json"),
        "spec": target_spec.model_dump(mode="json"), "profiles": {"resource_authority_ref": "synthetic-material@1"}})
    judge_spec = original.spec.model_copy(update={"agent_id": "test-judge", "release_id": "judge-" + actor.tenant_id,
        "model_profile_ref": "judge-model@1", "allowed_tools": (), "knowledge_bindings": (),
        "execution_limits": original.spec.execution_limits.model_copy(update={"max_model_steps": 1})})
    judge = ReleaseManifest.model_validate({**original.model_dump(mode="json"),
        "spec": judge_spec.model_dump(mode="json"), "tool_schema_hashes": {},
        "profiles": {"judge": {"implementation": "synthetic-judge@1"}, "resource_authority_ref": "judge-target@1"}})
    _, blobs = read_freeze(Path(__file__).resolve().parents[2] / "evals/m5")
    blobs["judge.json"] = json.dumps({"version": "judge@1", "required": True,
        "implementation": "synthetic-judge@1", "prompt_sha256": judge.prompt_hash,
        "model_binding_ref": judge_spec.model_profile_ref, "judge_manifest_hash": judge.content_hash(),
        "dimensions": ["quality"]}).encode("utf-8")
    frozen = freeze_assets(blobs)
    row = cases(blobs["holdout.jsonl"])[0]
    await store.register_release(actor, target)
    await store.register_release(actor, judge)
    expiry = datetime.now(UTC) + timedelta(hours=1)
    context_args = {"dataset_hash": frozen.dataset_hash, "gate_policy_version": "m5-gates@1",
                    "resource_scopes": actor.resource_scopes, "expires_at": expiry}
    source_context = await store.create_evaluation(actor, EvaluationRequest(
        candidate_release_id=target_spec.release_id, **context_args))
    source_session = await store.create_session(actor, SessionRequest(
        agent_id=target_spec.agent_id, evaluation_context_id=source_context.context_id))
    source_request = RunRequest(
        request_id="target-run", message=row.message, expected_revision=0,
        attachment_versions=row.attachment_versions, business_options=row.business_options)
    source_run = await store.create_run(actor, source_session, source_request)
    target_result = {"answer": "合成答案", "answer_kind": "answer", "task_outcome": "achieved"}
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.runs SET status='succeeded',result=%s WHERE run_id=%s",
                           (Jsonb(target_result), source_run.run_id))
    material = {"reference": "合成参考材料"}
    request = JudgeInput(case_id=row.case_id, target_run_id=source_run.run_id,
        target_manifest_hash=target.content_hash(), freeze_hash=frozen.content_hash(), input_hash=row.input_hash(),
        target_result_hash=canonical_hash(target_result), event_seq=1, message=row.message,
        attachment_versions=row.attachment_versions, business_options=row.business_options, expected=row.expected,
        dimensions=("quality",), target_result=target_result, material_source_ref="synthetic-material@1",
        reference_material=material)
    judge_context = await store.create_evaluation(actor, EvaluationRequest(
        candidate_release_id=judge_spec.release_id, **context_args))
    judge_session = await store.create_session(actor, SessionRequest(
        agent_id=judge_spec.agent_id, evaluation_context_id=judge_context.context_id))
    state = {"allowed": True, "material": material}

    class CurrentIdentity:
        def current(self, principal):
            return principal if state["allowed"] else principal.model_copy(update={"resource_scopes": frozenset()})

    async def read_material(conn, principal, source, session):
        return state["material"]

    store.identity_source = CurrentIdentity()
    authority = PostgresJudgeAuthority(store, frozen, blobs, read_material, "synthetic-material@1")
    assert authority.ready_for(judge)
    store.installed_manifest_hashes = frozenset({judge.content_hash()})
    store.new_run_authority = authority.authorize_public_read
    store.resource_authority_ref = "judge-target@1"
    store.resource_authority = authority.authorize_public_read
    bad_input = request.model_copy(update={"expected": {"changed": True}})
    with pytest.raises(RuntimeFault, match="JUDGE_POLICY_OR_INPUT_MISMATCH"):
        await store.create_run(actor, judge_session, RunRequest(request_id="invalid-judge",
            message=canonical_json(bad_input.model_dump(mode="json")), expected_revision=0))
    async with store._transaction() as conn:
        session_row = await store._one(conn, "SELECT active_run_id FROM harness.sessions WHERE session_id=%s", (judge_session,))
        quota = await store._one(conn, "SELECT run_count FROM harness.evaluation_contexts WHERE context_id=%s",
                                 (judge_context.context_id,))
    assert session_row["active_run_id"] is None and quota["run_count"] == 0
    judge_request = RunRequest(request_id="judge-run", message=canonical_json(request.model_dump(mode="json")),
                               expected_revision=0)
    judge_run = await store.create_run(actor, judge_session, judge_request)
    claimed = ClaimedRun(run_id=judge_run.run_id, session_id=judge_session, principal=actor,
                         request=judge_request, manifest=judge, execution_deadline=expiry,
                         history=JsonSnapshot.capture([]))
    with pytest.raises(RuntimeFault, match="EXECUTABLE_RELEASE_NOT_INSTALLED"):
        await store.get_run(actor, source_run.run_id)
    await authority.authorize(claimed)
    assert (await store.get_run(actor, judge_run.run_id)).run_id == judge_run.run_id
    original_audit = store._audit

    async def reject_judge_audit(*args):
        raise RuntimeFault("SYNTHETIC_AUDIT_FAILURE")

    monkeypatch.setattr(store, "_audit", reject_judge_audit)
    with pytest.raises(RuntimeFault, match="SYNTHETIC_AUDIT_FAILURE"):
        await authority.authorize(claimed)
    monkeypatch.setattr(store, "_audit", original_audit)
    async with store._transaction() as conn:
        altered_request = source_request.model_copy(update={"message": "已替换的目标输入"})
        await conn.execute("UPDATE harness.runs SET request=%s WHERE run_id=%s",
                           (Jsonb(altered_request.model_dump(mode="json")), source_run.run_id))
    with pytest.raises(RuntimeFault, match="JUDGE_TARGET_BINDING_MISMATCH"):
        await authority.authorize(claimed)
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.runs SET request=%s WHERE run_id=%s",
                           (Jsonb(source_request.model_dump(mode="json")), source_run.run_id))
    state["material"] = {"reference": "被替换"}
    with pytest.raises(RuntimeFault, match="JUDGE_MATERIAL_CHANGED"):
        await authority.authorize(claimed)
    state["material"] = {"reference": "合成参考材料"}
    state["allowed"] = False
    with pytest.raises(RuntimeFault, match="EXECUTION_AUTHORITY_REVOKED"):
        await store.get_run(actor, judge_run.run_id)
    state["allowed"] = True
    await store.revoke_evaluation(actor, source_context.context_id, EvaluationRevocationRequest(
        request_id="revoke-source", expected_manifest_hash=target.content_hash()))
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await authority.authorize(claimed)


async def test_snapshot_records_exact_approval_unknown_attempt_and_real_sandbox_write(write_case):
    store, actor, approver, owner, run, _, operation, handler = write_case
    await store.decide_approval(approver, operation.approval_id, ApprovalDecisionRequest(
        request_id="gate-approve", binding_hash=operation.binding_hash, expected_revision=0, decision="approve"))
    permit = await store.begin_dispatch(actor, owner, operation.operation_id)
    with pytest.raises(RuntimeFault, match="SANDBOX_WRITE_UNCONFIRMED"):
        await handler.invoke(permit)
    await store.mark_write_unknown(permit)
    evidence = await store.evaluation_evidence(actor, run.run_id)
    op = next(op for op in evidence.operations if op.operation_id == operation.operation_id)
    assert op.state == "unknown" and op.recorded_attempts == len(op.attempts) == 1
    assert op.binding_hash == op.approval_binding_hash and op.approver == approver.subject_id
    assert any(event.type == "approval.decided" for event in evidence.events)
    async with handler._client() as client:
        response = await client.get("/counts", params={"tenant_id": actor.tenant_id, "application_id": actor.application_id})
    assert response.json()["writes"] == 1
    # 全租户实际计数只验证本隔离用例发生写入，不伪造成门禁的逐operation计数回执。
