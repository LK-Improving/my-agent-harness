"""真实PG成功历史、摘要父预算/目的隔离与失败不污染定义；本批未运行。"""

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from harness.adapters.artifacts_postgres import PostgresArtifacts
from harness.adapters.context_postgres import PostgresContextSource
from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.runtime_postgres import PostgresRuntimeStore
from harness.adapters.summaries_postgres import PostgresSummaryStore
from harness.context.pipeline import ContextPipeline, summary_messages
from harness.contracts.artifacts import ArtifactProfile
from harness.contracts.context import ModelCallPurpose, SummaryCommitProfile, source_hash
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.models import ModelResult, ModelUsage
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AdmissionProfile, AnswerCandidate, EvaluationRequest, RuntimeFault, SessionRequest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.state import TaskState, ValidationReport
from harness.runtime.models import GovernedModelGateway
from tests.unit.test_context_pipeline import PROMPT, context_values
from tests.integration.test_artifacts_postgres import artifact_case as artifact_case

pytestmark = pytest.mark.integration


def candidate(run, *, answer="请补充信息。", call="controlled", attempt="controlled", draft="controlled"):
    value = AnswerCandidate(draft_id=draft, model_call_id=call, model_attempt_id=attempt, answer=answer,
        answer_kind="clarification", task_outcome="needs_input", task=TaskState(objective=run.request.message, source_turn_id=run.run_id),
        validation=ValidationReport(validator_ref=run.manifest.spec.completion_validator_ref, input_hash="0" * 64,
            verdict="needs_clarification", checks={"controlled_fixture": True}, checked_at=datetime.now(UTC)))
    return value.model_copy(update={"validation": value.validation.model_copy(update={"input_hash": canonical_hash(value.validation_input())})})


@pytest_asyncio.fixture
async def context_case(prepared, request):
    _, actor, original, _ = prepared
    _, execution, profile, _, _, _ = context_values()
    summary_mode = getattr(request, "param", None)
    persist = summary_mode in {"persist", "persist_expire"}
    store_type = PostgresExecutionStore if persist else PostgresRuntimeStore
    base = store_type(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    spec = original.spec.model_copy(update={"release_id": "context-test", "allowed_tools": ()})
    manifest = ReleaseManifest.model_validate({**original.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
        "tool_schema_hashes": {}, "model_profile": execution.route.model_dump(mode="json"),
        "profiles": {"execution": execution.model_dump(mode="json"), "context": profile.model_dump(mode="json"),
            "artifacts": ArtifactProfile().model_dump(mode="json")}})
    if persist:
        manifest = ReleaseManifest.model_validate({**manifest.model_dump(mode="json"), "profiles": {
            **manifest.profiles.as_object(), "summary_commit": SummaryCommitProfile(
                stage_ttl_seconds=1 if summary_mode == "persist_expire" else 900).model_dump(mode="json")}})
        base.summary_authority = PostgresSummaryStore(base).commit_in_transaction
        base.summary_authority_ref = "summary-commit@1"
    await base.register_release(actor, manifest)
    evaluation = await base.create_evaluation(actor, EvaluationRequest(candidate_release_id=spec.release_id,
        dataset_hash="e" * 64, gate_policy_version="context-test@1", resource_scopes=actor.resource_scopes,
        expires_at=datetime.now(UTC) + timedelta(hours=1)))
    session = await base.create_session(actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=evaluation.context_id))
    owner = uuid4().hex
    await base.register_worker(owner)
    current = None
    try:
        # 控制历史通过实际业务commit建立；不把fixture候选当真实模型或质量验收。
        for revision in range(3):
            await base.create_run(actor, session, RunRequest(request_id=f"history-{revision}", message=f"旧任务{revision}", expected_revision=revision))
            old = await base.claim(owner)
            assert old is not None and old.session_id == session
            history_candidate = candidate(old, answer="已确认的用户约束。" * 20, draft=f"draft-{revision}")
            if revision == 2:
                history_candidate = history_candidate.model_copy(update={"task": history_candidate.task.model_copy(
                    update={"pending_items": ("用户需补充日期",)})})
                history_candidate = history_candidate.model_copy(update={"validation": history_candidate.validation.model_copy(
                    update={"input_hash": canonical_hash(history_candidate.validation_input())})})
            if persist:
                reservation = await base.reserve_model(old, owner, JsonSnapshot.capture([
                    {"role": "user", "content": old.request.message}]), JsonSnapshot.capture([]), execution)
                prepared = reservation.request
                await base.save_model_result(prepared, ModelResult(model=execution.route.model, content=history_candidate.answer,
                    finish_reason="stop", request_id="seed", usage=ModelUsage(input_tokens=10, output_tokens=10),
                    model_call_id=prepared.model_call_id, model_attempt_id=prepared.model_attempt_id,
                    request_fingerprint=prepared.fingerprint(), draft_id=prepared.draft_id))
                history_candidate = history_candidate.model_copy(update={"model_call_id": prepared.model_call_id,
                    "model_attempt_id": prepared.model_attempt_id, "draft_id": prepared.draft_id})
            await base.complete(owner, old, history_candidate)
            await base.finish_slot(owner)
        store = PostgresExecutionStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
        summaries = PostgresSummaryStore(store) if persist else None
        if summaries is not None:
            store.summary_authority = summaries.commit_in_transaction
            store.summary_authority_ref = "summary-commit@1"
        await store.create_run(actor, session, RunRequest(request_id="current", message="更正条件后继续处理", expected_revision=3))
        current = await store.claim(owner)
        assert current is not None and current.session_id == session
        source, artifacts = PostgresContextSource(store), PostgresArtifacts(store, ArtifactProfile())

        class Transport:
            calls = 0
            summary_content = '{"summary":"历史确认了约束，保留用户纠错。"}'
            fail_summary = False

            async def complete(self, request):
                self.calls += 1
                is_summary = request.messages.as_object_array()[0]["content"] == PROMPT
                if is_summary and self.fail_summary:
                    raise RuntimeFault("CONTROLLED_TRANSPORT_FAILURE", 503)
                return ModelResult(model=execution.route.model, content=self.summary_content if is_summary else "受控回答",
                    tool_calls=JsonSnapshot.capture([]), finish_reason="stop", request_id="test",
                    usage=ModelUsage(input_tokens=10, output_tokens=10), model_call_id=request.model_call_id,
                    model_attempt_id=request.model_attempt_id, request_fingerprint=request.fingerprint(), draft_id=request.draft_id)

        transport = Transport()
        gateway = GovernedModelGateway(store, transport, execution)
        pipeline = ContextPipeline(source, artifacts, gateway.complete_summary, execution, profile, PROMPT, summaries)
        gateway.context_builder = pipeline

        async def authority(run, worker):
            await pipeline.authorize(run, worker)
            await artifacts.authorize_context(run, worker)

        gateway.context_authority = authority
        messages = JsonSnapshot.capture([{"role": "system", "content": "固定规则"}, *current.history.as_object_array(),
            {"role": "user", "content": current.request.message}])
        yield store, actor, owner, current, execution, profile, source, pipeline, gateway, transport, messages
    finally:
        if current is not None:
            await base.fail(owner, current.run_id, "TEST_FINISHED")
        await base.finish_slot(owner)
        await base.release_worker(owner)


async def test_summary_uses_parent_ledger_and_replay_reuses_both_model_receipts(context_case):
    store, _, owner, run, _, _, _, _, gateway, transport, messages = context_case
    first = await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    assert await gateway.complete(run, owner, messages, JsonSnapshot.capture([])) == first
    assert transport.calls == 2
    async with store._transaction() as conn:
        rows = await (await conn.execute("SELECT purpose,purpose_ref,purpose_binding,request FROM harness.model_calls WHERE run_id=%s ORDER BY purpose", (run.run_id,))).fetchall()
        assert [row["purpose"] for row in rows] == ["answer", "context_summary"]
        assert rows[1]["purpose_ref"] and rows[1]["purpose_binding"]["source_revision"] == 3
        assert "purpose" not in rows[1]["request"]
        state = await store._one(conn, "SELECT model_steps,model_reserved_tokens FROM harness.runs WHERE run_id=%s", (run.run_id,))
        assert state["model_steps"] == 2 and state["model_reserved_tokens"] > 0
        session = await store._one(conn, "SELECT revision FROM harness.sessions WHERE session_id=%s", (run.session_id,))
        assert session["revision"] == 3
        assert await (await conn.execute("SELECT 1 FROM harness.turns WHERE run_id=%s", (run.run_id,))).fetchone() is None


@pytest.mark.parametrize("failure", ["unknown", "invalid_json"])
async def test_summary_failure_never_advances_history_or_retries_unknown(context_case, failure):
    store, _, owner, run, _, _, _, _, gateway, transport, messages = context_case
    transport.fail_summary = failure == "unknown"
    transport.summary_content = "not-json"
    for _ in range(2):
        with pytest.raises(RuntimeFault):
            await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    assert transport.calls == 1
    async with store._transaction() as conn:
        rows = await (await conn.execute("SELECT purpose,state FROM harness.model_calls WHERE run_id=%s", (run.run_id,))).fetchall()
        assert rows == [{"purpose": "context_summary", "state": "unknown" if failure == "unknown" else "succeeded"}]
        state = await store._one(conn, "SELECT model_steps,model_reserved_tokens FROM harness.runs WHERE run_id=%s", (run.run_id,))
        assert state["model_steps"] == 1 and state["model_reserved_tokens"] > 0
        assert (await store._one(conn, "SELECT revision FROM harness.sessions WHERE session_id=%s", (run.session_id,)))["revision"] == 3


async def test_forged_sources_and_same_input_different_purpose_are_distinct(context_case):
    store, _, owner, run, execution, profile, source, _, _, _, _ = context_case
    snapshot = await source.load(run, owner, profile)
    older = snapshot.turns[:-profile.keep_tail_turns]
    purpose = ModelCallPurpose(source_revision=3, covered_turn_ids=tuple(t.run_id for t in older),
        source_hash=source_hash(3, older), summary_prompt_hash=profile.summary_prompt_hash)
    messages = summary_messages(PROMPT, 3, older, purpose.source_hash)
    for update in ({"source_hash": "0" * 64}, {"source_revision": 2}, {"covered_turn_ids": ("forged",)}):
        with pytest.raises(RuntimeFault, match="CONTEXT_SUMMARY_SOURCE_MISMATCH"):
            await store.reserve_purpose_model(run, owner, messages, JsonSnapshot.capture([]), execution, purpose.model_copy(update=update))
    child = await store.reserve_purpose_model(run, owner, messages, JsonSnapshot.capture([]), execution, purpose)
    answer = await store.reserve_model(run, owner, messages, JsonSnapshot.capture([]), execution)
    assert child.request.model_call_id != answer.request.model_call_id
    # 同一供应商输入hash仍不改变，区别只在父账本的目的身份。
    assert child.request.budget.measured_input_hash == answer.request.budget.measured_input_hash


async def test_source_revocation_blocks_summary_before_any_provider_dispatch(context_case, monkeypatch):
    store, _, owner, run, _, _, source, _, gateway, transport, messages = context_case
    snapshot = await source.load(run, owner, gateway.context_builder.profile)
    target = snapshot.turns[0].run_id
    original = store._authorize_public_resource

    async def revoke(conn, actor, row, session):
        if row["run_id"] == target:
            raise RuntimeFault("SOURCE_RESOURCE_REVOKED", 403)
        await original(conn, actor, row, session)

    monkeypatch.setattr(store, "_authorize_public_resource", revoke)
    with pytest.raises(RuntimeFault, match="SOURCE_RESOURCE_REVOKED"):
        await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    assert transport.calls == 0


async def test_summary_call_cannot_be_published_as_answer(context_case):
    store, _, owner, run, _, _, _, pipeline, _, _, messages = context_case
    built = await pipeline.build(run, owner, messages, JsonSnapshot.capture([]))
    summary = built.candidate
    assert "用户需补充日期" in built.messages.canonical_text
    async with store._transaction() as conn:
        call = await store._one(conn, "SELECT request FROM harness.model_calls WHERE model_call_id=%s", (summary.model_call_id,))
    forged = candidate(run, call=summary.model_call_id, attempt=summary.model_attempt_id, draft=call["request"]["draft_id"])
    with pytest.raises(RuntimeFault, match="SUMMARY_CANNOT_PUBLISH_ANSWER"):
        await store.stage_answer(owner, run, forged)


@pytest.mark.parametrize("limit", ["tokens", "steps"])
async def test_summary_respects_existing_parent_budget_and_steps(context_case, limit):
    store, _, owner, run, execution, _, _, _, gateway, transport, messages = context_case
    async with store._transaction() as conn:
        if limit == "tokens":
            await conn.execute("UPDATE harness.runs SET model_reserved_tokens=%s WHERE run_id=%s",
                               (execution.budget.max_total_tokens - 1, run.run_id))
        else:
            await conn.execute("UPDATE harness.runs SET model_steps=%s WHERE run_id=%s",
                               (run.manifest.spec.execution_limits.max_model_steps, run.run_id))
    with pytest.raises(RuntimeFault, match="MODEL_BUDGET_EXHAUSTED"):
        await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    assert transport.calls == 0
    async with store._transaction() as conn:
        assert await (await conn.execute("SELECT 1 FROM harness.model_calls WHERE run_id=%s", (run.run_id,))).fetchone() is None


async def test_tool_projection_rebuilds_authoritative_data_instead_of_trusting_message_payload(artifact_case):
    import json

    _, artifacts, reads, _, owner, run, _, _ = artifact_case
    receipt = await reads.execute(run, owner, "large_result", "source", {})
    info = (await artifacts.project(run, owner, receipt)).as_object()["artifact"]
    messages = JsonSnapshot.capture([
        {"role": "user", "content": run.request.message},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "type": "function",
            "function": {"name": "large_result", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call", "content": json.dumps({"operation_id": receipt.operation_id,
            "status": "succeeded", "data": {"forged": "不得成为事实"}})},
    ])
    projected = await artifacts.project_messages(run, owner, messages)
    data = json.loads(projected.as_object_array()[-1]["content"])["data"]
    assert data["artifact"]["artifact_ref"] == info["artifact_ref"] and "forged" not in data
    with pytest.raises(RuntimeFault):
        await artifacts.project_messages(run, owner, messages, source_run_id="not-committed")


async def test_claim_history_denial_fails_only_that_run_and_releases_worker_slot(context_case, monkeypatch):
    store, actor, owner, run, _, profile, source, _, _, _, _ = context_case
    snapshot = await source.load(run, owner, profile)
    target = snapshot.turns[0].run_id
    await store.fail(owner, run.run_id, "PREPARE_NEXT_CASE")
    await store.finish_slot(owner)
    queued = await store.create_run(actor, run.session_id,
        RunRequest(request_id="blocked-history", message="下一轮", expected_revision=3))
    original = store._authorize_public_resource

    async def revoked(conn, principal, row, session):
        if row["run_id"] == target:
            raise RuntimeFault("HISTORY_SOURCE_REVOKED", 403)
        await original(conn, principal, row, session)

    monkeypatch.setattr(store, "_authorize_public_resource", revoked)
    assert await store.claim(owner) is None
    view = await store.get_run(actor, queued.run_id)
    assert view.status == "failed" and view.error_code == "HISTORY_SOURCE_REVOKED"
    async with store._transaction() as conn:
        worker = await store._worker(conn, owner)
        assert worker["active_run_id"] is None and worker["state"] == "idle"
        session = await store._one(conn, "SELECT revision,active_run_id FROM harness.sessions WHERE session_id=%s", (run.session_id,))
        assert session == {"revision": 3, "active_run_id": None}
