"""官方PG saver+create_agent+业务账本集成定义；仅测试控制模型，不冒充真实供应商证据。"""

import json
import os
from datetime import UTC, datetime, timedelta
from functools import partial
from uuid import uuid4

import pytest
from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.postgres.events import PostgresEventJournal
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.bootstrap.checkpoints import saver
from harness.contracts.execution import ExecutionProfile
from harness.contracts.extensions import ExtensionDecision, ExtensionInvocation, ExtensionSpec
from harness.contracts.hashing import canonical_hash
from harness.contracts.model_envelope import chat_envelope_tokens, prepared_chat_envelope
from harness.contracts.models import ModelResult, ModelUsage, PreparedModelRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import (
    AdmissionProfile,
    EvaluationRequest,
    RuntimeFault,
    SessionRequest,
)
from harness.contracts.state import ValidationReport
from harness.graphs.agent import AgentEngine
from harness.runtime.extensions import ExtensionChain, RegisteredExtension
from harness.runtime.models import GovernedModelGateway
from harness.tools.reads import ReadCoordinator

from domain_tools.operations.queries import registered_order_query

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("store_type", [PostgresExecutionStore, PostgresRecoveryStore])
async def test_official_agent_saver_durable_model_read_and_unique_commit(
    prepared, store_type, monkeypatch
):
    _, original_actor, old, _ = prepared
    actor = original_actor.model_copy(
        update={"resource_scopes": frozenset({"work_orders:read:synthetic"})}
    )
    profile = ExecutionProfile.model_validate(
        {
            "route": {
                "provider": "controlled-test",
                "model": "controlled-test",
                "adapter_ref": "test@1",
                "configuration_hash": "f" * 64,
                "capabilities": ["tool_calling"],
                "capabilities_verified": True,
                "context_window_tokens": 100000,
                "max_output_tokens": 4096,
            },
            "budget": {
                "version": "test-budget@1",
                "max_input_tokens": 16000,
                "max_output_tokens": 4096,
                "max_total_tokens": 32000,
                "execution_budget_ms": 120000,
            },
        }
    )
    store = store_type(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    owner = uuid4().hex

    async def policy(invocation):
        return ExtensionDecision(
            extension_id="controlled-boundary",
            version="1",
            invocation_fingerprint=invocation.fingerprint(),
        )

    chain = ExtensionChain(
        tuple(
            RegisteredExtension(
                ExtensionSpec(
                    extension_id="controlled-boundary",
                    version="1",
                    handler_ref="test.boundary@1",
                    stage=stage,
                    order=0,
                ),
                policy,
            )
            for stage in ("before_model", "after_tool", "before_commit")
        ),
        record_decision=partial(store.record_extension, owner),
    )
    entry = registered_order_query()
    spec = old.spec.model_copy(
        update={
            "release_id": "controlled-execution-test",
            "allowed_tools": (old.spec.allowed_tools[0],),
        }
    )
    manifest = ReleaseManifest.model_validate(
        {
            **old.model_dump(mode="json"),
            "spec": spec.model_dump(mode="json"),
            "tool_schema_hashes": {"get_work_order@1": entry.spec.content_hash},
            "model_profile": profile.route.model_dump(mode="json"),
            "profiles": {
                "execution": profile.model_dump(mode="json"),
                "extensions": chain.snapshot.as_object(),
            },
        }
    )
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(
        actor,
        EvaluationRequest(
            candidate_release_id=spec.release_id,
            dataset_hash="e" * 64,
            gate_policy_version="test@1",
            resource_scopes=actor.resource_scopes,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
    )
    session_id = await store.create_session(
        actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id)
    )
    from harness.contracts.hashing import RunRequest

    await store.create_run(
        actor,
        session_id,
        RunRequest(request_id="r1", message="查询synthetic-demo", expected_revision=0),
    )
    await store.register_worker(owner)
    run = await store.claim(owner)
    assert run is not None and run.session_id == session_id
    await store.started(owner, run, manifest.content_hash())

    class ControlledTransport:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            from harness.contracts.snapshots import JsonSnapshot

            has_tool = request.messages.as_object_array()[-1]["role"] == "tool"
            content = (
                json.dumps(
                    {
                        "answer": "本地合成示例 synthetic-demo 状态为pending。"
                        + "这是合成示例说明。" * 180,
                        "answer_kind": "answer",
                        "task_outcome": "achieved",
                    },
                    ensure_ascii=False,
                )
                if has_tool
                else ""
            )
            tools = (
                []
                if has_tool
                else [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {
                            "name": "get_work_order",
                            "arguments": '{"order_id":"synthetic-demo"}',
                        },
                    }
                ]
            )
            return ModelResult(
                model="controlled-test",
                content=content,
                tool_calls=JsonSnapshot.capture(tools),
                finish_reason="stop",
                request_id="test",
                usage=ModelUsage(input_tokens=10, output_tokens=2048 if has_tool else 10),
                model_call_id=request.model_call_id,
                model_attempt_id=request.model_attempt_id,
                request_fingerprint=request.fingerprint(),
                draft_id=request.draft_id,
            )

    async def validate(current, candidate):
        receipts = await store.read_receipts(actor, current.run_id)
        return ValidationReport(
            validator_ref=spec.completion_validator_ref,
            input_hash=canonical_hash(candidate.validation_input()),
            verdict="pass",
            checks={"authorized_read": len(receipts) == 1},
            checked_at=datetime.now(UTC),
        )

    transport = ControlledTransport()
    try:
        async with saver(os.environ["HARNESS_M1_TEST_DSN"]) as checkpoint:
            # 专用测试库中显式安装官方迁移；不是应用启动时自动setup。
            from psycopg import AsyncConnection

            async with await AsyncConnection.connect(
                os.environ["HARNESS_M1_TEST_DSN"], autocommit=True
            ) as conn:
                await conn.execute("CREATE SCHEMA IF NOT EXISTS harness_checkpoint")
            await checkpoint.setup()
            engine = AgentEngine(
                store,
                GovernedModelGateway(store, transport, profile, chain),
                ReadCoordinator(store, (entry,), extensions=chain),
                None,
                checkpoint,
                owner,
                {spec.release_id: manifest},
                {spec.release_id: "受控测试。"},
                validate,
                extensions=chain,
            )
            candidate = await engine.execute(run)
            # 真PG审计先落库；同一指纹改变决定必须拒绝，不能影响已记录工具事实。
            async with store._transaction() as conn:
                facts = await (
                    await conn.execute(
                        (
                            "SELECT spec,invocation,decision FROM harness.extension_decisions "
                            "WHERE run_id=%s"
                        ),
                        (run.run_id,),
                    )
                ).fetchall()
            assert len(facts) >= 3
            fact = facts[0]
            hook_spec = ExtensionSpec.model_validate(fact["spec"])
            invocation = ExtensionInvocation.model_validate(fact["invocation"])
            original_decision = ExtensionDecision.model_validate(fact["decision"])
            await store.record_extension(owner, run, hook_spec, invocation, original_decision)
            with pytest.raises(RuntimeFault, match="EXTENSION_REPLAY_DECISION_CHANGED"):
                await store.record_extension(
                    owner,
                    run,
                    hook_spec,
                    invocation,
                    original_decision.model_copy(
                        update={"verdict": "deny", "reason_code": "changed-on-replay"}
                    ),
                )
            await store.stage_answer(owner, run, candidate)
            # 模拟候选/片段已落盘但业务commit前中断，不再向控制模型发起新请求。
            candidate = await engine.execute(run)
            await engine.before_commit(run, candidate)
            # 在完成事件已写入、事务提交前注入故障，验证跨模块仍是整笔业务事务。
            async with store._transaction() as conn:
                before = await store._one(
                    conn,
                    "SELECT revision,event_seq FROM harness.runs WHERE run_id=%s",
                    (run.run_id,),
                )
                session_before = await store._one(
                    conn,
                    "SELECT revision,active_run_id FROM harness.sessions WHERE session_id=%s",
                    (run.session_id,),
                )
            persist = PostgresEventJournal._persist

            async def fail_after_completion(conn, run_id, event_type, payload):
                await persist(conn, run_id, event_type, payload)
                if run_id == run.run_id and event_type == "run.completed":
                    raise RuntimeError("controlled_commit_fault")

            with monkeypatch.context() as patch:
                patch.setattr(PostgresEventJournal, "_persist", staticmethod(fail_after_completion))
                with pytest.raises(RuntimeError, match="controlled_commit_fault"):
                    await store.complete(owner, run, candidate)
            async with store._transaction() as conn:
                after = await store._one(
                    conn,
                    "SELECT revision,event_seq,status,result FROM harness.runs WHERE run_id=%s",
                    (run.run_id,),
                )
                assert after["revision"] == before["revision"]
                assert after["event_seq"] == before["event_seq"]
                assert after["status"] == "running" and after["result"] is None
                assert (
                    await store._one(
                        conn,
                        "SELECT revision,active_run_id FROM harness.sessions WHERE session_id=%s",
                        (run.session_id,),
                    )
                    == session_before
                )
                counts = await store._one(
                    conn,
                    (
                        "SELECT (SELECT count(*) FROM harness.commits WHERE run_id=%s) AS "
                        "commits,(SELECT count(*) FROM harness.turns WHERE run_id=%s) AS t"
                        "urns,(SELECT count(*) FROM harness.events WHERE run_id=%s AND typ"
                        "e='run.completed') AS completed"
                    ),
                    (run.run_id, run.run_id, run.run_id),
                )
                assert counts == {"commits": 0, "turns": 0, "completed": 0}
                draft = await store._one(
                    conn, "SELECT state FROM harness.answer_drafts WHERE run_id=%s", (run.run_id,)
                )
                assert draft["state"] == "published"
            await store.complete(owner, run, candidate)
            await store.complete(owner, run, candidate)
            assert transport.calls == 2
            async with store._transaction() as conn:
                calls = await (
                    await conn.execute(
                        "SELECT request FROM harness.model_calls WHERE run_id=%s",
                        (run.run_id,),
                    )
                ).fetchall()
            assert len(calls) == 2
            for call in calls:
                fixed = PreparedModelRequest.model_validate(call["request"])
                wire = prepared_chat_envelope(fixed)
                assert chat_envelope_tokens(wire) == fixed.budget.input_tokens
                assert "harness_draft_id" not in wire.canonical_text
            assert (await store.get_run(actor, run.run_id)).status == "succeeded"
            events = await store.events(actor, run.run_id, 0)
            assert sum(item["type"] == "answer.started" for item in events) == 1
            assert sum(item["type"] == "run.completed" for item in events) == 1
            deltas = [item["payload"]["delta"] for item in events if item["type"] == "answer.delta"]
            assert len(deltas) >= 3 and "".join(deltas) == candidate.answer
    finally:
        await store.fail(owner, run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)
