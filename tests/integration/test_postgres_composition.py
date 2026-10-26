"""真实 PG 组合事务故障定义；直接构造派发状态，不代表真实写入/审批闭环。"""

import os
from uuid import uuid4

import pytest
from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.postgres.events import PostgresEventJournal
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.adapters.writes_postgres import PostgresWriteStore
from harness.contracts.hashing import RunRequest
from harness.contracts.runtime import AdmissionProfile, RuntimeFault
from psycopg.types.json import Jsonb

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "store_type", [PostgresWriteStore, PostgresExecutionStore, PostgresRecoveryStore]
)
async def test_unknown_termination_is_atomic_and_keeps_guard(prepared, store_type, monkeypatch):
    _, actor, manifest, session_id = prepared
    store = store_type(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    created = await store.create_run(
        actor,
        session_id,
        RunRequest(request_id="fault-state", message="受控故障", expected_revision=0),
    )
    owner, operation_id, action_key = uuid4().hex, uuid4().hex, uuid4().hex
    await store.register_worker(owner)
    run = await store.claim(owner)
    assert run is not None and run.run_id == created.run_id
    await store.started(owner, run, manifest.content_hash())
    has_drafts = store_type is not PostgresWriteStore
    try:
        async with store._transaction() as conn:
            # 仅制造持久化故障窗口，不调用供应商或伪造审批成功结论。
            await conn.execute(
                """INSERT INTO harness.operations(operation_id,run_id,logical_call_id,principal,
                tool,tool_hash,arguments,args_hash,action_key,plan_hash,binding_hash,approval_id,state,attempts)
                VALUES(%s,%s,'fault-call',%s,'{}',%s,'{}',%s,%s,%s,%s,%s,'dispatched',1)""",
                (
                    operation_id,
                    run.run_id,
                    Jsonb(actor.model_dump(mode="json")),
                    "1" * 64,
                    "2" * 64,
                    action_key,
                    "3" * 64,
                    "4" * 64,
                    uuid4().hex,
                ),
            )
            await conn.execute(
                "INSERT INTO harness.business_action_guards VALUES(%s,%s,%s,%s)",
                (actor.tenant_id, actor.application_id, action_key, operation_id),
            )
            if has_drafts:
                await conn.execute(
                    """INSERT INTO harness.answer_drafts
                    (run_id,draft_id,model_attempt_id,candidate_hash,candidate,state)
                    VALUES(%s,%s,'fault-attempt',%s,'{}','published')""",
                    (run.run_id, uuid4().hex, "5" * 64),
                )
            before = await store._one(
                conn, "SELECT revision,event_seq FROM harness.runs WHERE run_id=%s", (run.run_id,)
            )
        persist = PostgresEventJournal._persist

        async def fail_reconciling(conn, run_id, event_type, payload):
            await persist(conn, run_id, event_type, payload)
            if run_id == run.run_id and event_type == "run.reconciling":
                raise RuntimeError("controlled_termination_fault")

        with monkeypatch.context() as patch:
            patch.setattr(PostgresEventJournal, "_persist", staticmethod(fail_reconciling))
            with pytest.raises(RuntimeError, match="controlled_termination_fault"):
                await store.fail(owner, run.run_id, "FAULT")
        async with store._transaction() as conn:
            row = await store._one(
                conn,
                "SELECT revision,event_seq,status FROM harness.runs WHERE run_id=%s",
                (run.run_id,),
            )
            assert row == {**before, "status": "running"}
            op = await store._one(
                conn, "SELECT state FROM harness.operations WHERE operation_id=%s", (operation_id,)
            )
            assert op["state"] == "dispatched"
            if has_drafts:
                draft = await store._one(
                    conn, "SELECT state FROM harness.answer_drafts WHERE run_id=%s", (run.run_id,)
                )
                assert draft["state"] == "published"
        await store.fail(owner, run.run_id, "FAULT")
        assert (await store.get_run(actor, run.run_id)).status == "reconciling"
        async with store._transaction() as conn:
            session = await store._one(
                conn,
                "SELECT active_run_id FROM harness.sessions WHERE session_id=%s",
                (session_id,),
            )
            assert session["active_run_id"] == run.run_id
            op = await store._one(
                conn, "SELECT state FROM harness.operations WHERE operation_id=%s", (operation_id,)
            )
            assert op["state"] == "unknown"
            guard = await store._one(
                conn,
                "SELECT operation_id FROM harness.business_action_guards WHERE action_key=%s",
                (action_key,),
            )
            assert guard["operation_id"] == operation_id
        events = await store.events(actor, run.run_id, 0)
        types = [event["type"] for event in events]
        assert types.count("run.reconciling") == 1
        assert not {"run.failed", "run.completed"} & set(types)
        if has_drafts:
            assert types.index("answer.invalidated") < types.index("run.reconciling")
            with pytest.raises(RuntimeFault, match="ANSWER_DRAFT_INVALIDATED"):
                await store.candidate_for_run(actor, run.run_id)
        other = actor.model_copy(update={"tenant_id": "other"})
        with pytest.raises(RuntimeFault):
            await store.write_receipts(other, run.run_id)
        if has_drafts:
            with pytest.raises(RuntimeFault):
                await store.operation_state(other, operation_id)
    finally:
        await store.fail(owner, run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)
