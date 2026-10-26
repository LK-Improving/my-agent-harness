"""PG 发布兼容/恢复入队定义；受控终止证明不代表真实进程、模型或沙箱验收。"""

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.contracts.execution import ProcessIdentity, TerminationProof
from harness.contracts.hashing import RunRequest
from harness.contracts.runtime import (
    AdmissionProfile,
    CancelRequest,
    EvaluationRequest,
    RuntimeFault,
    SessionRequest,
)

pytestmark = pytest.mark.integration


async def test_missing_old_release_is_not_claimed_or_changed(prepared):
    base, actor, old, old_session = prepared
    request = RunRequest(request_id="old", message="合成查询", expected_revision=0)
    old_run = await base.create_run(actor, old_session, request)
    new = old.model_copy(
        update={"spec": old.spec.model_copy(update={"release_id": "new-" + uuid4().hex})}
    )
    await base.register_release(actor, new)
    context = await base.create_evaluation(
        actor,
        EvaluationRequest(
            candidate_release_id=new.spec.release_id,
            dataset_hash="9" * 64,
            gate_policy_version="test@1",
            resource_scopes=actor.resource_scopes,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
    )
    session = await base.create_session(
        actor, SessionRequest(agent_id=new.spec.agent_id, evaluation_context_id=context.context_id)
    )
    new_run = await base.create_run(
        actor, session, RunRequest(request_id="new", message="合成查询", expected_revision=0)
    )
    store = PostgresRecoveryStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    store.installed_manifest_hashes = frozenset({new.content_hash()})
    owner = uuid4().hex
    await store.register_worker(owner)
    try:
        claimed = await store.claim(owner)
        assert claimed is not None and claimed.run_id == new_run.run_id
        after = await store.get_run(actor, old_run.run_id)
        assert after == old_run
        # 已有请求的幂等读回不受当前安装集合变化影响。
        assert await store.create_run(actor, old_session, request) == old_run
        await store.fail(owner, new_run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        assert await store.claim(owner) is None
    finally:
        await store.fail(owner, new_run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)
        await base.cancel_run(
            actor,
            old_run.run_id,
            CancelRequest(request_id="cleanup", expected_run_revision=old_run.revision),
        )
    with pytest.raises(RuntimeFault, match="EXECUTABLE_RELEASE_NOT_INSTALLED"):
        await store.create_run(
            actor, old_session, request.model_copy(update={"request_id": "old-new-request"})
        )


@pytest.mark.parametrize("abort", [False, True])
async def test_recovery_missing_implementation_does_not_reset_facts(prepared, abort):
    _, actor, manifest, session = prepared
    store = PostgresRecoveryStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    store.installed_manifest_hashes = frozenset({manifest.content_hash()})
    await store.create_run(
        actor, session, RunRequest(request_id="recover", message="合成查询", expected_revision=0)
    )
    owner = uuid4().hex
    process = ProcessIdentity(
        host_id="controlled",
        boot_time=1,
        pid=123,
        started_at=2,
        namespace_id="controlled",
        execution_class="local-no-subprocess@1",
        boot_id="controlled",
        start_marker="controlled",
    )
    store.process_identity = process
    await store.register_worker(owner)
    run = await store.claim(owner)
    assert run is not None
    await store.started(owner, run, manifest.content_hash())
    operator = actor.model_copy(
        update={
            "subject_id": "operator",
            "roles": frozenset({"recovery_operator"}),
            "resource_scopes": frozenset({"harness:recovery:local"}),
        }
    )
    # 仅验证存储协议，真实prove_terminated/进程排空另有待执行验收。
    await store.retire_worker(
        operator, owner, TerminationProof(process=process, reason="controlled-test")
    )
    async with store._transaction() as conn:
        # 合成累计计数，只用于检查恢复不会归零，不冒充已发生供应商调用。
        await conn.execute(
            (
                "UPDATE harness.runs SET "
                "execution_used_ms=500,model_reserved_tokens=42,model_steps=2,tool"
                "_dispatches=1 WHERE run_id=%s"
            ),
            (run.run_id,),
        )
    recovered, revision, _ = await store.recovery_run(operator, run.run_id)
    assert await store.release_for_run(operator, run.run_id) == manifest
    store.installed_manifest_hashes = frozenset()
    async with store._transaction() as conn:
        before = await store._one(
            conn,
            (
                "SELECT "
                "status,revision,event_seq,execution_used_ms,model_reserved_tokens"
                ",model_steps,tool_dispatches,recovery_required FROM harness.runs "
                "WHERE run_id=%s"
            ),
            (run.run_id,),
        )
    if abort:
        assert await store.recover_run(operator, recovered, revision, None, abort=True) == "failed"
    else:
        with pytest.raises(RuntimeFault, match="EXECUTABLE_RELEASE_NOT_INSTALLED"):
            await store.recover_run(operator, recovered, revision, None)
        async with store._transaction() as conn:
            assert (
                await store._one(
                    conn,
                    (
                        "SELECT "
                        "status,revision,event_seq,execution_used_ms,model_reserved_tokens"
                        ",model_steps,tool_dispatches,recovery_required FROM harness.runs "
                        "WHERE run_id=%s"
                    ),
                    (run.run_id,),
                )
                == before
            )
        store.installed_manifest_hashes = frozenset({manifest.content_hash()})
        assert await store.recover_run(operator, recovered, revision, None) == "queued"
    async with store._transaction() as conn:
        counts = await store._one(
            conn,
            (
                "SELECT "
                "execution_used_ms,model_reserved_tokens,model_steps,tool_dispatch"
                "es FROM harness.runs WHERE run_id=%s"
            ),
            (run.run_id,),
        )
        assert counts == {
            "execution_used_ms": 500,
            "model_reserved_tokens": 42,
            "model_steps": 2,
            "tool_dispatches": 1,
        }
    if not abort:
        view = await store.get_run(actor, run.run_id)
        await store.cancel_run(
            actor,
            run.run_id,
            CancelRequest(request_id="cleanup", expected_run_revision=view.revision),
        )
