"""M2 正式账本+独立沙箱库集成定义；HTTP使用ASGI，不等于真实Worker/API-SSE验收。"""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from harness.adapters.writes_postgres import PostgresWriteStore
from harness.bootstrap.sandbox import create_sandbox, initialize, load_sandbox_settings
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.evaluation_contexts import EvaluationAdmission
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import (
    AdmissionProfile,
    EvaluationRequest,
    RuntimeFault,
    SessionRequest,
)
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.writes import ApprovalDecisionRequest, ReconcileRequest, WriteProposal
from harness.tools.writes import WriteCoordinator, WriteRegistry

from domain_tools.operations.work_orders import WorkOrderSandboxHandler, registered_work_order

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture
async def write_case(prepared):
    _, base_actor, original, _ = prepared
    actor = base_actor.model_copy(
        update={"resource_scopes": frozenset({"work_orders:write:synthetic"})}
    )
    store = PostgresWriteStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    store.profile = store.profile.model_copy(update={"evaluation": EvaluationAdmission(
        test_resource_scopes=frozenset({"work_orders:write:synthetic"}))})
    settings = load_sandbox_settings(ROOT)
    await initialize(ROOT)
    app = create_sandbox(settings)

    class LocalHandler(WorkOrderSandboxHandler):
        def _client(self):
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://sandbox",
                headers={"Authorization": "Bearer " + settings.token.get_secret_value()},
            )

    handler = LocalHandler(settings.base_url, settings.token, fault="after_write")
    entry = registered_work_order(handler)
    spec = original.spec.model_copy(
        update={"release_id": "m2-write-test", "allowed_tools": (original.spec.allowed_tools[1],)}
    )
    assert spec.allowed_tools[0].name == "create_work_order"
    manifest = ReleaseManifest.model_validate(
        {
            **original.model_dump(mode="json"),
            "spec": spec.model_dump(mode="json"),
            "tool_schema_hashes": {"create_work_order@1": entry.spec.content_hash},
        }
    )
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(
        actor,
        EvaluationRequest(
            candidate_release_id=spec.release_id,
            dataset_hash="d" * 64,
            gate_policy_version="write-test@1",
            resource_scopes=actor.resource_scopes,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
    )
    session_id = await store.create_session(
        actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id)
    )
    await store.create_run(
        actor,
        session_id,
        RunRequest(request_id="m2-run", message="创建合成工单", expected_revision=0),
    )
    owner = uuid4().hex
    await store.register_worker(owner)
    run = await store.claim(owner)
    assert run is not None and run.session_id == session_id
    args = JsonSnapshot.capture({"order_id": "synthetic-" + uuid4().hex})
    proposal = WriteProposal(
        run_id=run.run_id,
        logical_call_id="logical-call-1",
        tool=entry.spec,
        arguments=args,
        action_key=canonical_hash(
            {
                "resolver": "operations.order-action@1",
                "identity": "create-work-order:" + args.as_object()["order_id"],
            }
        ),
        plan_hash="f" * 64,
    )
    operation = await store.prepare_write(actor, owner, proposal)
    approver = actor.model_copy(
        update={
            "subject_id": "independent-approver",
            "roles": frozenset({"approver", "reconciler"}),
        }
    )
    yield store, actor, approver, owner, run, proposal, operation, handler
    # 本用例没有后台引擎；测试退出时只收尾本owner，unknown会继续保留session/guard。
    await store.fail(owner, run.run_id, "TEST_FINISHED")
    await store.finish_slot(owner)
    await store.release_worker(owner)


async def test_exact_approval_and_replay_never_creates_another_operation(write_case):
    store, actor, approver, owner, run, proposal, operation, _ = write_case
    same = await store.prepare_write(actor, owner, proposal)
    assert same.operation_id == operation.operation_id
    changed = proposal.model_copy(
        update={"arguments": JsonSnapshot.capture({"order_id": "synthetic-other"})}
    )
    with pytest.raises(RuntimeFault, match="LOGICAL_CALL_CONFLICT"):
        await store.prepare_write(actor, owner, changed)
    request = ApprovalDecisionRequest(
        request_id="approve",
        binding_hash=operation.binding_hash,
        expected_revision=0,
        decision="approve",
    )
    with pytest.raises(RuntimeFault):
        await store.decide_approval(
            actor.model_copy(update={"roles": actor.roles | {"approver"}}),
            operation.approval_id,
            request,
        )
    decision = await store.decide_approval(approver, operation.approval_id, request)
    assert decision["state"] == "approved"
    assert (await store.decide_approval(approver, operation.approval_id, request))["revision"] == 1
    # 早到批准没有实际checkpoint绑定，不能把running转换为queued。
    assert (await store.get_run(actor, run.run_id)).status == "running"


async def test_response_loss_reconcile_receipt_and_external_count(write_case):
    store, actor, approver, owner, run, _, operation, handler = write_case
    await store.decide_approval(
        approver,
        operation.approval_id,
        ApprovalDecisionRequest(
            request_id="approve",
            binding_hash=operation.binding_hash,
            expected_revision=0,
            decision="approve",
        ),
    )
    permit = await store.begin_dispatch(actor, owner, operation.operation_id)
    with pytest.raises(RuntimeFault, match="SANDBOX_WRITE_UNCONFIRMED"):
        await handler.invoke(permit)
    await store.mark_write_unknown(permit)
    # 独立库HTTP计数：真实行已提交，Harness还没有成功回执。
    async with handler._client() as client:
        count = await client.get(
            "/counts", params={"tenant_id": actor.tenant_id, "application_id": actor.application_id}
        )
    assert count.json()["writes"] == 1
    await store.fail(owner, run.run_id, "RESPONSE_LOST")
    assert (await store.get_run(actor, run.run_id)).status == "reconciling"

    await store.finish_slot(owner)
    coordinator = WriteCoordinator(store, WriteRegistry((registered_work_order(handler),)))
    request = ReconcileRequest(request_id="reconcile-once", operation_id=operation.operation_id)
    reconciled = await coordinator.reconcile(approver, run.run_id, request)
    assert reconciled.state == "succeeded"
    assert (
        await coordinator.reconcile(approver, run.run_id, request)
    ).receipt == reconciled.receipt
    # 没有实际checkpoint证明，即使核对成功也不自动恢复执行。
    assert (await store.get_run(actor, run.run_id)).status == "reconciling"


async def test_approval_race_has_one_authoritative_decision(write_case):
    store, _, approver, _, _, _, operation, _ = write_case
    requests = [
        ApprovalDecisionRequest(
            request_id="decision-" + decision,
            binding_hash=operation.binding_hash,
            expected_revision=0,
            decision=decision,
        )
        for decision in ("approve", "reject")
    ]
    results = await asyncio.gather(
        *(store.decide_approval(approver, operation.approval_id, item) for item in requests),
        return_exceptions=True,
    )
    assert sum(isinstance(item, dict) for item in results) == 1
    assert sum(isinstance(item, RuntimeFault) for item in results) == 1


async def test_output_schema_failure_preserves_effect_and_never_replays_write(write_case):
    store, actor, approver, owner, run, proposal, operation, handler = write_case
    handler.fault = "invalid_output"
    await store.decide_approval(
        approver,
        operation.approval_id,
        ApprovalDecisionRequest(
            request_id="approve",
            binding_hash=operation.binding_hash,
            expected_revision=0,
            decision="approve",
        ),
    )
    coordinator = WriteCoordinator(store, WriteRegistry((registered_work_order(handler),)))
    for _ in range(2):
        with pytest.raises(RuntimeFault, match="TOOL_OUTPUT_INVALID"):
            await coordinator.execute(
                run,
                owner,
                "create_work_order",
                proposal.logical_call_id,
                proposal.arguments.as_object(),
                proposal.plan_hash,
            )
    persisted = await store.prepare_write(actor, owner, proposal)
    assert persisted.state == "succeeded" and persisted.output_error == "TOOL_OUTPUT_INVALID"
    assert persisted.receipt is not None and persisted.receipt.external_reference is not None
    async with handler._client() as client:
        count = await client.get(
            "/counts", params={"tenant_id": actor.tenant_id, "application_id": actor.application_id}
        )
    assert count.json()["writes"] == 1
