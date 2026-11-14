"""撤销后的派发阻断与独立沙箱unknown核对定义；当前不运行服务或写入。"""

import pytest

from domain_tools.operations.work_orders import registered_work_order
from harness.contracts.evaluation_contexts import EvaluationRevocationRequest
from harness.contracts.runtime import RuntimeFault
from harness.contracts.writes import ApprovalDecisionRequest, ReconcileRequest
from harness.tools.writes import WriteCoordinator, WriteRegistry
from tests.integration.test_m2_writes import write_case as write_case

pytestmark = pytest.mark.integration


async def revoke(store, actor, run):
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT evaluation_context_id FROM harness.sessions WHERE session_id=%s", (run.session_id,))
    return await store.revoke_evaluation(actor, row["evaluation_context_id"], EvaluationRevocationRequest(
        request_id="revoke-write", expected_manifest_hash=run.manifest.content_hash()))


async def approve(store, approver, operation):
    await store.decide_approval(approver, operation.approval_id, ApprovalDecisionRequest(
        request_id="approve", binding_hash=operation.binding_hash, expected_revision=0, decision="approve"))


async def test_revoked_context_cannot_dispatch_previously_approved_write(write_case):
    store, actor, approver, owner, run, _, operation, handler = write_case
    await approve(store, approver, operation)
    await revoke(store, actor, run)
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await store.begin_dispatch(actor, owner, operation.operation_id)
    async with handler._client() as client:
        response = await client.get("/counts", params={"tenant_id": actor.tenant_id, "application_id": actor.application_id})
    assert response.json()["writes"] == 0
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT state,attempts FROM harness.operations WHERE operation_id=%s", (operation.operation_id,))
        assert row["attempts"] == 0 and row["state"] == "prepared"


async def test_revoked_unknown_keeps_guard_and_governance_can_reconcile_without_replay(write_case):
    store, actor, approver, owner, run, _, operation, handler = write_case
    await approve(store, approver, operation)
    permit = await store.begin_dispatch(actor, owner, operation.operation_id)
    with pytest.raises(RuntimeFault, match="SANDBOX_WRITE_UNCONFIRMED"):
        await handler.invoke(permit)
    await store.mark_write_unknown(permit)
    await store.fail(owner, run.run_id, "RESPONSE_LOST")
    await store.finish_slot(owner)
    await revoke(store, actor, run)
    coordinator = WriteCoordinator(store, WriteRegistry((registered_work_order(handler),)))
    reconciled = await coordinator.reconcile(approver, run.run_id,
        ReconcileRequest(request_id="reconcile-revoked", operation_id=operation.operation_id))
    assert reconciled.state == "succeeded"
    async with handler._client() as client:
        response = await client.get("/counts", params={"tenant_id": actor.tenant_id, "application_id": actor.application_id})
    assert response.json()["writes"] == 1
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await store.get_run(actor, run.run_id)
    async with store._transaction() as conn:
        guard = await store._one(conn, "SELECT operation_id FROM harness.business_action_guards WHERE operation_id=%s", (operation.operation_id,))
        assert guard["operation_id"] == operation.operation_id
        session = await store._one(conn, "SELECT active_run_id FROM harness.sessions WHERE session_id=%s", (run.session_id,))
        assert session["active_run_id"] == run.run_id
