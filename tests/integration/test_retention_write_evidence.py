"""实际独立写沙箱计数与审批/unknown清理保护定义；本批不运行。"""

import pytest

from harness.adapters.retention_postgres import PostgresRetention
from harness.contracts.retention import CleanupRequest
from harness.contracts.runtime import RuntimeFault
from tests.integration.test_artifact_write_failures import artifact_write_case as artifact_write_case, count

pytestmark = pytest.mark.integration


async def test_session_delete_cannot_remove_write_receipts_guards_or_replay_unknown(artifact_write_case):
    store, _, writes, handler, actor, owner, run, approve = artifact_write_case
    success = await approve("success")
    receipt = await writes.execute(run, owner, "create_work_order", success.logical_call_id,
        success.arguments.as_object(), success.plan_hash)
    pending = await approve("unknown")
    handler.fault = "after_write"
    with pytest.raises(RuntimeFault, match="SANDBOX_WRITE_UNCONFIRMED"):
        await writes.execute(run, owner, "create_work_order", pending.logical_call_id,
            pending.arguments.as_object(), pending.plan_hash)
    assert await count(handler, actor) == 2
    operator = actor.model_copy(update={"roles": actor.roles | {"retention_operator"},
        "resource_scopes": actor.resource_scopes | {f"session:{run.session_id}:retention"}})
    manager = PostgresRetention(store)
    job = await manager.request(actor, run.session_id, CleanupRequest(request_id="delete-session", kind="session",
        reason="delete", expected_revision=0))
    blocked = await manager.advance(operator, job.job_id)
    assert blocked.state == "blocked" and {"unresolved", "write_audit_retained"} <= set(blocked.reasons)
    assert blocked.attempts == 0 and blocked.revoked_at is not None
    with pytest.raises(RuntimeFault, match="SESSION_REVOKED"):
        await store.get_run(actor, run.run_id)
    async with store._transaction() as conn:
        op = await store._one(conn, "SELECT state,attempts FROM harness.operations WHERE run_id=%s AND logical_call_id=%s",
                             (run.run_id, pending.logical_call_id))
        assert op == {"state": "unknown", "attempts": 1}
        saved = await store._one(conn, "SELECT receipt FROM harness.operations WHERE operation_id=%s", (receipt.operation_id,))
        assert saved["receipt"]["external_reference"]
        guards = await store._one(conn, """SELECT count(*) AS n FROM harness.business_action_guards g
            JOIN harness.operations o USING(operation_id) WHERE o.run_id=%s""", (run.run_id,))
        assert guards["n"] == 2
        approvals = await store._one(conn, "SELECT count(*) AS n FROM harness.approvals a JOIN harness.operations o USING(operation_id) WHERE o.run_id=%s", (run.run_id,))
        assert approvals["n"] == 2
    assert await count(handler, actor) == 2
