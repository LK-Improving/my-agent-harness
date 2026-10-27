"""独立真实写沙箱计数+PG故障定义；不执行供应商，不冒充真实进程验收。"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from psycopg.errors import DivisionByZero

from domain_tools.operations.work_orders import WorkOrderSandboxHandler, registered_work_order
from harness.adapters.artifact_retention_postgres import PostgresArtifactRetention
from harness.adapters.artifacts_postgres import PostgresArtifacts
from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.bootstrap.sandbox import create_sandbox, initialize, load_sandbox_settings
from harness.contracts.artifacts import ArtifactDeletionRequest, ArtifactProfile, ArtifactRead, ArtifactRetentionPolicy
from harness.contracts.hashing import RunRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AdmissionProfile, EvaluationRequest, RuntimeFault, SessionRequest
from harness.contracts.specs import ToolBinding
from harness.contracts.writes import ApprovalDecisionRequest
from harness.tools.artifacts import registered_artifact_reader
from harness.tools.writes import WriteCoordinator, WriteRegistry

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def artifact_write_case(prepared):
    _, original_actor, original, _ = prepared
    actor = original_actor.model_copy(update={"resource_scopes": frozenset({"work_orders:write:synthetic"})})
    store = PostgresExecutionStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    settings = load_sandbox_settings(ROOT)
    await initialize(ROOT)
    app = create_sandbox(settings)

    class LocalHandler(WorkOrderSandboxHandler):
        def _client(self):
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sandbox",
                headers={"Authorization": "Bearer " + settings.token.get_secret_value()})

    handler = LocalHandler(settings.base_url, settings.token)
    entry = registered_work_order(handler)
    artifacts = PostgresArtifacts(store, ArtifactProfile(projection_chars=8))
    reader = registered_artifact_reader(artifacts)
    spec = original.spec.model_copy(update={"release_id": "artifact-write-test", "allowed_tools": (
        ToolBinding(name="create_work_order", version="1", max_operations_per_run=2),
        ToolBinding(name="read_artifact", version="1", max_operations_per_run=4))})
    manifest = ReleaseManifest.model_validate({**original.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
        "tool_schema_hashes": {"create_work_order@1": entry.spec.content_hash, "read_artifact@1": reader.spec.content_hash},
        "profiles": {"artifacts": artifacts.profile.model_dump(mode="json")}})
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(actor, EvaluationRequest(candidate_release_id=spec.release_id,
        dataset_hash="e" * 64, gate_policy_version="artifact-write@1", resource_scopes=actor.resource_scopes,
        expires_at=datetime.now(UTC) + timedelta(hours=1)))
    session = await store.create_session(actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id))
    await store.create_run(actor, session, RunRequest(request_id="write", message="创建合成工单", expected_revision=0))
    owner = uuid4().hex
    await store.register_worker(owner)
    run = await store.claim(owner)
    assert run is not None and run.session_id == session
    writes = WriteCoordinator(store, WriteRegistry((entry,)))
    approver = actor.model_copy(update={"subject_id": "independent-approver", "roles": frozenset({"approver", "reconciler"})})

    async def approve(logical):
        proposal = writes.proposal(run, "create_work_order", logical, {"order_id": "synthetic-" + uuid4().hex}, "f" * 64)
        operation = await store.prepare_write(actor, owner, proposal)
        await store.decide_approval(approver, operation.approval_id, ApprovalDecisionRequest(
            request_id="approve-" + logical, binding_hash=operation.binding_hash, expected_revision=0, decision="approve"))
        return proposal

    try:
        yield store, artifacts, writes, handler, actor, owner, run, approve
    finally:
        await store.fail(owner, run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)


async def count(handler, actor):
    async with handler._client() as client:
        response = await client.get("/counts", params={"tenant_id": actor.tenant_id, "application_id": actor.application_id})
    assert response.status_code == 200
    return response.json()["writes"]


@pytest.mark.parametrize("fault", ["insert", "audit", "event", "cancelled", "commit_ack_lost"])
async def test_successful_write_is_not_repeated_after_artifact_failure(artifact_write_case, monkeypatch, fault):
    store, artifacts, writes, handler, actor, owner, run, approve = artifact_write_case
    proposal = await approve("write-once")
    args = (run, owner, "create_work_order", proposal.logical_call_id, proposal.arguments.as_object(), proposal.plan_hash)
    receipt = await writes.execute(*args)
    assert await count(handler, actor) == 1
    expected = RuntimeFault
    with monkeypatch.context() as patch:
        if fault == "insert":
            original = store._one

            async def insert_failure(conn, query, args=()):
                if "INSERT INTO harness.artifacts" in query:
                    await conn.execute("SELECT 1/0")
                return await original(conn, query, args)

            patch.setattr(store, "_one", insert_failure)
            expected = DivisionByZero
        elif fault == "commit_ack_lost":
            original = store._transaction

            @asynccontextmanager
            async def lost_ack():
                async with original() as conn:
                    yield conn
                # PG已提交但调用方未得到引用；恢复只能查询同源记录。
                raise RuntimeFault("COMMIT_ACK_LOST", 503)

            patch.setattr(store, "_transaction", lost_ack)
        else:
            target = "_event" if fault == "event" else "_audit"
            original = getattr(store, target)

            async def unavailable(conn, actor_or_run, action, resource):
                if action == "artifact.saved":
                    if fault == "cancelled":
                        raise asyncio.CancelledError()
                    raise RuntimeFault("PERSISTENCE_UNAVAILABLE", 503)
                return await original(conn, actor_or_run, action, resource)

            patch.setattr(store, target, unavailable)
            if fault == "cancelled":
                expected = asyncio.CancelledError
        with pytest.raises(expected):
            await artifacts.project(run, owner, receipt)
    # 持久成功回执必须复用，不能再次调用沙箱；未知提交结果也不能新增产物。
    assert await writes.execute(*args) == receipt
    info = (await artifacts.project(run, owner, receipt)).as_object()["artifact"]
    assert (await artifacts.read(actor, run.session_id, ArtifactRead(artifact_ref=info["artifact_ref"]))).source_operation_id == receipt.operation_id
    assert await count(handler, actor) == 1
    async with store._transaction() as conn:
        assert (await store._one(conn, "SELECT count(*) AS n FROM harness.artifacts WHERE run_id=%s", (run.run_id,)))["n"] == 1
        row = await store._one(conn, "SELECT attempts,state,receipt FROM harness.operations WHERE operation_id=%s", (receipt.operation_id,))
        assert row["attempts"] == 1 and row["state"] == "succeeded" and row["receipt"]["external_reference"]
    events = await store.events(actor, run.run_id, 0)
    assert len([event for event in events if event["type"] == "artifact.saved"]) == 1


async def test_delete_request_preserves_unknown_write_receipt_and_action_guards(artifact_write_case):
    store, artifacts, writes, handler, actor, owner, run, approve = artifact_write_case
    proposal = await approve("success")
    receipt = await writes.execute(run, owner, "create_work_order", proposal.logical_call_id,
                                   proposal.arguments.as_object(), proposal.plan_hash)
    info = (await artifacts.project(run, owner, receipt)).as_object()["artifact"]
    pending = await approve("unknown")
    handler.fault = "after_write"
    with pytest.raises(RuntimeFault, match="SANDBOX_WRITE_UNCONFIRMED"):
        await writes.execute(run, owner, "create_work_order", pending.logical_call_id,
                              pending.arguments.as_object(), pending.plan_hash)
    retention = PostgresArtifactRetention(artifacts, ArtifactRetentionPolicy())
    result = await retention.request_deletion(actor, run.session_id, info["artifact_ref"],
        ArtifactDeletionRequest(request_id="delete", expected_content_hash=info["content_hash"]))
    assert result.state == "retained" and "unresolved_write" in result.retention_reasons and not result.readable
    assert await count(handler, actor) == 2
    async with store._transaction() as conn:
        op = await store._one(conn, "SELECT state,attempts FROM harness.operations WHERE run_id=%s AND logical_call_id=%s", (run.run_id, pending.logical_call_id))
        assert op == {"state": "unknown", "attempts": 1}
        guards = await store._one(conn, "SELECT count(*) AS n FROM harness.business_action_guards g JOIN harness.operations o USING(operation_id) WHERE o.run_id=%s", (run.run_id,))
        assert guards["n"] == 2
    with pytest.raises(RuntimeFault, match="ARTIFACT_REVOKED"):
        await artifacts.read(actor, run.session_id, ArtifactRead(artifact_ref=info["artifact_ref"]))
