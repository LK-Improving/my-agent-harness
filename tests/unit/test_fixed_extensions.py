"""M2R-04固定链反例定义；内存账本/处理器仅证明协议，不替代真实PG与审批。"""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harness.contracts.extensions import ExtensionDecision, ExtensionSpec
from harness.contracts.hashing import RunRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec
from harness.contracts.state import ToolReceipt
from harness.contracts.writes import DispatchPermit, ExternalWriteResult, WriteOperation
from harness.runtime.extensions import ExtensionChain, RegisteredExtension
from harness.tools.writes import WriteCoordinator, WriteRegistry
from pydantic import SecretStr, ValidationError

from domain_tools.operations.task_policy import require_task_target
from domain_tools.operations.work_orders import WorkOrderSandboxHandler, registered_work_order


class MemoryRecorder:
    def __init__(self):
        self.decisions = {}

    async def __call__(self, run, spec, invocation, decision):
        key = (run.run_id, spec.extension_id, invocation.fingerprint())
        value = decision.model_dump_json()
        if key in self.decisions and self.decisions[key] != value:
            raise RuntimeFault("EXTENSION_REPLAY_DECISION_CHANGED")
        self.decisions[key] = value


def registration(name, stage, handler, *, order=0, mode="policy", timeout=500):
    return RegisteredExtension(
        ExtensionSpec(
            extension_id=name,
            version="1",
            handler_ref=name + "@1",
            stage=stage,
            order=order,
            mode=mode,
            timeout_ms=timeout,
        ),
        handler,
    )


def decision(invocation, name, **kwargs):
    return ExtensionDecision(
        extension_id=name, version="1", invocation_fingerprint=invocation.fingerprint(), **kwargs
    )


def run_for(chain):
    root = Path(__file__).resolve().parents[2]
    spec = AgentSpec.model_validate_json((root / "agents/operations.json").read_text("utf-8"))
    manifest = ReleaseManifest(
        spec=spec,
        source_revision=None,
        source_tree_hash="a" * 64,
        dependency_lock_hash="b" * 64,
        prompt_hash="c" * 64,
        tool_schema_hashes={f"{item.name}@{item.version}": "f" * 64 for item in spec.allowed_tools},
        model_profile={},
        profiles={"extensions": chain.snapshot.as_object()},
        environment_profile_hash="e" * 64,
    )
    return ClaimedRun(
        run_id="run",
        session_id="session",
        manifest=manifest,
        principal=Principal(
            tenant_id="tenant",
            subject_id="user",
            application_id="app",
            resource_scopes=frozenset({"work_orders:write:synthetic"}),
        ),
        request=RunRequest(request_id="r", expected_revision=0, message="为synthetic-demo创建工单"),
        execution_deadline=datetime.now(UTC) + timedelta(minutes=1),
        history=JsonSnapshot.capture([]),
    )


async def test_fixed_order_manifest_and_protected_context():
    visited = []

    async def first(inv):
        visited.append("first")
        return decision(
            inv,
            "first",
            context_prefix=JsonSnapshot.capture([{"role": "user", "content": "辅助上下文"}]),
        )

    async def second(inv):
        visited.append("second")
        return decision(inv, "second")

    chain = ExtensionChain(
        (
            registration("second", "before_model", second, order=2),
            registration("first", "before_model", first, order=1),
        ),
        record_decision=MemoryRecorder(),
    )
    run = run_for(chain)
    payload = JsonSnapshot.capture(
        {
            "messages": [
                {"role": "system", "content": "固定规则"},
                {"role": "user", "content": "原始目标"},
            ],
            "tools": [],
        }
    )
    result = await chain.run("before_model", run, payload)
    assert visited == ["first", "second"]
    assert result.as_object()["messages"] == [
        {"role": "system", "content": "固定规则"},
        {"role": "user", "content": "辅助上下文"},
        {"role": "user", "content": "原始目标"},
    ]
    assert payload.as_object()["messages"][-1]["content"] == "原始目标"
    with pytest.raises(FrozenInstanceError):
        chain.registrations = ()
    missing = run.model_copy(
        update={"manifest": run.manifest.model_copy(update={"profiles": JsonSnapshot.capture({})})}
    )
    with pytest.raises(RuntimeFault, match="EXTENSION_MANIFEST_MISMATCH"):
        await chain.run("before_model", missing, payload)
    with pytest.raises(ValueError):
        ExtensionChain(
            (
                registration("first", "before_model", first),
                registration("second", "before_model", second),
            )
        )


async def test_denial_is_monotonic_and_missing_audit_stops_before_callback():
    visited = []

    async def deny(inv):
        visited.append("deny")
        return decision(inv, "deny", verdict="deny", reason_code="policy_rejected")

    async def later(inv):
        visited.append("later")
        return decision(inv, "later")

    regs = (
        registration("deny", "before_run", deny),
        registration("later", "before_run", later, order=1),
    )
    chain = ExtensionChain(regs, record_decision=MemoryRecorder())
    with pytest.raises(RuntimeFault):
        await chain.run("before_run", run_for(chain), JsonSnapshot.capture({}))
    assert visited == ["deny"]
    visited.clear()
    no_audit = ExtensionChain(regs)
    with pytest.raises(RuntimeFault):
        await no_audit.run("before_run", run_for(no_audit), JsonSnapshot.capture({}))
    assert visited == []


async def test_observer_failure_isolated_but_cancellation_propagates():
    async def failed(inv):
        raise ValueError("private body must not escape")

    chain = ExtensionChain((registration("observe", "after_tool", failed, mode="observer"),))
    payload = JsonSnapshot.capture({"receipt": {"status": "succeeded"}})
    assert await chain.run("after_tool", run_for(chain), payload) == payload

    async def cancelled(inv):
        raise asyncio.CancelledError()

    chain = ExtensionChain((registration("observe", "after_tool", cancelled, mode="observer"),))
    with pytest.raises(asyncio.CancelledError):
        await chain.run("after_tool", run_for(chain), payload)

    async def slow(inv):
        await asyncio.sleep(1)
        return decision(inv, "slow")

    chain = ExtensionChain(
        (registration("slow", "before_run", slow, timeout=1),), record_decision=MemoryRecorder()
    )
    with pytest.raises(RuntimeFault):
        await chain.run("before_run", run_for(chain), JsonSnapshot.capture({}))


async def test_replay_of_changed_critical_decision_is_blocked():
    counter = 0

    async def patch(inv):
        nonlocal counter
        counter += 1
        return decision(
            inv, "patch", tool_arguments=JsonSnapshot.capture({"order_id": f"synthetic-{counter}"})
        )

    chain = ExtensionChain(
        (registration("patch", "before_tool", patch),), record_decision=MemoryRecorder()
    )
    run = run_for(chain)
    source = JsonSnapshot.capture({"arguments": {"order_id": "synthetic-demo"}})
    await chain.run("before_tool", run, source)
    with pytest.raises(RuntimeFault):
        await chain.run("before_tool", run, source)
    with pytest.raises(ValidationError):
        ExtensionDecision(
            extension_id="patch",
            version="1",
            invocation_fingerprint="a" * 64,
            verdict="abstain",
            receipt={"status": "not_executed"},
        )


class ControlledHandler(WorkOrderSandboxHandler):
    def __init__(self):
        super().__init__("http://127.0.0.1:18081", SecretStr("a" * 40))
        self.calls = 0

    async def invoke(self, permit):
        self.calls += 1
        return ExternalWriteResult(
            external_reference="wo_" + "a" * 32,
            data={
                "order_id": permit.operation.arguments.as_object()["order_id"],
                "work_order_id": "wo_" + "a" * 32,
            },
        )


class ControlledWriteLedger:
    def __init__(self):
        self.operation = None
        self.unknown = 0
        self.recorded = []

    async def prepare_write(self, actor, owner, proposal):
        if self.operation is not None:
            if self.operation.arguments != proposal.arguments:
                raise RuntimeFault("LOGICAL_CALL_CONFLICT")
            return self.operation
        self.operation = WriteOperation(
            operation_id="op",
            run_id=proposal.run_id,
            logical_call_id=proposal.logical_call_id,
            principal=actor,
            tool=proposal.tool,
            arguments=proposal.arguments,
            action_key=proposal.action_key,
            plan_hash=proposal.plan_hash,
            binding_hash="b" * 64,
            approval_id="approval",
            state="prepared",
        )
        return self.operation

    async def begin_dispatch(self, actor, owner, operation_id):
        # 仅控制端口模拟已批准，不能据此证明真实审批流程。
        return DispatchPermit(
            operation=self.operation,
            attempt_id="attempt",
            deadline_at=datetime.now(UTC) + timedelta(seconds=20),
        )

    async def record_write_result(self, permit, result, error):
        receipt = ToolReceipt(
            operation_id="op",
            attempt_id="attempt",
            status="succeeded",
            args_hash=permit.operation.arguments.content_hash,
            data=result.data,
            external_reference=result.external_reference,
        )
        self.recorded.append(receipt)
        self.operation = self.operation.model_copy(
            update={"state": "succeeded", "receipt": receipt, "output_error": error}
        )
        return receipt

    async def mark_write_unknown(self, permit):
        self.unknown += 1


async def test_tool_patch_revalidates_domain_target_before_any_dispatch():
    async def change_target(inv):
        return decision(
            inv, "change", tool_arguments=JsonSnapshot.capture({"order_id": "synthetic-other"})
        )

    chain = ExtensionChain(
        (registration("change", "before_tool", change_target),), record_decision=MemoryRecorder()
    )
    handler, ledger = ControlledHandler(), ControlledWriteLedger()
    service = WriteCoordinator(
        ledger,
        WriteRegistry((registered_work_order(handler),)),
        extensions=chain,
        task_guard=require_task_target,
    )
    with pytest.raises(RuntimeFault):
        await service.execute(
            run_for(chain),
            "owner",
            "create_work_order",
            "logical",
            {"order_id": "synthetic-demo"},
            "a" * 64,
        )
    assert handler.calls == 0 and ledger.operation is None


async def test_prepared_proposal_does_not_transform_twice_and_post_failure_keeps_receipt():
    transformed = []

    async def before(inv):
        transformed.append(True)
        return decision(
            inv, "before", tool_arguments=JsonSnapshot.capture({"order_id": "synthetic-demo"})
        )

    async def bad_after(inv):
        return decision(
            inv,
            "after",
            context_prefix=JsonSnapshot.capture([{"role": "user", "content": "erase write"}]),
        )

    chain = ExtensionChain(
        (
            registration("before", "before_tool", before),
            registration("after", "after_tool", bad_after),
        ),
        record_decision=MemoryRecorder(),
    )
    handler, ledger = ControlledHandler(), ControlledWriteLedger()
    service = WriteCoordinator(
        ledger, WriteRegistry((registered_work_order(handler),)), extensions=chain
    )
    run = run_for(chain)
    proposal = await service.prepare_proposal(
        run, "create_work_order", "logical", {"order_id": "synthetic-demo"}, "a" * 64
    )
    for _ in range(2):
        with pytest.raises(RuntimeFault):
            await service.execute(
                run,
                "owner",
                "create_work_order",
                "logical",
                proposal.arguments.as_object(),
                proposal.plan_hash,
                prepared_proposal=proposal,
            )
    assert transformed == [True] and handler.calls == 1
    assert len(ledger.recorded) == 1 and ledger.unknown == 0
    assert ledger.operation.state == "succeeded"


async def test_after_model_failure_keeps_result_and_never_reissues_transport():
    from harness.contracts.execution import ExecutionProfile, ModelReservation
    from harness.contracts.models import ModelResult, ModelUsage
    from harness.contracts.release import BudgetProfile
    from harness.runtime.models import GovernedModelGateway

    from tests.unit.test_m0_boundaries import prepared

    async def deny_after(inv):
        return decision(inv, "after", verdict="deny", reason_code="reject_candidate")

    chain = ExtensionChain(
        (registration("after", "after_model", deny_after),), record_decision=MemoryRecorder()
    )
    request = prepared()
    profile = ExecutionProfile(
        route=request.route,
        budget=BudgetProfile(
            version="budget@1",
            max_input_tokens=1000,
            max_output_tokens=128,
            max_total_tokens=3000,
            execution_budget_ms=120000,
        ),
    )

    class Ledger:
        result = None
        failed = []

        async def reserve_model(self, *args):
            return ModelReservation(request=request, result=self.result)

        async def save_model_result(self, req, result):
            self.result = result

        async def fail_model(self, req, failure):
            self.failed.append(failure)

    class Transport:
        calls = 0

        async def complete(self, req):
            self.calls += 1
            return ModelResult(
                model="controlled",
                content="candidate",
                finish_reason="stop",
                request_id=None,
                usage=ModelUsage(input_tokens=0, output_tokens=0),
                model_call_id=req.model_call_id,
                model_attempt_id=req.model_attempt_id,
                request_fingerprint=req.fingerprint(),
                draft_id=req.draft_id,
            )

    ledger, transport = Ledger(), Transport()
    gateway = GovernedModelGateway(ledger, transport, profile, chain)
    for _ in range(2):
        with pytest.raises(RuntimeFault):
            await gateway.complete(run_for(chain), "owner", request.messages, request.tools)
    assert transport.calls == 1 and ledger.result is not None and ledger.failed == []
