"""写工具唯一派发入口；不拥有领域业务逻辑或图的恢复权。"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.contracts.guards import GuardedToolCall, ToolGuard, compose_guard_decisions
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ExecutionContext, ToolSpec
from harness.contracts.state import ReconciliationConclusion, ToolReceipt
from harness.contracts.writes import (
    ExecutionSuspended,
    ExternalWriteResult,
    ReconcileRequest,
    ReconciliationResult,
    WriteHandler,
    WriteLedger,
    WriteOperation,
    WritePolicy,
    WriteProposal,
)
from harness.runtime.extensions import ExtensionChain
from harness.tools.preparation import TaskGuard, prepare_arguments


@dataclass(frozen=True)
class RegisteredWrite:
    spec: JsonSnapshot
    input_type: type[BaseModel]
    output_type: type[BaseModel]
    handler: WriteHandler
    authorize: Callable[[Principal, dict[str, Any]], bool]
    action_identity: Callable[[dict[str, Any]], str]


class WriteRegistry:
    def __init__(self, entries: tuple[RegisteredWrite, ...]) -> None:
        entries_by_name: dict[str, RegisteredWrite] = {}
        for entry in entries:
            tool = ToolSpec.model_validate(entry.spec.as_object())
            if tool.effect != "write" or tool.name in entries_by_name:
                raise ValueError("只接受无重名的写工具")
            if not tool.idempotency or not tool.idempotency.provider_supported:
                raise ValueError("首批写适配器必须具备供应商幂等和核对")
            if tool.cost.billable:
                raise ValueError("未接入持久金额预算前不注册计费写工具")
            if canonical_hash(entry.input_type.model_json_schema()) != canonical_hash(
                tool.input_schema
            ):
                raise ValueError("输入校验器与注册Schema不一致")
            if canonical_hash(entry.output_type.model_json_schema()) != canonical_hash(
                tool.output_schema
            ):
                raise ValueError("输出校验器与注册Schema不一致")
            entries_by_name[tool.name] = entry
        self._entries = MappingProxyType(entries_by_name)

    def get(self, name: str) -> RegisteredWrite:
        try:
            return self._entries[name]
        except KeyError:
            raise RuntimeFault("TOOL_NOT_REGISTERED", 422) from None


class WriteCoordinator:
    def __init__(
        self,
        ledger: WriteLedger,
        registry: WriteRegistry,
        guards: tuple[ToolGuard, ...] = (),
        policy: WritePolicy | None = None,
        extensions: ExtensionChain | None = None,
        task_guard: TaskGuard | None = None,
    ) -> None:
        self.ledger = ledger
        self.registry = registry
        self.guards = guards
        self.policy = policy or WritePolicy()
        self.extensions, self.task_guard = extensions or ExtensionChain(), task_guard

    async def prepare_proposal(
        self,
        run: ClaimedRun,
        name: str,
        logical_call_id: str,
        arguments: dict[str, Any],
        plan_hash: str,
        *,
        task_guard: TaskGuard | None = None,
    ) -> WriteProposal:
        entry = self.registry.get(name)
        spec = ToolSpec.model_validate(entry.spec.as_object())
        arguments = (
            await prepare_arguments(
                run,
                spec,
                entry.input_type,
                entry.authorize,
                arguments,
                logical_call_id,
                self.extensions,
                self.task_guard or task_guard,
            )
        ).as_object()
        return self.proposal(run, name, logical_call_id, arguments, plan_hash)

    def proposal(
        self,
        run: ClaimedRun,
        name: str,
        logical_call_id: str,
        arguments: dict[str, Any],
        plan_hash: str,
    ) -> WriteProposal:
        entry = self.registry.get(name)
        tool = ToolSpec.model_validate(entry.spec.as_object())
        bindings = {binding.name: binding.version for binding in run.manifest.spec.allowed_tools}
        if bindings.get(name) != tool.version:
            raise RuntimeFault("TOOL_NOT_ALLOWED", 403)
        try:
            validated = entry.input_type.model_validate(arguments, strict=True).model_dump(
                mode="json"
            )
        except ValidationError:
            raise RuntimeFault("TOOL_INPUT_INVALID", 422) from None
        if not entry.authorize(run.principal, validated):
            raise RuntimeFault("TOOL_FORBIDDEN", 403)
        action = entry.action_identity(validated)
        if not action or len(action) > 512:
            raise RuntimeFault("BUSINESS_ACTION_ID_INVALID", 422)
        proposal = WriteProposal(
            run_id=run.run_id,
            logical_call_id=logical_call_id,
            tool=entry.spec,
            arguments=JsonSnapshot.capture(validated),
            action_key=canonical_hash(
                {"resolver": tool.business_action_key_resolver_ref, "identity": action}
            ),
            plan_hash=plan_hash,
        )
        return proposal

    async def execute(
        self,
        run: ClaimedRun,
        owner: str,
        name: str,
        logical_call_id: str,
        arguments: dict[str, Any],
        plan_hash: str,
        *,
        prepared_proposal: WriteProposal | None = None,
        task_guard: TaskGuard | None = None,
    ) -> ToolReceipt:
        if prepared_proposal is None:
            proposal = await self.prepare_proposal(
                run, name, logical_call_id, arguments, plan_hash, task_guard=task_guard
            )
        else:
            # 可信图已在interrupt之前完成改参；恢复后不能第二次变换参数。
            proposal = self.proposal(run, name, logical_call_id, arguments, plan_hash)
            if proposal != prepared_proposal:
                raise RuntimeFault("TOOL_PREPARED_PROPOSAL_MISMATCH")
            effective_guard = self.task_guard or task_guard
            if effective_guard is not None:
                effective_guard(
                    run,
                    ToolSpec.model_validate(proposal.tool.as_object()),
                    proposal.arguments.as_object(),
                )
        self.extensions.require_bound(run)
        entry = self.registry.get(name)
        tool = ToolSpec.model_validate(entry.spec.as_object())
        validated = proposal.arguments.as_object()
        operation = await self.ledger.prepare_write(run.principal, owner, proposal)
        context = ExecutionContext(
            **run.principal.model_dump(mode="python"),
            run_id=run.run_id,
            release_id=run.manifest.spec.release_id,
            trace_id=run.run_id,
        )
        guarded = GuardedToolCall(
            context=context,
            operation_id=operation.operation_id,
            logical_call_id=logical_call_id,
            tool_name=name,
            tool_version=tool.version,
            arguments=proposal.arguments,
        )
        # 固定顺序、原快照；guard无权改参或扩大批准/范围/预算。
        summary = compose_guard_decisions(
            guarded, tuple(guard.evaluate(guarded) for guard in self.guards)
        )
        if summary.denied:
            raise RuntimeFault("TOOL_GUARD_DENIED", 403)
        if operation.state == "succeeded":
            if operation.output_error:
                raise RuntimeFault(operation.output_error, 422)
            if operation.receipt is None:
                raise RuntimeFault("WRITE_RECEIPT_MISSING")
            await self.after_receipt(run, operation.receipt)
            return operation.receipt
        await self.extensions.run(
            "wrap_tool_call",
            run,
            JsonSnapshot.capture(
                {
                    "operation_id": operation.operation_id,
                    "tool_name": name,
                    "logical_call_id": logical_call_id,
                    "arguments": validated,
                    "binding_hash": operation.binding_hash,
                }
            ),
        )
        # 任何潜在派发之前再次校验服务端授权；随后账本原子检查批准、取消和额度。
        if not entry.authorize(run.principal, validated):
            raise RuntimeFault("TOOL_FORBIDDEN", 403)
        try:
            permit = await self.ledger.begin_dispatch(run.principal, owner, operation.operation_id)
        except RuntimeFault as exc:
            if exc.code in {"APPROVAL_PENDING", "WRITE_RECONCILIATION_REQUIRED"}:
                raise ExecutionSuspended(operation.operation_id) from None
            raise
        try:
            timeout = min(
                tool.timeout_ms / 1000, (permit.deadline_at - datetime.now(UTC)).total_seconds()
            )
            if timeout <= 0:
                raise RuntimeFault("WRITE_DISPATCH_DEADLINE_EXPIRED")
            async with asyncio.timeout(timeout):
                result = await entry.handler.invoke(permit)
        except BaseException:
            # CancelledError也先记unknown再继续抛出，不能吞掉取消或盲目重试。
            await self.ledger.mark_write_unknown(permit)
            raise
        safe_result, error = self.validate_output(entry, result)
        receipt = await self.ledger.record_write_result(permit, safe_result, error)
        if error:
            raise RuntimeFault(error, 422)
        await self.after_receipt(run, receipt)
        return receipt

    async def after_receipt(self, run: ClaimedRun, receipt: ToolReceipt) -> None:
        await self.extensions.run(
            "after_tool",
            run,
            JsonSnapshot.capture(
                {
                    "receipt": receipt.model_dump(mode="json"),
                }
            ),
        )

    def validate_output(
        self, entry: RegisteredWrite, result: ExternalWriteResult
    ) -> tuple[ExternalWriteResult, str | None]:
        snapshot = JsonSnapshot.capture(result.data)
        result = ExternalWriteResult(
            external_reference=result.external_reference, data=snapshot.materialize()
        )
        if len(snapshot.canonical_text.encode("utf-8")) > self.policy.max_result_bytes:
            return result.model_copy(update={"data": None}), "TOOL_RESULT_TOO_LARGE"
        try:
            entry.output_type.model_validate(result.data, strict=True)
        except ValidationError:
            return result, "TOOL_OUTPUT_INVALID"
        return result, None

    async def reconcile(
        self, actor: Principal, run_id: str, request: ReconcileRequest
    ) -> WriteOperation:
        permit = await self.ledger.begin_reconciliation(actor, run_id, request)
        if permit.recorded_result is not None:
            return permit.operation
        tool = ToolSpec.model_validate(permit.operation.tool.as_object())
        entry = self.registry.get(tool.name)
        if entry.spec.content_hash != permit.operation.tool.content_hash:
            raise RuntimeFault("RECONCILE_ADAPTER_VERSION_MISMATCH")
        try:
            async with asyncio.timeout(self.policy.reconciliation_timeout_seconds):
                result = await entry.handler.reconcile(permit)
        except Exception:
            result = ReconciliationResult(
                operation_id=permit.operation.operation_id,
                args_hash=permit.operation.arguments.content_hash,
                conclusion=ReconciliationConclusion.UNKNOWN,
                evidence_ref="reconcile-transport-unconfirmed",
            )
        error = None
        if result.result is not None:
            safe, error = self.validate_output(entry, result.result)
            result = result.model_copy(update={"result": safe, "output_error": error})
        operation = await self.ledger.finish_reconciliation(actor, permit, result)
        # Schema错误必须与副作用事实一起落账，不能在核对后被当作新派发机会。
        if error:
            raise RuntimeFault("RECONCILED_OUTPUT_REQUIRES_REVIEW", 422)
        return operation
