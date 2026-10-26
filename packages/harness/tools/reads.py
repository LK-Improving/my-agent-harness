"""只读工具统一包装；结果过大时明确停止，不制造artifact引用。"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from harness.contracts.execution import ExecutionLedger
from harness.contracts.guards import GuardedToolCall, ToolGuard, compose_guard_decisions
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ExecutionContext, ToolSpec
from harness.contracts.state import ToolReceipt
from harness.runtime.extensions import ExtensionChain
from harness.tools.preparation import TaskGuard, prepare_arguments


@dataclass(frozen=True)
class RegisteredRead:
    spec: JsonSnapshot
    input_type: type[BaseModel]
    output_type: type[BaseModel]
    handler: Callable[[Principal, dict[str, Any]], Awaitable[Any]]
    authorize: Callable[[Principal, dict[str, Any]], bool]


class ReadCoordinator:
    def __init__(
        self,
        ledger: ExecutionLedger,
        entries: tuple[RegisteredRead, ...],
        guards: tuple[ToolGuard, ...] = (),
        extensions: ExtensionChain | None = None,
        task_guard: TaskGuard | None = None,
    ) -> None:
        self.ledger, self.guards = ledger, guards
        self.extensions, self.task_guard = extensions or ExtensionChain(), task_guard
        self.entries = MappingProxyType(
            {ToolSpec.model_validate(item.spec.as_object()).name: item for item in entries}
        )
        if len(self.entries) != len(entries):
            raise ValueError("只读工具不得重名")
        for item in entries:
            spec = ToolSpec.model_validate(item.spec.as_object())
            if (
                spec.effect not in {"read", "pure"}
                or spec.cost.billable
                or spec.retry.max_attempts_per_operation != 1
            ):
                raise ValueError("首版只读包装仅支持免费单次尝试")
            if canonical_hash(spec.input_schema) != canonical_hash(
                item.input_type.model_json_schema()
            ) or canonical_hash(spec.output_schema) != canonical_hash(
                item.output_type.model_json_schema()
            ):
                raise ValueError("只读Schema必须与实际校验器一致")

    async def execute(
        self,
        run: ClaimedRun,
        owner: str,
        name: str,
        logical_id: str,
        arguments: dict[str, Any],
        *,
        task_guard: TaskGuard | None = None,
    ) -> ToolReceipt:
        if name not in self.entries:
            raise RuntimeFault("TOOL_NOT_REGISTERED", 422)
        entry = self.entries[name]
        spec = ToolSpec.model_validate(entry.spec.as_object())
        args = (
            await prepare_arguments(
                run,
                spec,
                entry.input_type,
                entry.authorize,
                arguments,
                logical_id,
                self.extensions,
                self.task_guard or task_guard,
            )
        ).as_object()
        context = ExecutionContext(
            **run.principal.model_dump(mode="python"),
            run_id=run.run_id,
            release_id=run.manifest.spec.release_id,
            trace_id=run.run_id,
        )
        guarded = GuardedToolCall(
            context=context,
            operation_id="read-" + canonical_hash({"run": run.run_id, "logical": logical_id}),
            logical_call_id=logical_id,
            tool_name=name,
            tool_version=spec.version,
            arguments=JsonSnapshot.capture(args),
        )
        if compose_guard_decisions(
            guarded, tuple(item.evaluate(guarded) for item in self.guards)
        ).denied:
            raise RuntimeFault("TOOL_GUARD_DENIED", 403)
        await self.extensions.run(
            "wrap_tool_call",
            run,
            JsonSnapshot.capture(
                {
                    "tool_name": name,
                    "logical_call_id": logical_id,
                    "arguments": args,
                }
            ),
        )
        reservation = await self.ledger.reserve_read(
            run, owner, logical_id, entry.spec, guarded.arguments
        )
        if reservation.receipt:
            await self.after_receipt(run, reservation.receipt)
            return reservation.receipt
        try:
            async with asyncio.timeout(spec.timeout_ms / 1000):
                result = JsonSnapshot.capture(await entry.handler(run.principal, args))
            if len(result.canonical_text.encode("utf-8")) > 1048576:
                raise RuntimeFault("TOOL_RESULT_TOO_LARGE", 422)
            entry.output_type.model_validate(result.materialize(), strict=True)
        except BaseException:
            await self.ledger.save_read(reservation, None, "READ_FAILED")
            raise
        receipt = await self.ledger.save_read(reservation, result, None)
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
