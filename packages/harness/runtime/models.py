"""统一模型包装；持久派发意图优先，不自动重试未知计费尝试。"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from harness.context.projection import project_model_messages
from harness.contracts.execution import ExecutionLedger, ExecutionProfile
from harness.contracts.failures import ProviderError
from harness.contracts.models import ModelGateway, ModelResult
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.runtime.extensions import ExtensionChain


class GovernedModelGateway:
    def __init__(
        self,
        ledger: ExecutionLedger,
        transport: ModelGateway,
        profile: ExecutionProfile,
        extensions: ExtensionChain | None = None,
        *,
        release_guard: Callable[[ReleaseManifest], None] | None = None,
    ) -> None:
        self.ledger, self.transport, self.profile = ledger, transport, profile
        self.extensions = extensions or ExtensionChain()
        self.release_guard = release_guard

    async def complete(
        self, run: ClaimedRun, owner: str, messages: JsonSnapshot, tools: JsonSnapshot
    ) -> ModelResult:
        if self.release_guard is not None:
            self.release_guard(run.manifest)
        messages = project_model_messages(messages.as_object_array())
        transformed = await self.extensions.run(
            "before_model",
            run,
            JsonSnapshot.capture(
                {
                    "messages": messages.materialize(),
                    "tools": tools.materialize(),
                    "route": self.profile.route.model_dump(mode="json"),
                }
            ),
        )
        messages = project_model_messages(
            JsonSnapshot.capture(transformed.as_object()["messages"]).as_object_array()
        )
        reservation = await self.ledger.reserve_model(run, owner, messages, tools, self.profile)
        if reservation.result is not None:
            result = reservation.result.model_copy(
                update={"draft_id": reservation.request.draft_id}
            )
            await self.after_result(run, result)
            return result
        request = reservation.request
        try:
            if self.release_guard is not None:
                self.release_guard(run.manifest)
            await self.extensions.run(
                "wrap_model_call",
                run,
                JsonSnapshot.capture(
                    {
                        "request_fingerprint": request.fingerprint(),
                        "model_call_id": request.model_call_id,
                        "model_attempt_id": request.model_attempt_id,
                    }
                ),
            )
        except BaseException:
            # 尚未调用传输；关键扩展/取消阻断不能伪造成未知计费请求。
            await self.ledger.fail_model(
                request,
                JsonSnapshot.capture(
                    {
                        "code": "MODEL_PRE_DISPATCH_BLOCKED",
                        "dispatch_status": "not_sent",
                        "billing_status": "not_billable",
                    }
                ),
            )
            raise
        try:
            request.require_dispatchable(datetime.now(UTC))
            async with asyncio.timeout(
                max(0.001, (run.execution_deadline - datetime.now(UTC)).total_seconds())
            ):
                result = await self.transport.complete(request)
        except ProviderError as exc:
            await self.ledger.fail_model(
                request, JsonSnapshot.capture(exc.failure.model_dump(mode="json"))
            )
            raise
        except BaseException:
            await self.ledger.fail_model(
                request,
                JsonSnapshot.capture(
                    {"code": "MODEL_OUTCOME_UNKNOWN", "billing_status": "unknown"}
                ),
            )
            raise
        if (
            result.request_fingerprint != request.fingerprint()
            or result.model_attempt_id != request.model_attempt_id
            or result.model_call_id != request.model_call_id
        ):
            await self.ledger.fail_model(
                request,
                JsonSnapshot.capture(
                    {"code": "MODEL_RESPONSE_BINDING_MISMATCH", "billing_status": "unknown"}
                ),
            )
            raise RuntimeFault("MODEL_RESPONSE_BINDING_MISMATCH")
        await self.ledger.save_model_result(request, result)
        if (result.usage.input_tokens or 0) > request.budget.input_tokens or (
            result.usage.output_tokens or 0
        ) > request.max_output_tokens:
            raise RuntimeFault("MODEL_USAGE_BOUND_EXCEEDED", 429)
        result = result.model_copy(update={"draft_id": request.draft_id})
        # 真正结果先落账；调用后策略失败不得抹掉结果或重新派发。
        await self.after_result(run, result)
        return result

    async def after_result(self, run: ClaimedRun, result: ModelResult) -> None:
        await self.extensions.run(
            "after_model",
            run,
            JsonSnapshot.capture(
                {
                    "result": result.model_dump(mode="json"),
                }
            ),
        )
