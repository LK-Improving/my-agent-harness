"""构造即冻结的扩展链；仅传深快照，不暴露底层派发函数。"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from harness.contracts.extensions import (
    STAGE_ORDER,
    DecisionRecorder,
    ExtensionDecision,
    ExtensionInvocation,
    ExtensionSpec,
    ExtensionStage,
)
from harness.contracts.runtime import ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ExecutionContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegisteredExtension:
    spec: ExtensionSpec
    handler: Callable[[ExtensionInvocation], Awaitable[ExtensionDecision | None]] = field(
        repr=False
    )


@dataclass(frozen=True, slots=True, init=False)
class ExtensionChain:
    registrations: tuple[RegisteredExtension, ...]
    snapshot: JsonSnapshot
    record_decision: DecisionRecorder | None = field(repr=False)

    def __init__(
        self,
        registrations: tuple[RegisteredExtension, ...] = (),
        *,
        record_decision: DecisionRecorder | None = None,
    ) -> None:
        entries = tuple(
            RegisteredExtension(
                ExtensionSpec.model_validate_json(entry.spec.model_dump_json()), entry.handler
            )
            for entry in registrations
        )
        if len(entries) > 64:
            raise ValueError("扩展注册不能超过64项")
        identities = [(entry.spec.extension_id, entry.spec.stage) for entry in entries]
        positions = [(entry.spec.stage, entry.spec.order) for entry in entries]
        if len(set(identities)) != len(entries) or len(set(positions)) != len(entries):
            raise ValueError("扩展身份或阶段顺序重复/歧义")
        versions: dict[str, str] = {}
        for entry in entries:
            if not inspect.iscoroutinefunction(entry.handler):
                raise ValueError("扩展必须显式注册异步函数")
            previous = versions.setdefault(entry.spec.extension_id, entry.spec.version)
            if previous != entry.spec.version:
                raise ValueError("同一扩展跨阶段必须使用同一版本")
        entries = tuple(
            sorted(
                entries, key=lambda entry: (STAGE_ORDER.index(entry.spec.stage), entry.spec.order)
            )
        )
        object.__setattr__(self, "registrations", entries)
        object.__setattr__(self, "record_decision", record_decision)
        object.__setattr__(
            self,
            "snapshot",
            JsonSnapshot.capture(
                {
                    "version": "fixed-extensions@1",
                    "stage_order": list(STAGE_ORDER),
                    "registrations": [entry.spec.model_dump(mode="json") for entry in entries],
                }
            ),
        )

    def require_bound(self, run: ClaimedRun) -> None:
        installed = run.manifest.profiles.as_object().get("extensions")
        # 仅兼容旧的空链调用方；缺少绑定绝不允许安装非空扩展。
        if installed is None and not self.registrations:
            return
        if installed != self.snapshot.materialize():
            raise RuntimeFault("EXTENSION_MANIFEST_MISMATCH", 503)

    @staticmethod
    def _apply(
        stage: ExtensionStage, payload: JsonSnapshot, decision: ExtensionDecision
    ) -> JsonSnapshot:
        data = payload.as_object()
        if decision.context_prefix is not None:
            if stage != "before_model":
                raise ValueError("模型输入只能在before_model调整")
            prefix = decision.context_prefix.as_object_array()
            if any(
                set(item) != {"role", "content"}
                or item["role"] != "user"
                or not isinstance(item["content"], str)
                for item in prefix
            ):
                raise ValueError("扩展上下文仅允许额外文本用户消息，不得伪造工具或系统角色")
            messages = data.get("messages")
            if not isinstance(messages, list):
                raise ValueError("模型上下文必须是数组")
            # 原system和当前目标/工具组全部保留，只插入额外上下文，不允许替换历史。
            split = 0
            while (
                split < len(messages)
                and isinstance(message := messages[split], dict)
                and message.get("role") == "system"
            ):
                split += 1
            data["messages"] = [*messages[:split], *prefix, *messages[split:]]
        if decision.tool_arguments is not None:
            if stage != "before_tool":
                raise ValueError("工具参数只能在提案/审批前修改")
            data["arguments"] = decision.tool_arguments.as_object()
        return JsonSnapshot.capture(data)

    async def run(
        self, stage: ExtensionStage, run: ClaimedRun, payload: JsonSnapshot
    ) -> JsonSnapshot:
        self.require_bound(run)
        current = payload
        context = ExecutionContext(
            **run.principal.model_dump(mode="python"),
            run_id=run.run_id,
            release_id=run.manifest.spec.release_id,
            trace_id=run.run_id,
        )
        for entry in self.registrations:
            if entry.spec.stage != stage:
                continue
            try:
                if entry.spec.mode == "policy" and self.record_decision is None:
                    raise ValueError("关键扩展缺少持久决定账本")
                if len(current.canonical_text.encode("utf-8")) > 1024 * 1024:
                    raise ValueError("扩展输入超过1MiB")
                remaining = (run.execution_deadline - datetime.now(UTC)).total_seconds()
                if remaining <= 0:
                    raise RuntimeFault("EXTENSION_EXECUTION_EXPIRED", 408)
                invocation = ExtensionInvocation(
                    context=context,
                    manifest_hash=run.manifest.content_hash(),
                    stage=stage,
                    payload=current,
                    configuration=entry.spec.configuration,
                )
                timeout = min(entry.spec.timeout_ms / 1000, remaining)
                started = asyncio.get_running_loop().time()
                async with asyncio.timeout(timeout):
                    decision = await entry.handler(invocation)
                if asyncio.get_running_loop().time() - started >= timeout:
                    raise TimeoutError("扩展超过固定时限")
                if entry.spec.mode == "observer":
                    if decision is not None:
                        raise ValueError("观察扩展不得返回策略决定")
                    continue
                if not isinstance(decision, ExtensionDecision):
                    raise ValueError("关键扩展必须返回绑定的类型化决定")
                decision = ExtensionDecision.model_validate_json(decision.model_dump_json())
                if len(decision.model_dump_json().encode("utf-8")) > 65536:
                    raise ValueError("扩展决定不能超过64KiB")
                if (
                    decision.extension_id != entry.spec.extension_id
                    or decision.version != entry.spec.version
                    or decision.invocation_fingerprint != invocation.fingerprint()
                ):
                    raise ValueError("扩展决定身份不匹配")
                if self.record_decision is not None:
                    await self.record_decision(run, entry.spec, invocation, decision)
                if decision.verdict == "deny":
                    raise RuntimeFault("EXTENSION_POLICY_DENIED", 403)
                current = self._apply(stage, current, decision)
                if len(current.canonical_text.encode("utf-8")) > 1024 * 1024:
                    raise ValueError("扩展输出超过1MiB")
            except Exception:
                if entry.spec.mode == "observer":
                    # 非关键观察失败隔离，不导出原异常/正文；取消不在Exception内吞掉。
                    logger.warning(
                        "extension_observer_failed extension_id=%s stage=%s",
                        entry.spec.extension_id,
                        stage,
                    )
                    continue
                raise RuntimeFault("EXTENSION_POLICY_BLOCKED", 403) from None
        return current
