"""工具改参的唯一前置阶段；审批身份产生前重做Schema、权限与领域守卫。"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ToolSpec
from harness.runtime.extensions import ExtensionChain

TaskGuard = Callable[[ClaimedRun, ToolSpec, dict[str, Any]], None]


async def prepare_arguments(
    run: ClaimedRun,
    spec: ToolSpec,
    input_type: type[BaseModel],
    authorize: Callable[[Principal, dict[str, Any]], bool],
    arguments: dict[str, Any],
    logical_id: str,
    extensions: ExtensionChain,
    task_guard: TaskGuard | None,
) -> JsonSnapshot:
    if {binding.name: binding.version for binding in run.manifest.spec.allowed_tools}.get(
        spec.name
    ) != spec.version:
        raise RuntimeFault("TOOL_NOT_ALLOWED", 403)

    def validate(value: dict[str, Any]) -> dict[str, Any]:
        try:
            result = input_type.model_validate(value, strict=True).model_dump(mode="json")
        except ValidationError:
            raise RuntimeFault("TOOL_INPUT_INVALID", 422) from None
        if not authorize(run.principal, result):
            raise RuntimeFault("TOOL_FORBIDDEN", 403)
        if task_guard is not None:
            task_guard(run, spec, result)
        return result

    original = validate(arguments)
    payload = await extensions.run(
        "before_tool",
        run,
        JsonSnapshot.capture(
            {
                "tool_name": spec.name,
                "tool_version": spec.version,
                "logical_call_id": logical_id,
                "arguments": original,
            }
        ),
    )
    return JsonSnapshot.capture(
        validate(JsonSnapshot.capture(payload.as_object()["arguments"]).as_object())
    )
