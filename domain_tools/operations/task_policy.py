"""以用户本轮明确给出的订单约束工具目标，不能拿模型提案替换原任务。"""

import re
from typing import Any

from harness.contracts.runtime import ClaimedRun, RuntimeFault
from harness.contracts.specs import ToolSpec


def task_orders(message: str) -> frozenset[str]:
    return frozenset(
        re.findall(r"(?<![A-Za-z0-9_-])synthetic-[A-Za-z0-9_-]{1,40}(?![A-Za-z0-9_-])", message)
    )


def require_task_target(run: ClaimedRun, spec: ToolSpec, arguments: dict[str, Any]) -> None:
    targets = task_orders(run.request.message)
    actual = arguments.get("order_id")
    if not isinstance(actual, str):
        raise RuntimeFault("ORDER_ARGUMENT_INVALID", 422)
    if spec.effect == "write" and (len(targets) != 1 or actual not in targets):
        raise RuntimeFault("WRITE_TARGET_NOT_CONFIRMED_BY_USER", 403)
    if spec.effect == "read" and targets and actual not in targets:
        raise RuntimeFault("READ_TARGET_NOT_IN_TASK", 403)
