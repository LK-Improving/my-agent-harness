"""将费用观测的调用清单与已有运行账本比对，不推断来源可信。"""

from harness.contracts.evaluation import JudgePolicy
from harness.contracts.evaluation_gates import JudgeExecutionEvidence, ModelCallEvidence, RunGateEvidence
from harness.contracts.evaluation_metrics import CostCategory, MetricObservation
from harness.contracts.events import RuntimeEvent
from harness.contracts.release import ReleaseManifest

COST_CATEGORIES: tuple[CostCategory, ...] = ("model", "tool", "retrieval", "judge")


def inventory_matches(
    observed: MetricObservation, ledger: RunGateEvidence, events: list[RuntimeEvent], manifest: ReleaseManifest,
    judge: JudgePolicy, judge_execution: object, *, budget_passed: bool, unknown_passed: bool,
    judge_passed: bool,
) -> bool:
    """只确认调用清单完整；调用身份一致不代表费用来源或金额可信。"""
    if ledger.missing or not budget_passed or not unknown_passed or ledger.model_steps != len(ledger.model_calls):
        return False
    model_calls = tuple(ledger.model_calls)
    if any(call.state not in {"succeeded", "failed"} for call in model_calls):
        return False
    model_ids = {call.call_id for call in model_calls}
    for event in events:
        if event.type == "answer.started" and event.payload.get("producer") == "model":
            call_id = event.payload.get("model_call_id")
            if not isinstance(call_id, str) or call_id not in model_ids:
                return False
    tool_ids: list[str] = []
    for operation in ledger.operations:
        if operation.state not in {"prepared", "succeeded", "not_executed", "rejected", "failed"}:
            return False
        if operation.effect == "read":
            if operation.recorded_attempts != 1:
                return False
            tool_ids.append(operation.operation_id)
        else:
            if operation.recorded_attempts != len(operation.attempts):
                return False
            tool_ids.extend(attempt.attempt_id for attempt in operation.attempts)
    if ledger.tool_dispatches != len(tool_ids):
        return False

    retrieval_ids: list[str] = []
    if ledger.retrieval_children is None:
        if manifest.spec.knowledge_bindings or ledger.reserved_extension_tokens:
            return False
    else:
        for child in ledger.retrieval_children:
            if child.state in {"queued", "running"}:
                return False
            if child.context_scope is not None and child.calls_used != len(child.calls):
                return False
            if child.context_scope is None and child.calls:
                return False
            retrieval_ids.extend(call.call_id for call in child.calls)

    judge_calls: tuple[ModelCallEvidence, ...] = ()
    if judge.required:
        if not judge_passed:
            return False
        try:
            proof = JudgeExecutionEvidence.model_validate(judge_execution)
        except (TypeError, ValueError):
            return False
        judge_calls = proof.ledger.model_calls
    elif judge_execution is not None:
        # 非必需Judge若存在额外运行，当前无法证明其成本边界。
        return False

    actual: dict[CostCategory, list[str]] = {
        "model": [call.call_id for call in model_calls],
        "tool": tool_ids,
        "retrieval": retrieval_ids,
        "judge": [call.call_id for call in judge_calls],
    }
    if any(len(ids) != len(set(ids)) for ids in actual.values()):
        return False
    for name in COST_CATEGORIES:
        cost = observed.costs.get(name)
        if (cost is None or cost.state == "unknown" or cost.covered_call_ids is None
                or set(cost.covered_call_ids) != set(actual[name])):
            return False
        # 现有账本没有对任意已发生调用的不计费权威证明。
        if cost.state == "not_billable" and actual[name]:
            return False
        if cost.state == "known" and name in {"model", "judge"}:
            calls = model_calls if name == "model" else judge_calls
            if any(call.price_version_ref is None or call.price_version_ref != cost.price_version for call in calls):
                return False
    return True
