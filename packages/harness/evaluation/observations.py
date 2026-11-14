"""将外部来源的独立观测绑定到采集副本。"""

from copy import deepcopy
from typing import Any, Literal, cast

from harness.contracts.evaluation import EvaluationFreeze, GatePolicy, JudgePolicy
from harness.contracts.evaluation_gates import RunGateEvidence
from harness.contracts.evaluation_observations import ObservationBundle
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.evaluation.assets import cases, decode
from harness.evaluation.gates import evaluate_case, judge_check


def merge_observations(
    collection: dict[str, object], frozen: EvaluationFreeze, blobs: dict[str, bytes],
    split: Literal["development", "holdout"], candidate: ReleaseManifest, bundle: ObservationBundle,
) -> dict[str, object]:
    """验证已有运行证据及观测绑定，保留输入与历史同名回执。"""
    if (
        bundle.freeze_hash != frozen.content_hash()
        or bundle.manifest_hash != candidate.content_hash()
        or collection.get("freeze_hash") != bundle.freeze_hash
        or collection.get("dataset_hash") != frozen.dataset_hash
        or ReleaseManifest.model_validate(collection.get("candidate_manifest")).content_hash() != bundle.manifest_hash
    ):
        raise ValueError("观测批次与冻结包或候选不一致")
    rows = {row.case_id: row for row in cases(blobs[f"{split}.jsonl"])}
    policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
    judge = JudgePolicy.model_validate(decode(blobs["judge.json"]))
    if collection.get("gate_policy_version") != policy.version:
        raise ValueError("采集门禁版本不匹配")
    target_context = EvaluationContext.model_validate(collection.get("evaluation_context"))
    records = collection.get("cases")
    if not isinstance(records, list):
        raise ValueError("采集用例缺失")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str):
            raise ValueError("采集用例格式无效")
        case_id = record["case_id"]
        if case_id not in rows or case_id in indexed:
            raise ValueError("采集用例重复或不属于当前 split")
        indexed[case_id] = record

    additions: list[tuple[str, str, dict[str, Any]]] = []
    for observed in bundle.cases:
        row = rows.get(observed.case_id)
        record = indexed.get(observed.case_id)
        if row is None or record is None:
            raise ValueError("独立观测找不到对应采集用例")
        evaluated = evaluate_case(row, record, frozen, candidate, judge, gate_policy_version=policy.version,
                                  target_context=target_context)
        outcomes = {item.gate: item.outcome for item in evaluated.checks}
        required = ("case_execution", "persisted_sse", "final_result")
        invalid = ("evidence_identity", "case_input", "persisted_audit", "terminal_result")
        if any(outcomes.get(name) != "passed" for name in required) or any(name in outcomes for name in invalid):
            raise ValueError("采集运行的持久事件或最终结果未绑定")
        ledger = RunGateEvidence.model_validate(record.get("ledger_evidence"))
        if (ledger.manifest_hash != bundle.manifest_hash
                or ledger.environment_profile_hash != candidate.environment_profile_hash):
            raise ValueError("采集账本与候选版本不一致")
        external = observed.external_observation
        if external is not None and (
            external.run_id != ledger.run_id or external.freeze_hash != bundle.freeze_hash
            or external.result_hash != ledger.result_hash or external.event_seq != ledger.event_seq
            or (external.manifest_hash is not None and external.manifest_hash != bundle.manifest_hash)
            or ("credential_exposure" in external.coverage and external.manifest_hash != bundle.manifest_hash)
        ):
            raise ValueError("外部观测与运行证据不一致")
        metric = observed.metric_observation
        if metric is not None and (
            metric.case_id != row.case_id or metric.run_id != ledger.run_id
            or metric.freeze_hash != bundle.freeze_hash or metric.manifest_hash != bundle.manifest_hash
            or metric.result_hash != ledger.result_hash or metric.event_seq != ledger.event_seq
        ):
            raise ValueError("指标观测与运行证据不一致")
        receipt = observed.judge_receipt
        if receipt is not None and (
            not judge.required or not judge.implementation or not judge.prompt_sha256 or not judge.model_binding_ref
            or receipt.run_id != ledger.run_id or receipt.freeze_hash != bundle.freeze_hash
            or receipt.result_hash != ledger.result_hash or receipt.policy_version != judge.version
            or receipt.implementation != judge.implementation or receipt.prompt_sha256 != judge.prompt_sha256
            or receipt.model_binding_ref != judge.model_binding_ref
        ):
            raise ValueError("Judge 回执与冻结策略或运行证据不一致")
        execution = observed.judge_execution
        if execution is not None:
            supplied_receipt = receipt or record.get("judge_receipt")
            if supplied_receipt is None:
                raise ValueError("Judge 运行证明缺少回执")
            verdict = judge_check(judge, supplied_receipt, frozen, ledger, execution=execution,
                                  row=row, manifest=candidate, target_context=target_context)
            if verdict.outcome == "failed":
                raise ValueError("Judge 运行证明与回执或目标不一致")
        for name in ("external_observation", "judge_receipt", "judge_execution", "metric_observation"):
            value = getattr(observed, name)
            if value is None:
                continue
            body = value.model_dump(mode="json")
            existing = record.get(name)
            if existing is not None and type(value).model_validate(existing).model_dump(mode="json") != body:
                raise ValueError("不能覆盖同名独立观测")
            additions.append((row.case_id, name, body))

    result = deepcopy(collection)
    output_records = cast(list[dict[str, Any]], result["cases"])
    output_index = {record["case_id"]: record for record in output_records}
    for case_id, name, body in additions:
        output_index[case_id][name] = body
    return result
