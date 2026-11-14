"""顺序采集独立 Judge 运行证明；不将评分当作发布授权。"""

import asyncio
from collections.abc import Callable
from typing import Protocol

from harness.contracts.evaluation import CollectionFault, EvaluationAPI, EvaluationCase, EvaluationFreeze, GatePolicy, JudgePolicy
from harness.contracts.evaluation_contexts import EvaluationLimits
from harness.contracts.evaluation_gates import (
    JudgeExecutionEvidence, JudgeInput, JudgeModelResponse, JudgeOutput, JudgeReceipt, RunGateEvidence,
)
from harness.contracts.evaluation_observations import CaseObservations, ObservationBundle
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_json
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import RunView
from harness.evaluation.assets import cases, decode
from harness.evaluation.collector import collect
from harness.evaluation.gates import judge_check
from harness.evaluation.judge import prepare_judge_input


class JudgeEvaluationAPI(EvaluationAPI, Protocol):
    async def evaluation_evidence(self, run_id: str) -> RunGateEvidence: ...

    async def judge_model_response(self, run_id: str, call_id: str) -> JudgeModelResponse: ...


class JudgeBindingMismatch(Exception):
    """已观察到评分响应与请求身份不一致。"""


async def collect_judges(
    api: JudgeEvaluationAPI, frozen: EvaluationFreeze, blobs: dict[str, bytes], split: str,
    target_collection: dict[str, object], target: ReleaseManifest, judge: ReleaseManifest,
    inputs: tuple[JudgeInput, ...], batch_id: str, emit: Callable[[dict[str, object]], None],
    *, limits: EvaluationLimits | None = None, lifetime_seconds: int = 3600,
) -> tuple[dict[str, object], ObservationBundle | None]:
    """先校验所有输入，再通过单次创建和持久 SSE 采集，异常不重发创建。"""
    if split not in {"development", "holdout"} or not inputs or len(inputs) > 1000:
        raise ValueError("Judge 批次或 split 无效")
    policy = JudgePolicy.model_validate(decode(blobs["judge.json"]))
    gate = GatePolicy.model_validate(decode(blobs["gates.json"]))
    profile = judge.profiles.as_object().get("judge")
    if (not policy.required or not all((policy.implementation, policy.prompt_sha256,
                                        policy.model_binding_ref, policy.judge_manifest_hash))
            or not isinstance(profile, dict) or profile.get("implementation") != policy.implementation
            or judge.content_hash() != policy.judge_manifest_hash or judge.prompt_hash != policy.prompt_sha256
            or judge.spec.model_profile_ref != policy.model_binding_ref or judge.spec.allowed_tools
            or judge.spec.knowledge_bindings or judge.spec.execution_limits.max_model_steps != 1
            or judge.profiles.as_object().get("resource_authority_ref") != "judge-target@1"
            or len(set(policy.dimensions)) != len(policy.dimensions)):
        raise ValueError("Judge 清单与冻结策略不一致")
    target_context = EvaluationContext.model_validate(target_collection.get("evaluation_context"))
    rows = {row.case_id: row for row in cases(blobs[f"{split}.jsonl"])}
    records = target_collection.get("cases")
    if not isinstance(records, list):
        raise ValueError("目标采集缺少用例")
    if len({item.case_id for item in inputs}) != len(inputs):
        raise ValueError("Judge 输入不能重复用例")
    selected_limits = limits or EvaluationLimits()
    if (len(inputs) > selected_limits.max_sessions or len(inputs) > selected_limits.max_runs
            or not 1 <= lifetime_seconds <= 86400):
        raise ValueError("Judge 批次超出额度或有效期非法")
    material_authority = target.profiles.as_object().get("resource_authority_ref")
    if not isinstance(material_authority, str) or not material_authority:
        raise ValueError("目标缺少材料授权权威")
    actual_rows: list[EvaluationCase] = []
    target_ledgers: dict[str, RunGateEvidence] = {}
    input_by_case: dict[str, JudgeInput] = {}
    for request in inputs:
        row = rows.get(request.case_id)
        matching = [record for record in records if isinstance(record, dict)
                    and record.get("case_id") == request.case_id]
        if row is None or len(matching) != 1 or request.material_source_ref != material_authority:
            raise ValueError("Judge 目标或材料权威不一致")
        prepared = prepare_judge_input(target_collection, frozen, blobs, split, target, request.case_id,
                                       material_authority, request.reference_material)
        if prepared != request:
            raise ValueError("Judge 输入与目标采集不一致")
        target_ledgers[row.case_id] = RunGateEvidence.model_validate(matching[0].get("ledger_evidence"))
        input_by_case[row.case_id] = request
        actual_rows.append(EvaluationCase(case_id=row.case_id,
            message=canonical_json(request.model_dump(mode="json")), expected=row.expected))

    run_report = await collect(
        api, frozen, tuple(actual_rows), judge, gate.version, batch_id, target_context.resource_scopes, emit,
        limits=selected_limits, lifetime_seconds=lifetime_seconds,
        expected_identity=(target_context.tenant_id, target_context.subject_id, target_context.application_id),
    )
    raw_cases = run_report.get("cases")
    raw_context = run_report.get("evaluation_context")
    case_reports: list[dict[str, object]] = []
    observations: list[CaseObservations] = []
    for entry in raw_cases if isinstance(raw_cases, list) else []:
        if not isinstance(entry, dict) or entry.get("case_id") not in input_by_case:
            continue
        case_id = entry["case_id"]
        case_report: dict[str, object] = {"case_id": case_id, "outcome": "incomplete"}
        case_reports.append(case_report)
        try:
            context = EvaluationContext.model_validate(raw_context)
            accepted = RunView.model_validate(entry.get("accepted"))
            final = RunView.model_validate(entry.get("final"))
            events = tuple(RuntimeEvent.model_validate(item) for item in entry.get("events", []))
            ledger = RunGateEvidence.model_validate(entry.get("ledger_evidence"))
            started = [event for event in events if event.type == "answer.started"]
            if final.status != "succeeded" or len(started) != 1 or not isinstance(final.result, dict):
                raise ValueError("Judge 未提交有效评分")
            call_id = started[0].payload.get("model_call_id")
            if not isinstance(call_id, str):
                raise ValueError("Judge 模型调用缺失")
            response = await api.judge_model_response(accepted.run_id, call_id)
            if response.run_id != accepted.run_id or response.model_call_id != call_id:
                raise JudgeBindingMismatch()
            answer = final.result.get("answer")
            if not isinstance(answer, str):
                raise ValueError("Judge 评分正文缺失")
            output = JudgeOutput.model_validate(decode(answer.encode("utf-8")))
            target_ledger = target_ledgers[case_id]
            receipt = JudgeReceipt(run_id=target_ledger.run_id, freeze_hash=frozen.content_hash(),
                result_hash=target_ledger.result_hash, policy_version=policy.version,
                implementation=policy.implementation, prompt_sha256=policy.prompt_sha256,
                model_binding_ref=policy.model_binding_ref, model_call_id=call_id, receipt_ref=accepted.run_id,
                state="succeeded", scores=output.scores)
            proof = JudgeExecutionEvidence(manifest=judge, context=context, input=input_by_case[case_id],
                accepted=accepted, final=final, events=events, ledger=ledger, model_response=response.content)
            verdict = judge_check(policy, receipt, frozen, target_ledger, execution=proof,
                                  row=rows[case_id], manifest=target, target_context=target_context)
            case_report.update({"outcome": verdict.outcome, "reasons": list(verdict.reasons),
                                "judge_run_id": accepted.run_id})
            if verdict.outcome == "passed":
                observations.append(CaseObservations(case_id=case_id, judge_receipt=receipt, judge_execution=proof))
        except JudgeBindingMismatch:
            case_report.update({"outcome": "failed", "reasons": ["judge_response_identity_mismatch"]})
        except CollectionFault as error:
            if str(error) == "JUDGE_RESPONSE_IDENTITY_MISMATCH":
                case_report.update({"outcome": "failed", "reasons": ["judge_response_identity_mismatch"]})
            else:
                case_report["error_type"] = type(error).__name__
        except Exception as error:
            # 读取被撤权、响应不完整或超时均保留缺口；不把异常正文或原始凭据写日志。
            case_report["error_type"] = type(error).__name__
        await asyncio.to_thread(emit, {"phase": "judge_evidence", **case_report})

    if any(item["outcome"] == "failed" for item in case_reports):
        outcome = "failed"
    elif len(observations) == len(inputs) and run_report.get("collection_complete") is True:
        outcome = "passed"
    else:
        outcome = "incomplete"
    result: dict[str, object] = {"version": "evaluation-judge-batch@1", "batch_id": batch_id,
        "freeze_hash": frozen.content_hash(), "target_manifest_hash": target.content_hash(),
        "judge_manifest_hash": judge.content_hash(), "split": split, "outcome": outcome,
        "expected_case_ids": [item.case_id for item in inputs], "cases": case_reports,
        "collection_error_code": run_report.get("collection_error_code"), "collection": run_report}
    bundle = (ObservationBundle(freeze_hash=frozen.content_hash(), manifest_hash=target.content_hash(),
                                cases=tuple(observations)) if observations else None)
    return result, bundle
