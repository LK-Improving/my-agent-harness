"""复核 Judge 运行和固定输入；不派发模型调用。"""

from harness.contracts.evaluation import EvaluationCase, EvaluationFreeze, GatePolicy, JudgePolicy
from harness.contracts.evaluation_gates import (
    GateCheck, JudgeExecutionEvidence, JudgeInput, JudgeModelEnvelope, JudgeOutput, JudgeReceipt,
    JudgeSourceMaterial, RunGateEvidence,
)
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.evaluation.assets import cases, decode
from harness.evaluation.versions import compare_versions
from harness.runtime.event_projection import project_events


def prepare_judge_input(
    collection: dict[str, object], frozen: EvaluationFreeze, blobs: dict[str, bytes], split: str,
    candidate: ReleaseManifest, case_id: str, source_ref: str, material: object,
) -> JudgeInput:
    """从已绑定的采集及显式材料生成 Judge 输入，不派发模型。"""
    if split not in {"development", "holdout"}:
        raise ValueError("评测 split 无效")
    if (collection.get("freeze_hash") != frozen.content_hash()
            or collection.get("dataset_hash") != frozen.dataset_hash):
        raise ValueError("采集与冻结包不一致")
    gate_policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
    judge_policy = JudgePolicy.model_validate(decode(blobs["judge.json"]))
    if collection.get("gate_policy_version") != gate_policy.version:
        raise ValueError("采集门禁版本不匹配")
    rows = {item.case_id: item for item in cases(blobs[f"{split}.jsonl"])}
    row = rows.get(case_id)
    records = collection.get("cases")
    if row is None or not isinstance(records, list):
        raise ValueError("目标用例不存在")
    matches = [item for item in records if isinstance(item, dict) and item.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError("目标用例缺失或重复")
    record = matches[0]
    try:
        ledger = RunGateEvidence.model_validate(record.get("ledger_evidence"))
        context = EvaluationContext.model_validate(collection.get("evaluation_context"))
        final = record.get("final")
        raw_events = record.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > 10000:
            raise ValueError("目标事件无效")
        events = [RuntimeEvent.model_validate(item) for item in raw_events]
        projection = project_events(ledger.run_id, events)
        versions = compare_versions(collection, frozen, (row,), candidate)
    except (TypeError, ValueError):
        raise ValueError("目标运行证据无效") from None
    if (versions["outcome"] != "passed" or ledger.request_hash != row.input_hash()
            or ledger.status != "succeeded" or ledger.requester_id != context.subject_id
            or ledger.context_id != context.context_id or not isinstance(final, dict)
            or set(ledger.run_scopes) != set(context.resource_scopes)
            or set(ledger.context_scopes) != set(context.resource_scopes)
            or final.get("status") != "succeeded" or final.get("result") is None
            or not isinstance(final.get("result"), dict)
            or not isinstance(final["result"].get("answer"), str)
            or projection.terminal != "run.completed" or projection.answer != final["result"]["answer"]
            or canonical_hash(final["result"]) != ledger.result_hash
            or not events or [event.seq for event in events] != list(range(1, len(events) + 1))
            or any(event.run_id != ledger.run_id for event in events)
            or events[0].type != "run.accepted" or events[-1].type != "run.completed"
            or sum(event.type in {"run.completed", "run.failed", "run.cancelled", "run.expired"}
                   for event in events) != 1
            or events[-1].payload.get("result_hash") != ledger.result_hash
            or record.get("expected") != row.expected):
        raise ValueError("目标运行与冻结用例或实际结果未绑定")
    if isinstance(material, dict) and material.get("version") == "evaluation-judge-material@1":
        supplied = JudgeSourceMaterial.model_validate(material)
        if (supplied.run_id != ledger.run_id or supplied.context_id != context.context_id
                or supplied.dataset_hash != frozen.dataset_hash or supplied.manifest_hash != candidate.content_hash()
                or supplied.input_hash != row.input_hash() or supplied.result_hash != ledger.result_hash
                or supplied.event_seq != ledger.event_seq or supplied.source_ref != source_ref
                or supplied.source_ref != candidate.profiles.as_object().get("resource_authority_ref")):
            raise ValueError("Judge 参考材料与目标运行不一致")
        material = supplied.material
    request = JudgeInput.model_validate({
        "case_id": row.case_id, "target_run_id": ledger.run_id,
        "target_manifest_hash": candidate.content_hash(), "freeze_hash": frozen.content_hash(),
        "input_hash": row.input_hash(), "target_result_hash": ledger.result_hash, "event_seq": ledger.event_seq,
        "message": row.message, "attachment_versions": row.attachment_versions,
        "business_options": row.business_options, "expected": row.expected, "dimensions": judge_policy.dimensions,
        "target_result": final["result"],
        "material_source_ref": source_ref, "reference_material": material,
    })
    if len(canonical_json(request.model_dump(mode="json"))) > 16000:
        raise ValueError("Judge 输入超过运行入口上限")
    return request


def judge_execution_check(
    value: object, receipt: JudgeReceipt, policy: JudgePolicy, frozen: EvaluationFreeze,
    target: RunGateEvidence, row: EvaluationCase | None, manifest: ReleaseManifest | None,
    target_context: EvaluationContext | None,
) -> GateCheck:
    """缺实际运行证明时保留 incomplete，已知绑定或版本不符时失败。"""
    try:
        proof = JudgeExecutionEvidence.model_validate(value)
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_execution_missing",))
    if row is None or manifest is None or target_context is None:
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_target_context_missing",))
    actual, judge_context, request = proof.manifest, proof.context, proof.input
    if policy.judge_manifest_hash is None:
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_manifest_not_pinned",))
    material_authority = manifest.profiles.as_object().get("resource_authority_ref")
    if not isinstance(material_authority, str) or not material_authority:
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_material_authority_missing",))
    judge_profile = actual.profiles.as_object().get("judge")
    if not isinstance(judge_profile, dict) or "implementation" not in judge_profile:
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_profile_missing",))
    if (
        actual.content_hash() != policy.judge_manifest_hash
        or judge_profile["implementation"] != policy.implementation
        or actual.prompt_hash != policy.prompt_sha256
        or actual.spec.model_profile_ref != policy.model_binding_ref
        or actual.spec.allowed_tools or actual.spec.knowledge_bindings
        or actual.spec.execution_limits.max_model_steps != 1
        or actual.profiles.as_object().get("resource_authority_ref") != "judge-target@1"
        or (judge_context.tenant_id, judge_context.subject_id, judge_context.application_id)
        != (target_context.tenant_id, target_context.subject_id, target_context.application_id)
        or judge_context.resource_scopes != target_context.resource_scopes
        or target.context_id != target_context.context_id
        or target.requester_id != target_context.subject_id
        or set(target.run_scopes) != set(target_context.resource_scopes)
        or set(target.context_scopes) != set(target_context.resource_scopes)
        or judge_context.dataset_hash != frozen.dataset_hash
        or judge_context.gate_policy_version != target_context.gate_policy_version
        or request.case_id != row.case_id or request.target_run_id != target.run_id
        or request.target_manifest_hash != manifest.content_hash()
        or request.material_source_ref != material_authority
        or request.freeze_hash != frozen.content_hash() or request.input_hash != row.input_hash()
        or request.message != row.message or request.attachment_versions != row.attachment_versions
        or request.business_options != row.business_options or request.expected != row.expected
        or request.dimensions != policy.dimensions
        or request.target_result_hash != target.result_hash or request.event_seq != target.event_seq
        or target.status != "succeeded"
        or proof.ledger.run_id == target.run_id
    ):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_execution_binding_mismatch",))
    try:
        if canonical_hash(request.target_result) != target.result_hash:
            raise ValueError("目标结果不一致")
        message = canonical_json(request.model_dump(mode="json"))
        if len(message) > 16000:
            raise ValueError("Judge 输入超过运行入口上限")
        expected_input = canonical_hash({"message": message, "attachment_versions": [], "business_options": {}})
        if proof.ledger.request_hash != expected_input:
            raise ValueError("Judge 输入与实际运行不一致")
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_input_mismatch",))

    version_input = {
        "candidate_manifest": actual.model_dump(mode="json"),
        "evaluation_context": judge_context.model_dump(mode="json"),
        "gate_policy_version": judge_context.gate_policy_version,
        "cases": [{"case_id": row.case_id, "accepted": proof.accepted.model_dump(mode="json"),
                   "events": [event.model_dump(mode="json") for event in proof.events],
                   "final": proof.final.model_dump(mode="json"),
                   "ledger_evidence": proof.ledger.model_dump(mode="json")}],
    }
    try:
        versions = compare_versions(version_input, frozen, (row,), actual)
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_execution_invalid",))
    if versions["outcome"] == "failed":
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_execution_version_mismatch",))
    if versions["outcome"] != "passed":
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_execution_version_missing",))

    events, judge_ledger = proof.events, proof.ledger
    started = [event for event in events if event.type == "answer.started"]
    terminals = [event for event in events if event.type in {"run.completed", "run.failed", "run.cancelled", "run.expired"}]
    if (
        [event.seq for event in events] != list(range(1, len(events) + 1))
        or any(event.run_id != judge_ledger.run_id for event in events)
        or events[0].type != "run.accepted" or len(terminals) != 1
        or events[-1].type != "run.completed" or len(started) != 1
        or judge_ledger.status != "succeeded" or proof.final.status != "succeeded"
        or receipt.receipt_ref != judge_ledger.run_id
        or judge_ledger.event_seq != len(events)
        or judge_ledger.requester_id != judge_context.subject_id
        or set(judge_ledger.run_scopes) != set(judge_context.resource_scopes)
        or set(judge_ledger.context_scopes) != set(judge_context.resource_scopes)
        or judge_ledger.operations or judge_ledger.tool_dispatches
    ):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_run_evidence_invalid",))
    calls = [call for call in judge_ledger.model_calls if call.call_id == receipt.model_call_id]
    if (len(calls) != 1 or calls[0].state != "succeeded"
            or started[0].payload.get("model_call_id") != receipt.model_call_id
            or judge_ledger.model_steps != len(judge_ledger.model_calls)
            or judge_ledger.model_steps > actual.spec.execution_limits.max_model_steps
            or len({call.call_id for call in judge_ledger.model_calls}) != len(judge_ledger.model_calls)):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_model_call_mismatch",))
    if calls[0].response_content_hash is None or proof.model_response is None:
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_model_response_missing",))
    if canonical_hash(proof.model_response) != calls[0].response_content_hash:
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_model_response_mismatch",))
    if (judge_ledger.missing or any(call.state not in {"succeeded", "failed"} for call in judge_ledger.model_calls)
            or judge_ledger.reserved_extension_tokens is None
            or judge_ledger.validation_checks is None or judge_ledger.validation_verdict is None
            or any(value is None for value in (judge_ledger.max_input_tokens, judge_ledger.max_output_tokens,
                                                judge_ledger.max_total_tokens, judge_ledger.max_execution_ms))):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_ledger_incomplete",))
    if (not judge_ledger.validation_checks or not all(judge_ledger.validation_checks.values())
            or judge_ledger.validation_verdict != "pass"):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_validation_failed",))
    reserved = sum(call.input_tokens + call.reserved_output_tokens for call in judge_ledger.model_calls)
    if (judge_ledger.reserved_extension_tokens != 0 or judge_ledger.reserved_tokens != reserved
            or reserved > int(judge_ledger.max_total_tokens or 0)
            or judge_ledger.execution_used_ms > int(judge_ledger.max_execution_ms or 0)
            or any(call.input_tokens > int(judge_ledger.max_input_tokens or 0)
                   or call.reserved_output_tokens > int(judge_ledger.max_output_tokens or 0)
                   or call.input_tokens + call.reserved_output_tokens > call.remaining_total_tokens
                   or (call.reported_input_tokens is not None and call.reported_input_tokens > call.input_tokens)
                   or (call.output_tokens is not None and call.output_tokens > call.reserved_output_tokens)
                   for call in judge_ledger.model_calls)):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_budget_mismatch",))

    result = proof.final.result
    if (not isinstance(result, dict) or not isinstance(result.get("answer"), str)
            or result.get("answer_kind") != "answer" or result.get("task_outcome") != "achieved"):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_output_missing",))
    try:
        envelope = JudgeModelEnvelope.model_validate(decode(proof.model_response.encode("utf-8")))
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_model_response_invalid",))
    task = result.get("task")
    if (envelope.answer != result["answer"] or envelope.evidence_refs
            or not isinstance(task, dict) or task.get("source_turn_id") != judge_ledger.run_id
            or task.get("evidence_refs") not in ([], ())):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_model_response_mismatch",))
    try:
        projection = project_events(judge_ledger.run_id, events)
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_output_binding_mismatch",))
    try:
        result_hash = canonical_hash(result)
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_output_invalid",))
    if (projection.terminal != "run.completed" or projection.answer != result["answer"]
            or result_hash != judge_ledger.result_hash
            or events[-1].payload.get("result_hash") != judge_ledger.result_hash):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_output_binding_mismatch",))
    try:
        output = JudgeOutput.model_validate(decode(result["answer"].encode("utf-8")))
    except (TypeError, ValueError):
        return GateCheck(gate="judge", outcome="incomplete", reasons=("judge_output_invalid",))
    if (output.case_id != row.case_id or output.target_run_id != target.run_id
            or output.target_result_hash != target.result_hash or output.freeze_hash != frozen.content_hash()
            or output.scores != receipt.scores):
        return GateCheck(gate="judge", outcome="failed", reasons=("judge_output_binding_mismatch",))
    return GateCheck(gate="judge", outcome="passed")
