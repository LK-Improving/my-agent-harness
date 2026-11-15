"""独立 Judge 运行和模型回执绑定；全部材料均为合成夹具。"""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from harness.contracts.evaluation import JudgePolicy
from harness.contracts.evaluation_gates import JudgeInput, RunGateEvidence
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import RunView
from harness.contracts.snapshots import JsonSnapshot
from harness.evaluation.assets import cases
from harness.evaluation.gates import judge_check
from harness.evaluation.judge import prepare_judge_input
from tests.unit.test_evaluation_reports import report_case
from tests.unit.test_evaluation_versions import synthetic_version


def governed_judge_case():
    _, _, original, _ = report_case()
    target = original.model_copy(update={"profiles": JsonSnapshot.capture({
        "resource_authority_ref": "synthetic-material@1"})})
    frozen, blobs, target_manifest, collection = report_case(target)
    target = collection["cases"][0]
    target_ledger = RunGateEvidence.model_validate(target["ledger_evidence"])
    target_context = EvaluationContext.model_validate(collection["evaluation_context"])
    row = cases(blobs["holdout.jsonl"])[0]
    policy = JudgePolicy(version="judge@1", implementation="governed-judge@1", prompt_sha256="a" * 64,
                         model_binding_ref="judge-model@1", dimensions=("quality",))
    spec = target_manifest.spec.model_copy(update={"agent_id": "judge", "release_id": "judge-release@1",
        "model_profile_ref": policy.model_binding_ref, "allowed_tools": (), "knowledge_bindings": (),
        "execution_limits": target_manifest.spec.execution_limits.model_copy(update={"max_model_steps": 1})})
    profiles = dict(target_manifest.profiles.as_object())
    profiles["resource_authority_ref"] = "judge-target@1"
    profiles["judge"] = {"implementation": policy.implementation}
    judge_manifest = ReleaseManifest.model_validate({**target_manifest.model_dump(mode="json"),
        "spec": spec.model_dump(mode="json"), "prompt_hash": policy.prompt_sha256,
        "tool_schema_hashes": {}, "profiles": JsonSnapshot.capture(profiles).materialize()})
    policy = policy.model_copy(update={"judge_manifest_hash": judge_manifest.content_hash()})
    context = target_context.model_copy(update={"context_id": "judge-context", "agent_id": spec.agent_id,
        "candidate_release_id": spec.release_id, "manifest_hash": judge_manifest.content_hash(),
        "environment_profile_hash": judge_manifest.environment_profile_hash})
    judge_input = JudgeInput(case_id=row.case_id, target_run_id=target_ledger.run_id,
        target_manifest_hash=target_manifest.content_hash(), freeze_hash=frozen.content_hash(),
        input_hash=row.input_hash(), target_result_hash=target_ledger.result_hash,
        event_seq=target_ledger.event_seq, message=row.message, attachment_versions=row.attachment_versions,
        business_options=row.business_options, expected=row.expected, dimensions=policy.dimensions,
        target_result=target["final"]["result"],
        material_source_ref="synthetic-material@1", reference_material={"text": "合成资料"})
    answer = canonical_json({"version": "evaluation-judge-output@1", "case_id": row.case_id,
        "target_run_id": target_ledger.run_id, "target_result_hash": target_ledger.result_hash,
        "freeze_hash": frozen.content_hash(), "scores": {"quality": 0.8}})
    model_response = canonical_json({"answer": answer, "answer_kind": "answer", "task_outcome": "achieved"})
    result = {"answer": answer, "answer_kind": "answer", "task_outcome": "achieved",
              "task": {"source_turn_id": "judge-run", "evidence_refs": []}}
    digest = canonical_hash(result)
    now = datetime.now(UTC)
    version = synthetic_version(judge_manifest).model_dump(mode="json")
    event_base = {"run_id": "judge-run", "timestamp": now}
    events = [
        RuntimeEvent(**event_base, seq=1, type="run.accepted", payload={"release_id": spec.release_id,
            "manifest_hash": judge_manifest.content_hash(), "environment_profile_hash": judge_manifest.environment_profile_hash}),
        RuntimeEvent(**event_base, seq=2, type="run.version_observed", payload={"phase": "start",
            "execution_identity": "judge-worker", "observation": version, "error_code": None}),
        RuntimeEvent(**event_base, seq=3, type="run.started", payload={"execution_identity": "judge-worker",
            "version_evidence_seq": 2, "observed_manifest_hash": judge_manifest.content_hash(),
            "observed_environment_profile_hash": judge_manifest.environment_profile_hash}),
        RuntimeEvent(**event_base, seq=4, type="answer.started", payload={"draft_id": "judge-draft",
            "producer": "model", "model_call_id": "judge-call", "model_attempt_id": "judge-attempt"}),
        RuntimeEvent(**event_base, seq=5, type="answer.delta", payload={"draft_id": "judge-draft",
            "model_attempt_id": "judge-attempt", "delta": answer}),
        RuntimeEvent(**event_base, seq=6, type="run.version_observed", payload={"phase": "terminal",
            "execution_identity": "judge-worker", "observation": version, "error_code": None}),
        RuntimeEvent(**event_base, seq=7, type="run.completed", payload={"committed_draft_id": "judge-draft",
            "commit_id": "judge-commit", "result_hash": digest, "version_evidence_seq": 6,
            "observed_manifest_hash": judge_manifest.content_hash(),
            "observed_environment_profile_hash": judge_manifest.environment_profile_hash}),
    ]
    view = RunView(run_id="judge-run", session_id="judge-session", status="queued", revision=0,
        release_id=spec.release_id, manifest_hash=judge_manifest.content_hash(),
        environment_profile_hash=judge_manifest.environment_profile_hash, cancel_requested=False,
        recovery_required=False, created_at=now)
    final = view.model_copy(update={"status": "succeeded", "result": result})
    message = canonical_json(judge_input.model_dump(mode="json"))
    ledger = RunGateEvidence(run_id=view.run_id, session_id=view.session_id, context_id=context.context_id,
        dataset_hash=frozen.dataset_hash, gate_policy_version=context.gate_policy_version,
        request_hash=canonical_hash({"message": message, "attachment_versions": [], "business_options": {}}),
        manifest_hash=judge_manifest.content_hash(), environment_profile_hash=judge_manifest.environment_profile_hash,
        snapshot_at=now, status="succeeded", requester_id=context.subject_id, event_seq=len(events), result_hash=digest,
        run_scopes=tuple(context.resource_scopes), context_scopes=tuple(context.resource_scopes), operations=(),
        model_calls=({"call_id": "judge-call", "state": "succeeded", "input_tokens": 10,
            "reserved_output_tokens": 20, "output_tokens": 5, "remaining_total_tokens": 100,
            "response_content_hash": canonical_hash(model_response)},),
        model_steps=1, reserved_tokens=30, tool_dispatches=0, execution_used_ms=1,
        max_input_tokens=100, max_output_tokens=20, max_total_tokens=100, max_execution_ms=1000,
        reserved_extension_tokens=0, validation_checks={"judge_output": True}, validation_verdict="pass", events=(),
        event_hashes=tuple(canonical_hash(event.model_dump(mode="json")) for event in events))
    proof = {"manifest": judge_manifest.model_dump(mode="json"), "context": context.model_dump(mode="json"),
             "input": judge_input.model_dump(mode="json"), "accepted": view.model_dump(mode="json"),
             "final": final.model_dump(mode="json"), "events": [event.model_dump(mode="json") for event in events],
             "ledger": ledger.model_dump(mode="json"), "model_response": model_response}
    receipt = {"run_id": target_ledger.run_id, "freeze_hash": frozen.content_hash(),
        "result_hash": target_ledger.result_hash, "policy_version": policy.version,
        "implementation": policy.implementation, "prompt_sha256": policy.prompt_sha256,
        "model_binding_ref": policy.model_binding_ref, "model_call_id": "judge-call", "receipt_ref": "judge-run",
        "state": "succeeded", "scores": {"quality": 0.8}}
    return policy, receipt, frozen, target_ledger, row, target_manifest, target_context, proof


def verdict(materials):
    policy, receipt, frozen, ledger, row, manifest, context, proof = materials
    return judge_check(policy, receipt, frozen, ledger, execution=proof, row=row,
                       manifest=manifest, target_context=context)


def test_judge_requires_real_run_shape_and_actual_model_call_binding():
    materials = governed_judge_case()
    assert verdict(materials).outcome == "passed"
    policy, receipt, frozen, ledger, row, manifest, context, _ = materials
    assert judge_check(policy, receipt, frozen, ledger, row=row, manifest=manifest,
                       target_context=context).outcome == "incomplete"


def test_judge_rejects_target_subject_and_scope_mismatch():
    for update in ({"requester_id": "other"}, {"run_scopes": ("other",), "context_scopes": ("other",)}):
        materials = list(governed_judge_case())
        materials[3] = materials[3].model_copy(update=update)
        assert verdict(materials).outcome == "failed"


def test_missing_judge_manifest_or_model_response_keeps_evaluation_incomplete():
    materials = list(governed_judge_case())
    materials[0] = materials[0].model_copy(update={"judge_manifest_hash": None})
    assert verdict(materials).outcome == "incomplete"
    materials = list(governed_judge_case())
    materials[-1]["model_response"] = None
    assert verdict(materials).outcome == "incomplete"


@pytest.mark.parametrize("mutation", ["input", "prompt", "event", "score", "identity", "call", "response", "receipt_ref"])
def test_judge_evidence_drift_is_not_accepted(mutation):
    materials = list(governed_judge_case())
    proof = deepcopy(materials[-1])
    materials[-1] = proof
    if mutation == "input":
        proof["input"]["reference_material"] = {"text": "被替换"}
    elif mutation == "prompt":
        proof["manifest"]["prompt_hash"] = "b" * 64
    elif mutation == "event":
        proof["events"][4]["payload"]["delta"] = "被替换"
    elif mutation == "score":
        materials[1] = {**materials[1], "scores": {"quality": 0.9}}
    elif mutation == "identity":
        proof["context"]["tenant_id"] = "other-tenant"
    elif mutation == "call":
        proof["ledger"]["model_calls"][0]["state"] = "unknown"
    elif mutation == "receipt_ref":
        materials[1] = {**materials[1], "receipt_ref": "other-judge-run"}
    else:
        proof["model_response"] = "被替换"
    assert verdict(materials).outcome == "failed"


def test_prepare_judge_input_binds_case_result_and_supplied_material():
    frozen, blobs, manifest, collection = report_case()
    row = cases(blobs["holdout.jsonl"])[0]
    request = prepare_judge_input(collection, frozen, blobs, "holdout", manifest, row.case_id,
                                  "synthetic-material@1", {"evidence": "合成材料"})
    item = collection["cases"][0]
    assert request.message == row.message and request.expected == row.expected
    assert request.target_result == item["final"]["result"]
    assert request.target_result_hash == item["ledger_evidence"]["result_hash"]
    assert request.reference_material == {"evidence": "合成材料"}


def test_prepare_judge_rejects_mismatched_public_answer_even_with_matching_event_hashes():
    frozen, blobs, manifest, collection = report_case()
    row = cases(blobs["holdout.jsonl"])[0]
    item = collection["cases"][0]
    delta = next(event for event in item["events"] if event["type"] == "answer.delta")
    delta["payload"]["delta"] = "与最终结果不同的公开正文"
    item["ledger_evidence"]["event_hashes"] = [canonical_hash(event) for event in item["events"]]
    with pytest.raises(ValueError):
        prepare_judge_input(collection, frozen, blobs, "holdout", manifest, row.case_id,
                            "synthetic-material@1", {"evidence": "合成材料"})


def test_prepare_judge_input_validates_authorized_material_snapshot_binding():
    _, _, original, _ = report_case()
    updated = original.model_copy(update={"profiles": JsonSnapshot.capture({"resource_authority_ref": "knowledge-read@1"})})
    frozen, blobs, manifest, collection = report_case(updated)
    row = cases(blobs["holdout.jsonl"])[0]
    ledger = collection["cases"][0]["ledger_evidence"]
    material = {"evidence": "合成材料"}
    source = {"version": "evaluation-judge-material@1", "run_id": ledger["run_id"],
        "context_id": ledger["context_id"], "dataset_hash": frozen.dataset_hash,
        "manifest_hash": manifest.content_hash(), "input_hash": row.input_hash(),
        "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"],
        "source_ref": "knowledge-read@1", "material": material, "material_hash": canonical_hash(material)}
    request = prepare_judge_input(collection, frozen, blobs, "holdout", manifest, row.case_id,
                                  "knowledge-read@1", source)
    assert request.reference_material == material
    source["run_id"] = "different"
    with pytest.raises(ValueError, match="参考材料"):
        prepare_judge_input(collection, frozen, blobs, "holdout", manifest, row.case_id,
                            "knowledge-read@1", source)


@pytest.mark.parametrize("mutation", ["case", "result", "version", "events", "policy", "oversized"])
def test_prepare_judge_rejects_unbound_or_unusable_inputs(mutation):
    frozen, blobs, manifest, collection = report_case()
    row = cases(blobs["holdout.jsonl"])[0]
    case_id = row.case_id
    material = {"evidence": "合成材料"}
    if mutation == "case":
        case_id = "missing-case"
    elif mutation == "result":
        collection["cases"][0]["final"]["result"]["answer"] = "被替换"
    elif mutation == "version":
        collection["cases"][0]["accepted"]["manifest_hash"] = "0" * 64
    elif mutation == "events":
        collection["cases"][0]["events"][-1]["seq"] = 99
    elif mutation == "policy":
        collection["gate_policy_version"] = "other@1"
    else:
        material = {"evidence": "x" * 20000}
    with pytest.raises(ValueError):
        prepare_judge_input(collection, frozen, blobs, "holdout", manifest, case_id,
                            "synthetic-material@1", material)
