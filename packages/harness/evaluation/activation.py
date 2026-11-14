"""用部署侧固定的冻结材料复核发布报告。"""

from harness.contracts.activation import ActivationPolicy
from harness.contracts.evaluation import EvaluationFreeze, GatePolicy, JudgePolicy
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import RuntimeFault
from harness.evaluation.assets import decode
from harness.evaluation.reports import absolute


def verify_report(
    policy: ActivationPolicy,
    candidate: ReleaseManifest,
    frozen: EvaluationFreeze,
    blobs: dict[str, bytes],
    collection: dict[str, object],
    report: dict[str, object],
) -> EvaluationContext:
    body = {key: value for key, value in report.items() if key != "report_hash"}
    gate = GatePolicy.model_validate(decode(blobs["gates.json"]))
    context = EvaluationContext.model_validate(collection.get("evaluation_context"))
    if (
        report.get("version") != "evaluation-report@1"
        or report.get("report_hash") != policy.report_hash
        or canonical_hash(body) != policy.report_hash
        or report.get("collection_hash") != canonical_hash(collection)
        or report.get("split") != "holdout"
        or report.get("candidate_manifest_hash") != candidate.content_hash()
        or report.get("freeze_hash") != policy.freeze_hash
        or frozen.content_hash() != policy.freeze_hash
        or report.get("dataset_hash") != policy.dataset_hash
        or frozen.dataset_hash != policy.dataset_hash
        or gate.version != policy.gate_policy_version
        or context.tenant_id != policy.tenant_id
        or context.application_id != policy.application_id
        or context.agent_id != policy.agent_id
        or context.candidate_release_id != candidate.spec.release_id
        or context.manifest_hash != candidate.content_hash()
        or context.dataset_hash != policy.dataset_hash
        or context.gate_policy_version != policy.gate_policy_version
        or context.environment_profile_hash != policy.environment_profile_hash
        or candidate.environment_profile_hash != policy.environment_profile_hash
    ):
        raise RuntimeFault("ACTIVATION_REPORT_BINDING_MISMATCH")

    gates = absolute(collection, frozen, blobs, "holdout", candidate)
    # 拒绝通过删减冻结策略来放行当前尚缺的正式证据。
    if (
        not {"formal_thresholds", "cost_coverage"} <= set(gate.required_evidence)
        or "credential_disclosures" not in gate.hard_failures
        or not JudgePolicy.model_validate(decode(blobs["judge.json"])).required
        or report.get("outcome") != "passed"
        or report.get("release_eligible") is not True
        or gates["outcome"] != "passed"
        or canonical_hash(report.get("absolute_gates")) != canonical_hash(gates)
    ):
        raise RuntimeFault("ACTIVATION_GATES_NOT_PASSED")
    return context
