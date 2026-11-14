"""带规范hash的绝对门禁报告与显式基线比较；不激活发布。"""

from dataclasses import dataclass
from typing import Literal, cast

from harness.contracts.evaluation import EvaluationFreeze, GatePolicy, JudgePolicy, MetricPolicy
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.evaluation.assets import cases, decode
from harness.evaluation.comparison import compare_metrics
from harness.evaluation.gates import evaluate_batch
from harness.evaluation.metrics import summarize
from harness.evaluation.thresholds import cost_coverage, formal_thresholds


@dataclass(frozen=True)
class BaselineInput:
    collection: dict[str, object]
    frozen: EvaluationFreeze
    blobs: dict[str, bytes]
    report: dict[str, object]


def metric_breakdown(
    collection: dict[str, object], frozen: EvaluationFreeze, blobs: dict[str, bytes],
    split: Literal["development", "holdout"], candidate: ReleaseManifest, gates: dict[str, object],
) -> dict[str, object]:
    policy = MetricPolicy.model_validate(decode(blobs["metrics.json"]))
    judge = JudgePolicy.model_validate(decode(blobs["judge.json"]))
    result = summarize(collection, cases(blobs[f"{split}.jsonl"]), frozen, candidate, policy, judge, gates)
    if policy.thresholds_confirmed:
        result["threshold_outcome"] = formal_thresholds(policy, result, judge_required=judge.required).outcome
    return result


def absolute(collection: dict[str, object], frozen: EvaluationFreeze, blobs: dict[str, bytes],
             split: Literal["development", "holdout"], candidate: ReleaseManifest) -> dict[str, object]:
    rows = cases(blobs[f"{split}.jsonl"])
    policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
    judge = JudgePolicy.model_validate(decode(blobs["judge.json"]))
    metric_policy = MetricPolicy.model_validate(decode(blobs["metrics.json"]))
    result = evaluate_batch(collection, frozen, rows, candidate, policy, judge, split=split)
    required = set(policy.required_evidence) & {"formal_thresholds", "cost_coverage"}
    if not required:
        return result
    breakdown = summarize(collection, rows, frozen, candidate, metric_policy, judge, result)
    checks = []
    if "formal_thresholds" in required:
        checks.append(formal_thresholds(metric_policy, breakdown, judge_required=judge.required))
    if "cost_coverage" in required:
        checks.append(cost_coverage(breakdown))
    deferred = [item for item in cast(list[str], result["deferred_evidence"]) if item not in required]
    result["deferred_evidence"] = deferred
    result["metric_checks"] = [item.model_dump(mode="json") for item in checks]
    if result["outcome"] == "failed" or any(item.outcome == "failed" for item in checks):
        result["outcome"] = "failed"
    elif result["outcome"] != "not_run":
        version = result["version_comparison"]
        if (result["deterministic_outcome"] == "passed" and isinstance(version, dict)
                and version.get("outcome") == "passed" and not deferred
                and all(item.outcome == "passed" for item in checks)):
            result["outcome"] = "passed"
        else:
            result["outcome"] = "incomplete"
    return result


def build_report(collection: dict[str, object], frozen: EvaluationFreeze, blobs: dict[str, bytes],
                 split: Literal["development", "holdout"], candidate: ReleaseManifest,
                 baseline: BaselineInput | None = None) -> dict[str, object]:
    """baseline为空也完整重算绝对门禁，不信任collection或baseline自报passed。"""
    gates = absolute(collection, frozen, blobs, split, candidate)
    breakdown = metric_breakdown(collection, frozen, blobs, split, candidate, gates)
    comparison: dict[str, object] | None = None
    if baseline is not None:
        reasons: list[str] = []
        saved = baseline.report
        claimed = saved.get("report_hash")
        body = {key: value for key, value in saved.items() if key != "report_hash"}
        if (saved.get("version") != "evaluation-report@1" or claimed != canonical_hash(body)
                or saved.get("collection_hash") != canonical_hash(baseline.collection)
                or saved.get("freeze_hash") != baseline.frozen.content_hash()):
            reasons.append("baseline_report_hash_or_binding_invalid")
        baseline_gates = None
        baseline_candidate = None
        baseline_metrics = None
        try:
            baseline_candidate = ReleaseManifest.model_validate(baseline.collection.get("candidate_manifest"))
            baseline_gates = absolute(baseline.collection, baseline.frozen, baseline.blobs, split, baseline_candidate)
            baseline_metrics = metric_breakdown(baseline.collection, baseline.frozen, baseline.blobs,
                                                split, baseline_candidate, baseline_gates)
            if (canonical_hash(saved.get("absolute_gates")) != canonical_hash(baseline_gates)
                    or saved.get("outcome") != baseline_gates["outcome"]):
                reasons.append("baseline_report_replay_mismatch")
            if canonical_hash(saved.get("metric_breakdown")) != canonical_hash(baseline_metrics):
                reasons.append("baseline_metric_replay_mismatch")
        except (ValueError, TypeError, KeyError):
            reasons.append("baseline_evidence_invalid")
        if baseline.frozen.dataset_hash != frozen.dataset_hash or saved.get("split") != split:
            reasons.append("dataset_or_split_incomparable")
        candidate_files = {item.name: item.sha256 for item in frozen.files}
        baseline_files = {item.name: item.sha256 for item in baseline.frozen.files}
        if any(candidate_files.get(name) != baseline_files.get(name)
               for name in ("judge.json", "metrics.json", "gates.json")):
            reasons.append("evaluation_policy_incomparable")
        differences = []
        if baseline_candidate is not None:
            if candidate.spec.agent_id != baseline_candidate.spec.agent_id:
                reasons.append("agent_incomparable")
            if candidate.environment_profile_hash != baseline_candidate.environment_profile_hash:
                reasons.append("environment_incomparable")
            if candidate.dependency_lock_hash != baseline_candidate.dependency_lock_hash:
                reasons.append("dependency_lock_incomparable")
            for field in ("source_tree_hash", "prompt_hash", "tool_schema_hashes", "model_profile", "profiles"):
                if canonical_hash(candidate.model_dump(mode="json")[field]) != canonical_hash(
                        baseline_candidate.model_dump(mode="json")[field]):
                    differences.append(field)
            if candidate.spec.knowledge_bindings != baseline_candidate.spec.knowledge_bindings:
                differences.append("knowledge_bindings")
            if canonical_hash(candidate.spec.model_dump(mode="json", exclude={"release_id", "knowledge_bindings"})) != canonical_hash(
                    baseline_candidate.spec.model_dump(mode="json", exclude={"release_id", "knowledge_bindings"})):
                differences.append("agent_spec")
        try:
            current_context = EvaluationContext.model_validate(collection.get("evaluation_context"))
            baseline_context = EvaluationContext.model_validate(baseline.collection.get("evaluation_context"))
            if ((current_context.tenant_id, current_context.application_id, current_context.subject_id,
                 current_context.resource_scopes) != (baseline_context.tenant_id, baseline_context.application_id,
                 baseline_context.subject_id, baseline_context.resource_scopes)):
                reasons.append("evaluation_identity_or_scope_incomparable")
        except (ValueError, TypeError):
            reasons.append("evaluation_identity_missing")
        if baseline_gates is None or baseline_gates["outcome"] != "passed" or gates["outcome"] != "passed":
            reasons.append("absolute_gates_not_passed")
        comparison = {"baseline_report_hash": claimed, "outcome": "incomplete" if reasons else "comparable",
            "reasons": list(dict.fromkeys(reasons)), "candidate_differences": differences,
            "baseline_absolute_outcome": baseline_gates["outcome"] if baseline_gates else "incomplete",
            "quality_change": None, "metric_changes": None,
            "note": "分项差值只描述同口径观测，不推断变化原因"}
        if not reasons and baseline_metrics is not None:
            changes = compare_metrics(breakdown, baseline_metrics)
            comparison["metric_changes"] = changes
            if changes is not None:
                comparison["quality_change"] = changes.get("quality")
    outcome = gates["outcome"]
    if outcome not in {"failed", "not_run"} and comparison is not None and comparison["outcome"] != "comparable":
        outcome = "incomplete"
    report: dict[str, object] = {"version": "evaluation-report@1", "collection_hash": canonical_hash(collection),
        "freeze_hash": frozen.content_hash(), "dataset_hash": frozen.dataset_hash, "split": split,
        "candidate_manifest_hash": candidate.content_hash(), "absolute_gates": gates,
        "baseline": comparison, "outcome": outcome, "release_eligible": False}
    report["candidate_identity"] = {"source_revision": candidate.source_revision,
        "source_tree_hash": candidate.source_tree_hash, "dependency_lock_hash": candidate.dependency_lock_hash,
        "environment_profile_hash": candidate.environment_profile_hash, "prompt_hash": candidate.prompt_hash,
        "tool_schema_hashes_hash": canonical_hash(candidate.tool_schema_hashes.as_object()),
        "model_profile_hash": canonical_hash(candidate.model_profile.as_object()),
        "profiles_hash": canonical_hash(candidate.profiles.as_object())}
    report["frozen_files"] = [item.model_dump(mode="json") for item in frozen.files]
    report["metric_breakdown"] = breakdown
    report["report_hash"] = canonical_hash(report)
    return report
