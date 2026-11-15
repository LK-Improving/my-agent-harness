"""按冻结指标策略判定质量阈值与费用覆盖。"""

import math
from typing import Any, cast

from harness.contracts.evaluation import MetricPolicy
from harness.contracts.evaluation_gates import GateCheck


def _measured_score(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    score = cast(dict[str, Any], value)
    mean = score.get("mean")
    applicable = score.get("applicable")
    if (
        type(applicable) is not int
        or applicable <= 0
        or score.get("n") != applicable
        or score.get("missing") != 0
        or type(mean) not in (int, float)
        or not math.isfinite(mean)
        or not 0 <= mean <= 1
    ):
        return None
    return float(mean)


def cost_coverage(breakdown: dict[str, object]) -> GateCheck:
    cost = breakdown.get("cost")
    if not isinstance(cost, dict):
        return GateCheck(gate="cost_coverage", outcome="incomplete", reasons=("cost_evidence_missing",))
    if cost.get("source_trust") != "verified":
        return GateCheck(gate="cost_coverage", outcome="incomplete", reasons=("cost_source_unverified",))
    expected = cost.get("expected_cases")
    if type(expected) is not int or expected <= 0 or cost.get("fully_covered_cases") != expected:
        return GateCheck(gate="cost_coverage", outcome="incomplete", reasons=("cost_evidence_incomplete",))
    return GateCheck(gate="cost_coverage", outcome="passed")


def formal_thresholds(
    policy: MetricPolicy, breakdown: dict[str, object], *, judge_required: bool = False,
) -> GateCheck:
    if not policy.thresholds_confirmed:
        return GateCheck(gate="formal_thresholds", outcome="incomplete", reasons=("thresholds_unconfirmed",))
    if len(set(policy.metrics)) != len(policy.metrics) or set(policy.metrics) != set(policy.thresholds):
        return GateCheck(gate="formal_thresholds", outcome="incomplete", reasons=("thresholds_incomplete",))
    if judge_required and "answer_quality" not in policy.metrics:
        return GateCheck(gate="formal_thresholds", outcome="incomplete", reasons=("judge_threshold_missing",))

    retrieval = breakdown.get("retrieval")
    quality = breakdown.get("quality")
    cost = breakdown.get("cost")
    failures: list[str] = []
    missing: list[str] = []
    for name in policy.metrics:
        threshold = policy.thresholds[name]
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            missing.append("threshold_invalid")
            continue
        values: list[float | None]
        if name in {"recall_at_5", "precision_at_5", "mrr_at_5", "ndcg_at_5"}:
            values = [_measured_score(retrieval.get(name))] if isinstance(retrieval, dict) else [None]
        elif name == "answer_quality":
            dimensions = quality.get("dimensions") if isinstance(quality, dict) else None
            values = [_measured_score(value) for value in dimensions.values()] if isinstance(dimensions, dict) else []
        elif name == "cost_coverage":
            covered = cost_coverage(breakdown).outcome == "passed"
            value = cost.get("coverage") if isinstance(cost, dict) else None
            values = [float(value) if covered and type(value) in (int, float) else None]
        else:
            missing.append("metric_unsupported")
            continue
        if not values or any(value is None for value in values):
            missing.append(f"metric_evidence_incomplete:{name}")
        elif any(value is not None and value < threshold for value in values):
            failures.append(f"threshold_not_met:{name}")
    return GateCheck(
        gate="formal_thresholds",
        outcome="failed" if failures else "incomplete" if missing else "passed",
        reasons=tuple(dict.fromkeys(failures + missing)),
    )
