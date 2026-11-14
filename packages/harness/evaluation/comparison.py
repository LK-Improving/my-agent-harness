"""比较同口径、完整观测的评测分项。"""

import math
from decimal import Decimal, InvalidOperation
from typing import Any


def _score_change(current: object, baseline: object) -> dict[str, object] | None:
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        return None
    count = current.get("applicable")
    if (type(count) is not int or count <= 0 or baseline.get("applicable") != count
            or current.get("n") != count or baseline.get("n") != count
            or current.get("missing") != 0 or baseline.get("missing") != 0):
        return None
    current_mean, baseline_mean = current.get("mean"), baseline.get("mean")
    if (type(current_mean) not in (int, float) or type(baseline_mean) not in (int, float)
            or not math.isfinite(current_mean) or not math.isfinite(baseline_mean)):
        return None
    return {"baseline": baseline_mean, "candidate": current_mean,
            "delta": current_mean - baseline_mean, "n": count}


def compare_metrics(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object] | None:
    changes: dict[str, object] = {}
    for section, names in (
        ("retrieval", ("precision_at_5", "recall_at_5", "mrr_at_5", "ndcg_at_5")),
        ("memory", ("key_fact_retained", "erroneous_deletion", "correction_followed", "compression_savings")),
    ):
        present, previous = current.get(section), baseline.get(section)
        if not isinstance(present, dict) or not isinstance(previous, dict):
            continue
        values = {name: change for name in names
                  if (change := _score_change(present.get(name), previous.get(name))) is not None}
        if values:
            changes[section] = values
    present_quality, previous_quality = current.get("quality"), baseline.get("quality")
    if isinstance(present_quality, dict) and isinstance(previous_quality, dict):
        present_dimensions = present_quality.get("dimensions")
        previous_dimensions = previous_quality.get("dimensions")
        if isinstance(present_dimensions, dict) and isinstance(previous_dimensions, dict):
            quality: dict[str, Any] = {}
            for name in sorted(present_dimensions.keys() & previous_dimensions.keys()):
                change = _score_change(present_dimensions[name], previous_dimensions[name])
                if change is not None:
                    quality[name] = change
            if quality:
                changes["quality"] = quality
    present_cost, previous_cost = current.get("cost"), baseline.get("cost")
    if isinstance(present_cost, dict) and isinstance(previous_cost, dict):
        present_total = present_cost.get("batch_total_by_currency")
        previous_total = previous_cost.get("batch_total_by_currency")
        expected = present_cost.get("expected_cases")
        if (isinstance(present_total, dict) and isinstance(previous_total, dict)
                and present_total.keys() == previous_total.keys()
                and type(expected) is int and expected > 0
                and previous_cost.get("expected_cases") == expected
                and present_cost.get("fully_covered_cases") == expected
                and previous_cost.get("fully_covered_cases") == expected):
            cost: dict[str, object] = {}
            for currency in present_total:
                try:
                    present_amount = Decimal(present_total[currency])
                    previous_amount = Decimal(previous_total[currency])
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if (present_amount.is_finite() and previous_amount.is_finite()
                        and present_amount >= 0 and previous_amount >= 0):
                    cost[currency] = {"baseline": str(previous_amount), "candidate": str(present_amount),
                                      "delta": str(present_amount - previous_amount)}
            if cost:
                changes["cost_by_currency"] = cost
    return changes or None
