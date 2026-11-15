"""M5-06 指标、覆盖率、证据绑定及小样本口径定义。"""

import json
from copy import deepcopy

import pytest

from harness.contracts.evaluation import MetricPolicy
from harness.contracts.evaluation_metrics import CostObservation
from harness.contracts.hashing import canonical_hash
from harness.evaluation.assets import cases, freeze_assets
from harness.evaluation.metrics import distribution, ranking_scores
from harness.evaluation.reports import build_report
from harness.evaluation.thresholds import cost_coverage, formal_thresholds
from tests.unit.test_evaluation_reports import report_case


def test_fixed_slots_dedup_and_binary_relevance():
    scores = ranking_scores({"a", "b"}, ("other", "a", "a"))
    assert scores["precision_at_5"] == 0.2
    assert scores["recall_at_5"] == 0.5
    assert scores["mrr_at_5"] == 0.5
    assert scores["ndcg_at_5"] == pytest.approx(0.3868528)


def test_nearest_rank_counts_and_empty_latency():
    assert distribution([]) == {"n": 0, "p50": None, "p95": None, "p99": None, "small_sample": False}
    assert distribution([20, 10, 30]) == {"n": 3, "p50": 20, "p95": 30, "p99": 30, "small_sample": True}


@pytest.mark.parametrize("bound", [True, False])
def test_report_partial_cost_and_retrieval_only_use_bound_observations(bound):
    frozen, blobs, manifest, collection = report_case()
    row = collection["cases"][0]
    ledger = row["ledger_evidence"]
    row["metric_observation"] = {
        "source_ref": "synthetic-observer", "case_id": row["case_id"], "run_id": ledger["run_id"],
        "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
        "result_hash": ledger["result_hash"] if bound else "0" * 64, "event_seq": ledger["event_seq"],
        "ranked_chunk_aliases": ["c02", "c02"], "costs": {
            "model": {"state": "known", "amount": "0.03", "currency": "CNY", "price_version": "test@1"},
            "tool": {"state": "unknown"},
        },
    }
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    breakdown = report["metric_breakdown"]
    assert report["report_hash"] == canonical_hash({k: v for k, v in report.items() if k != "report_hash"})
    assert breakdown["retrieval"]["recall_at_5"] == {
        "mean": 1 if bound else None, "n": int(bound), "applicable": 2,
        "missing": 2 - int(bound), "not_applicable": 2,
    }
    assert breakdown["cost"]["batch_total_by_currency"] is None
    assert breakdown["cost"]["known_subtotal_by_currency"] is None
    assert breakdown["cost"]["reported_subtotal_by_currency"] == ({"CNY": "0.03"} if bound else None)
    assert breakdown["cost"]["categories"]["tool"]["unknown"] == int(bound)
    assert breakdown["cost"]["coverage"] is None
    assert breakdown["cost"]["source_trust"] == "unverified"
    assert breakdown["memory"]["key_fact_retained"]["not_applicable"] == 4
    assert breakdown["capacity"]["percentiles_ms"]["total_ms"]["n"] == 4
    assert breakdown["capacity"]["percentiles_ms"]["total_ms"]["small_sample"]
    assert breakdown["threshold_outcome"] == "not_evaluated"
    assert report["release_eligible"] is False


def metric_summary():
    score = {"mean": 0.8, "n": 2, "applicable": 2, "missing": 0, "not_applicable": 0}
    return {"retrieval": {"recall_at_5": dict(score)},
            "quality": {"dimensions": {"support": dict(score)}},
            "cost": {"expected_cases": 2, "fully_covered_cases": 2, "coverage": 1.0,
                     "source_trust": "verified"}}


def test_final_result_conflict_excludes_metric_observation():
    frozen, blobs, manifest, collection = report_case()
    item = collection["cases"][0]
    ledger = item["ledger_evidence"]
    item["metric_observation"] = {
        "source_ref": "synthetic-observer@1", "case_id": item["case_id"], "run_id": ledger["run_id"],
        "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
        "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"],
        "ranked_chunk_aliases": ["c02"],
    }
    # 保留公开答案文本，只篡改最终结果的其他字段。
    item["final"]["result"]["answer_kind"] = "answer"
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["metric_breakdown"]["cases"][0]["evidence"] == "missing_or_unbound"
    assert report["metric_breakdown"]["retrieval"]["recall_at_5"]["n"] == 0


def test_thresholds_require_frozen_confirmation_and_complete_scores():
    policy = MetricPolicy(version="test@1", metrics=("recall_at_5", "answer_quality", "cost_coverage"),
                          thresholds={"recall_at_5": 0.7, "answer_quality": 0.7, "cost_coverage": 1.0},
                          thresholds_confirmed=True)
    summary = metric_summary()
    assert formal_thresholds(policy, summary).outcome == "passed"
    assert formal_thresholds(policy.model_copy(update={"thresholds_confirmed": False}), summary).outcome == "incomplete"
    assert formal_thresholds(policy.model_copy(update={"thresholds": {"recall_at_5": 0.7}}), summary).outcome == "incomplete"
    summary["quality"]["dimensions"]["support"]["missing"] = 1
    assert formal_thresholds(policy, summary).outcome == "incomplete"


def test_threshold_failure_wins_over_another_missing_metric():
    policy = MetricPolicy(version="test@1", metrics=("recall_at_5", "answer_quality"),
                          thresholds={"recall_at_5": 0.9, "answer_quality": 0.7}, thresholds_confirmed=True)
    summary = metric_summary()
    summary["quality"]["dimensions"] = {}
    result = formal_thresholds(policy, summary)
    assert result.outcome == "failed"
    assert "threshold_not_met:recall_at_5" in result.reasons
    assert "metric_evidence_incomplete:answer_quality" in result.reasons


def test_cost_coverage_requires_all_cases_and_does_not_convert_unknown_to_zero():
    summary = metric_summary()
    assert cost_coverage(summary).outcome == "passed"
    summary["cost"]["source_trust"] = "unverified"
    assert cost_coverage(summary).outcome == "incomplete"
    summary["cost"]["source_trust"] = "verified"
    summary["cost"]["fully_covered_cases"] = 1
    summary["cost"]["coverage"] = 0.5
    assert cost_coverage(summary).outcome == "incomplete"
    policy = MetricPolicy(version="test@1", metrics=("cost_coverage",),
                          thresholds={"cost_coverage": 0.5}, thresholds_confirmed=True)
    assert formal_thresholds(policy, summary).outcome == "incomplete"


def test_unknown_metric_or_out_of_range_score_cannot_pass():
    policy = MetricPolicy(version="test@1", metrics=("unknown_metric",),
                          thresholds={"unknown_metric": 0.5}, thresholds_confirmed=True)
    assert formal_thresholds(policy, metric_summary()).outcome == "incomplete"
    policy = MetricPolicy(version="test@1", metrics=("recall_at_5",),
                          thresholds={"recall_at_5": 0.5}, thresholds_confirmed=True)
    summary = metric_summary()
    summary["retrieval"]["recall_at_5"]["mean"] = 1.1
    assert formal_thresholds(policy, summary).outcome == "incomplete"
    assert formal_thresholds(policy, metric_summary(), judge_required=True).outcome == "incomplete"


def test_absolute_report_replays_bound_thresholds_and_cost_coverage():
    _, blobs, manifest, collection = report_case()
    blobs = dict(blobs)
    gate = json.loads(blobs["gates.json"])
    gate["required_evidence"] += ["formal_thresholds", "cost_coverage"]
    blobs["gates.json"] = json.dumps(gate).encode()
    blobs["metrics.json"] = json.dumps({
        "version": "synthetic-metrics@1", "metrics": ["recall_at_5", "cost_coverage"],
        "thresholds": {"recall_at_5": 1.0, "cost_coverage": 1.0}, "thresholds_confirmed": True,
    }).encode()
    frozen = freeze_assets(blobs)
    collection["freeze_hash"] = frozen.content_hash()
    rows = {row.case_id: row for row in cases(blobs["holdout.jsonl"])}
    for item in collection["cases"]:
        ledger = item["ledger_evidence"]
        ledger.update({"model_steps": 1, "reserved_tokens": 15,
            "model_calls": [{"call_id": "call", "state": "succeeded", "price_version_ref": "price@1",
                             "input_tokens": 10, "reserved_output_tokens": 5, "remaining_total_tokens": 100}]})
        item["external_observation"]["freeze_hash"] = frozen.content_hash()
        item["metric_observation"] = {
            "source_ref": "synthetic-observer@1", "case_id": item["case_id"], "run_id": ledger["run_id"],
            "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
            "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"],
            "ranked_chunk_aliases": rows[item["case_id"]].expected["relevant_aliases"],
            "costs": {"model": {"state": "known", "amount": "0", "currency": "CNY",
                                "price_version": "price@1", "covered_call_ids": ["call"]},
                      **{name: {"state": "not_billable", "covered_call_ids": []}
                         for name in ("tool", "retrieval", "judge")}},
        }
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["absolute_gates"]["outcome"] == "incomplete"
    assert report["metric_breakdown"]["threshold_outcome"] == "incomplete"
    assert report["metric_breakdown"]["cost"]["inventory_matched_cases"] == len(collection["cases"])
    assert report["metric_breakdown"]["cost"]["batch_total_by_currency"] is None
    changed = deepcopy(collection)
    changed["cases"][0]["metric_observation"]["costs"]["model"] = {"state": "unknown"}
    report = build_report(changed, frozen, blobs, "holdout", manifest)
    assert report["absolute_gates"]["outcome"] == "incomplete"
    assert report["metric_breakdown"]["cost"]["batch_total_by_currency"] is None
    changed = deepcopy(collection)
    changed["cases"][0]["metric_observation"]["ranked_chunk_aliases"] = ["other"]
    assert build_report(changed, frozen, blobs, "holdout", manifest)["outcome"] == "failed"


def test_cost_inventory_rejects_missing_calls_and_unproven_free_claims():
    frozen, blobs, manifest, collection = report_case()
    item = collection["cases"][0]
    ledger = item["ledger_evidence"]
    ledger["model_calls"] = [
        {"call_id": call_id, "state": "succeeded", "price_version_ref": "price@1",
         "input_tokens": 10, "reserved_output_tokens": 5, "remaining_total_tokens": 100}
        for call_id in ("call", "model-call-2")
    ]
    ledger.update({"model_steps": 2, "reserved_tokens": 30})
    costs = {name: {"state": "not_billable", "covered_call_ids": []}
             for name in ("tool", "retrieval", "judge")}
    costs["model"] = {"state": "known", "amount": "0.1", "currency": "CNY",
                      "price_version": "price@1", "covered_call_ids": ["call"]}
    item["metric_observation"] = {
        "source_ref": "synthetic-observer@1", "case_id": item["case_id"], "run_id": ledger["run_id"],
        "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
        "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"], "costs": costs,
    }
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["metric_breakdown"]["cost"]["inventory_matched_cases"] == 0
    costs["model"]["covered_call_ids"] = ["call", "model-call-2"]
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["metric_breakdown"]["cost"]["inventory_matched_cases"] == 1
    assert report["metric_breakdown"]["cost"]["coverage"] is None
    assert report["metric_breakdown"]["cost"]["batch_total_by_currency"] is None
    costs["model"]["price_version"] = "other-price@1"
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["metric_breakdown"]["cost"]["inventory_matched_cases"] == 0
    costs["model"]["price_version"] = "price@1"
    missing_ledger = deepcopy(collection)
    missing_case = missing_ledger["cases"][0]
    missing_case["ledger_evidence"]["model_calls"] = [missing_case["ledger_evidence"]["model_calls"][1]]
    missing_case["ledger_evidence"].update({"model_steps": 1, "reserved_tokens": 15})
    missing_case["metric_observation"]["costs"]["model"]["covered_call_ids"] = ["model-call-2"]
    report = build_report(missing_ledger, frozen, blobs, "holdout", manifest)
    assert report["metric_breakdown"]["cost"]["inventory_matched_cases"] == 0
    costs["model"] = {"state": "not_billable", "covered_call_ids": ["call", "model-call-2"]}
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["metric_breakdown"]["cost"]["inventory_matched_cases"] == 0


def test_cost_observation_rejects_duplicate_call_identities():
    with pytest.raises(ValueError, match="调用身份不能重复"):
        CostObservation(state="not_billable", covered_call_ids=("call", "call"))
