"""M5-04 报告hash/首次绝对门禁/基线比较定义；全部为合成输入。"""

import json
from copy import deepcopy
from pathlib import Path

from harness.adapters.evaluation_files import read_freeze
from harness.contracts.hashing import canonical_hash
from harness.contracts.specs import KnowledgeBinding
from harness.evaluation.assets import cases, freeze_assets
from harness.evaluation.reports import BaselineInput, build_report
from tests.unit.test_evaluation_versions import synthetic_version, version_case

ROOT = Path(__file__).resolve().parents[2]


def report_case(manifest=None):
    _, initial, _, template = version_case()
    manifest = manifest or initial
    _, blobs = read_freeze(ROOT / "evals/m5")
    # 独立合成策略，仅验证报告协议；不改仓库冻结资产或生产质量阈值。
    blobs = dict(blobs)
    judge = json.loads(blobs["judge.json"])
    judge["required"] = False
    blobs["judge.json"] = json.dumps(judge).encode()
    policy = json.loads(blobs["gates.json"])
    policy["hard_failures"] = ["unresolved_unknown", "manifest_mismatches"]
    policy["required_evidence"] = ["all_cases", "api_persisted_sse", "accepted_started_terminal_manifest"]
    blobs["gates.json"] = json.dumps(policy).encode()
    frozen = freeze_assets(blobs)
    records = []
    for index, row in enumerate(cases(blobs["holdout.jsonl"])):
        item = deepcopy(template["cases"][0])
        run_id, sid = f"run-{index}", f"session-{index}"
        item.update({"case_id": row.case_id, "expected": row.expected})
        version = synthetic_version(manifest).model_dump(mode="json")
        for event in item["events"]:
            event["run_id"] = run_id
            if event["type"] == "run.version_observed":
                event["payload"]["observation"] = deepcopy(version)
            elif event["type"] in {"run.started", "run.failed"}:
                event["payload"]["observed_manifest_hash"] = manifest.content_hash()
                event["payload"]["observed_environment_profile_hash"] = manifest.environment_profile_hash
        item["events"][0]["payload"].update({"release_id": manifest.spec.release_id,
            "manifest_hash": manifest.content_hash(), "environment_profile_hash": manifest.environment_profile_hash})
        answer = {"answer": "合成夹具", "answer_kind": "abstention", "task_outcome": "not_achieved"}
        digest = canonical_hash(answer)
        last = item["events"][-1]
        last["type"] = "run.completed"
        last["payload"].update({"committed_draft_id": "draft", "commit_id": "commit", "result_hash": digest})
        item["accepted"].update({"run_id": run_id, "session_id": sid, "release_id": manifest.spec.release_id,
            "manifest_hash": manifest.content_hash(), "environment_profile_hash": manifest.environment_profile_hash})
        item["final"].update({**item["accepted"], "status": "succeeded", "result": answer})
        item["ledger_evidence"].update({"run_id": run_id, "session_id": sid, "request_hash": row.input_hash(),
            "manifest_hash": manifest.content_hash(), "environment_profile_hash": manifest.environment_profile_hash,
            "status": "succeeded", "result_hash": digest,
            "validation_checks": {"citation_integrity": True}, "validation_verdict": "pass",
            "event_hashes": [canonical_hash(event) for event in item["events"]]})
        item["external_observation"].update({"run_id": run_id, "freeze_hash": frozen.content_hash(), "result_hash": digest})
        records.append(item)
    collection = {**template, "cases": records, "candidate_manifest": manifest.model_dump(mode="json"),
                  "freeze_hash": frozen.content_hash()}
    collection["evaluation_context"].update({"manifest_hash": manifest.content_hash(),
        "candidate_release_id": manifest.spec.release_id, "environment_profile_hash": manifest.environment_profile_hash})
    return frozen, blobs, manifest, collection


def test_first_baseline_null_still_runs_absolute_gates_and_hashes_report():
    frozen, blobs, manifest, collection = report_case()
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    assert report["baseline"] is None and report["absolute_gates"]["outcome"] == "passed"
    assert report["release_eligible"] is False
    assert report["report_hash"] == canonical_hash({k: v for k, v in report.items() if k != "report_hash"})
    collection["cases"].pop()
    failed = build_report(collection, frozen, blobs, "holdout", manifest)
    assert failed["outcome"] == "failed" and failed["baseline"] is None


def test_baseline_pass_label_cannot_override_corrupt_artifact_or_hard_failure():
    frozen, blobs, manifest, collection = report_case()
    saved = build_report(collection, frozen, blobs, "holdout", manifest)
    altered = deepcopy(saved)
    altered["report_hash"] = "0" * 64
    report = build_report(collection, frozen, blobs, "holdout", manifest,
                          BaselineInput(collection, frozen, blobs, altered))
    assert report["outcome"] == "incomplete"
    assert "baseline_report_hash_or_binding_invalid" in report["baseline"]["reasons"]
    broken = deepcopy(collection)
    broken["cases"][0]["ledger_evidence"]["tool_dispatches"] = 999
    report = build_report(broken, frozen, blobs, "holdout", manifest,
                          BaselineInput(collection, frozen, blobs, saved))
    assert report["outcome"] == "failed"


def test_snapshot_change_is_visible_but_missing_retrieval_ledger_keeps_comparison_incomplete():
    frozen, blobs, original, baseline_collection = report_case()
    saved = build_report(baseline_collection, frozen, blobs, "holdout", original)
    modified = original.model_copy(update={"spec": original.spec.model_copy(update={
        "release_id": "knowledge-change", "knowledge_bindings": (KnowledgeBinding(
            knowledge_base_id="test-kb", snapshot_id="snapshot-2", retrieval_profile_ref="profile@1"),)})})
    candidate_freeze, candidate_blobs, candidate, collection = report_case(modified)
    report = build_report(collection, candidate_freeze, candidate_blobs, "holdout", candidate,
                          BaselineInput(baseline_collection, frozen, blobs, saved))
    assert report["outcome"] == "incomplete" and report["baseline"]["outcome"] == "incomplete"
    assert report["absolute_gates"]["version_comparison"]["outcome"] == "passed"
    assert "absolute_gates_not_passed" in report["baseline"]["reasons"]
    assert "knowledge_bindings" in report["baseline"]["candidate_differences"]
    assert report["baseline"]["quality_change"] is None


def test_baseline_environment_difference_is_incomparable_but_not_per_run_mismatch():
    frozen, blobs, original, baseline_collection = report_case()
    saved = build_report(baseline_collection, frozen, blobs, "holdout", original)
    modified = original.model_copy(update={"environment_profile_hash": "f" * 64})
    current_freeze, current_blobs, candidate, collection = report_case(modified)
    report = build_report(collection, current_freeze, current_blobs, "holdout", candidate,
                          BaselineInput(baseline_collection, frozen, blobs, saved))
    assert report["absolute_gates"]["outcome"] == "passed"
    assert report["outcome"] == "incomplete" and "environment_incomparable" in report["baseline"]["reasons"]


def test_self_consistent_baseline_pass_label_is_rejected_when_raw_evidence_replay_fails():
    frozen, blobs, manifest, collection = report_case()
    saved = build_report(collection, frozen, blobs, "holdout", manifest)
    broken = deepcopy(collection)
    broken["cases"].pop()
    forged = deepcopy(saved)
    forged["collection_hash"] = canonical_hash(broken)
    forged["report_hash"] = canonical_hash({k: v for k, v in forged.items() if k != "report_hash"})
    report = build_report(collection, frozen, blobs, "holdout", manifest,
                          BaselineInput(broken, frozen, blobs, forged))
    assert report["absolute_gates"]["outcome"] == "passed" and report["outcome"] == "incomplete"
    assert "baseline_report_replay_mismatch" in report["baseline"]["reasons"]


def test_comparable_metrics_are_recomputed_from_original_collections():
    frozen, blobs, manifest, original = report_case()
    rows = {row.case_id: row for row in cases(blobs["holdout.jsonl"])}
    for item in original["cases"]:
        ledger = item["ledger_evidence"]
        item["metric_observation"] = {
            "source_ref": "synthetic-observer@1", "case_id": item["case_id"], "run_id": ledger["run_id"],
            "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
            "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"],
            "ranked_chunk_aliases": rows[item["case_id"]].expected["relevant_aliases"],
            "costs": {name: {"state": "not_billable"} for name in ("model", "tool", "retrieval", "judge")},
        }
    saved = build_report(original, frozen, blobs, "holdout", manifest)
    candidate = deepcopy(original)
    candidate["cases"][0]["metric_observation"]["ranked_chunk_aliases"] = ["other"]
    report = build_report(candidate, frozen, blobs, "holdout", manifest,
                          BaselineInput(original, frozen, blobs, saved))
    assert report["baseline"]["outcome"] == "comparable"
    assert report["baseline"]["metric_changes"]["retrieval"]["recall_at_5"] == {
        "baseline": 1.0, "candidate": 0.5, "delta": -0.5, "n": 2,
    }

    forged = deepcopy(saved)
    forged["metric_breakdown"]["retrieval"]["recall_at_5"]["mean"] = 0.0
    forged["report_hash"] = canonical_hash({key: value for key, value in forged.items() if key != "report_hash"})
    report = build_report(candidate, frozen, blobs, "holdout", manifest,
                          BaselineInput(original, frozen, blobs, forged))
    assert report["baseline"]["outcome"] == "incomplete"
    assert "baseline_metric_replay_mismatch" in report["baseline"]["reasons"]
