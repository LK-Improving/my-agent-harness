"""M5-05 精确发布绑定及策略降级反例。"""

import json

import pytest

from harness.contracts.activation import ActivationPolicy
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.evaluation.activation import verify_report
from harness.evaluation.assets import cases, freeze_assets
from harness.evaluation.reports import build_report
from tests.unit.test_evaluation_reports import report_case


@pytest.fixture
def materials():
    frozen, blobs, manifest, collection = report_case()
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    context = collection["evaluation_context"]
    policy = ActivationPolicy(
        tenant_id=context["tenant_id"], application_id=context["application_id"],
        agent_id=manifest.spec.agent_id, environment_profile_hash=manifest.environment_profile_hash,
        freeze_hash=frozen.content_hash(), dataset_hash=frozen.dataset_hash,
        gate_policy_version=collection["gate_policy_version"], report_hash=report["report_hash"],
    )
    return policy, manifest, frozen, blobs, collection, report


@pytest.mark.parametrize("changed", ["release", "prompt", "tool", "environment"])
def test_report_cannot_publish_changed_candidate(materials, changed):
    policy, manifest, frozen, blobs, collection, report = materials
    update = {
        "release": {"spec": manifest.spec.model_copy(update={"release_id": "different-release"})},
        "prompt": {"prompt_hash": "b" * 64},
        "tool": {"tool_schema_hashes": JsonSnapshot.capture({"new-tool@1": "b" * 64})},
        "environment": {"environment_profile_hash": "b" * 64},
    }
    with pytest.raises(RuntimeFault, match="ACTIVATION_REPORT_BINDING_MISMATCH"):
        verify_report(policy, manifest.model_copy(update=update[changed]), frozen, blobs, collection, report)


@pytest.mark.parametrize("field,value", [
    ("report_hash", "e" * 64), ("freeze_hash", "e" * 64), ("dataset_hash", "e" * 64),
    ("gate_policy_version", "other@1"), ("tenant_id", "other"), ("application_id", "other"),
])
def test_deployment_pins_are_mandatory(materials, field, value):
    policy, *remaining = materials
    with pytest.raises(RuntimeFault, match="ACTIVATION_REPORT_BINDING_MISMATCH"):
        verify_report(policy.model_copy(update={field: value}), *remaining)


def test_even_pinned_self_report_cannot_lower_gate_requirements(materials):
    policy, manifest, frozen, blobs, collection, report = materials
    # 夹具的绝对门禁可通过，但它刻意关闭了正式证据要求。
    assert report["outcome"] == "passed"
    report["release_eligible"] = True
    report["report_hash"] = canonical_hash({key: value for key, value in report.items() if key != "report_hash"})
    policy = policy.model_copy(update={"report_hash": report["report_hash"]})
    with pytest.raises(RuntimeFault, match="ACTIVATION_GATES_NOT_PASSED"):
        verify_report(policy, manifest, frozen, blobs, collection, report)


def test_collection_mutation_breaks_approved_report_binding(materials):
    policy, manifest, frozen, blobs, collection, report = materials
    collection["cases"].pop()
    with pytest.raises(RuntimeFault, match="ACTIVATION_REPORT_BINDING_MISMATCH"):
        verify_report(policy, manifest, frozen, blobs, collection, report)


def test_synthetic_holdout_cannot_claim_release_eligibility():
    _, blobs, manifest, collection = report_case()
    blobs = dict(blobs)
    judge = {"version": "synthetic-judge@1", "required": True, "implementation": "judge@1",
             "prompt_sha256": "a" * 64, "model_binding_ref": "model@1", "dimensions": ["quality"]}
    blobs["judge.json"] = json.dumps(judge).encode()
    gate = json.loads(blobs["gates.json"])
    gate["required_evidence"] += ["formal_thresholds", "cost_coverage", "required_judge"]
    gate["hard_failures"].append("credential_disclosures")
    blobs["gates.json"] = json.dumps(gate).encode()
    blobs["metrics.json"] = json.dumps({
        "version": "synthetic-metrics@1", "metrics": ["answer_quality", "cost_coverage"],
        "thresholds": {"answer_quality": 0.7, "cost_coverage": 1.0}, "thresholds_confirmed": True,
    }).encode()
    frozen = freeze_assets(blobs)
    collection["freeze_hash"] = frozen.content_hash()
    rows = {row.case_id: row for row in cases(blobs["holdout.jsonl"])}
    for item in collection["cases"]:
        ledger = item["ledger_evidence"]
        item["external_observation"].update({
            "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
            "credential_disclosures": 0, "coverage": ["authorization_delivery", "credential_exposure"],
        })
        item["judge_receipt"] = {
            "run_id": ledger["run_id"], "freeze_hash": frozen.content_hash(), "result_hash": ledger["result_hash"],
            "policy_version": judge["version"], "implementation": judge["implementation"],
            "prompt_sha256": judge["prompt_sha256"], "model_binding_ref": judge["model_binding_ref"],
            "model_call_id": "synthetic-call", "receipt_ref": "synthetic-receipt", "state": "succeeded",
            "scores": {"quality": 0.8},
        }
        item["metric_observation"] = {
            "source_ref": "synthetic-observer@1", "case_id": item["case_id"], "run_id": ledger["run_id"],
            "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
            "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"],
            "ranked_chunk_aliases": rows[item["case_id"]].expected["relevant_aliases"],
            "costs": {name: {"state": "not_billable"} for name in ("model", "tool", "retrieval", "judge")},
        }
    report = build_report(collection, frozen, blobs, "holdout", manifest)
    context = collection["evaluation_context"]
    policy = ActivationPolicy(
        tenant_id=context["tenant_id"], application_id=context["application_id"],
        agent_id=manifest.spec.agent_id, environment_profile_hash=manifest.environment_profile_hash,
        freeze_hash=frozen.content_hash(), dataset_hash=frozen.dataset_hash,
        gate_policy_version=collection["gate_policy_version"], report_hash=report["report_hash"],
    )
    assert report["outcome"] == "incomplete" and report["release_eligible"] is False
    with pytest.raises(RuntimeFault, match="ACTIVATION_GATES_NOT_PASSED"):
        verify_report(policy, manifest, frozen, blobs, collection, report)
    collection["cases"][0]["external_observation"]["credential_disclosures"] = 1
    failed = build_report(collection, frozen, blobs, "holdout", manifest)
    assert failed["outcome"] == "failed" and failed["release_eligible"] is False
