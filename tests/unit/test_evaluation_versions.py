"""M5-04 版本比较反例定义；合成观察不代表真实工作区/供应商证据。"""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from harness.contracts.evaluation_versions import ExecutionVersion
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext
from harness.evaluation.versions import compare_versions
from tests.unit.test_evaluation_gates import fixture


def synthetic_version(manifest):
    return ExecutionVersion(observed_at=datetime.now(UTC), observed_manifest_hash=manifest.content_hash(),
        source_tree_hash=manifest.source_tree_hash, dependency_lock_hash=manifest.dependency_lock_hash,
        prompt_hash=manifest.prompt_hash, tool_schema_hashes_hash=canonical_hash(manifest.tool_schema_hashes.as_object()),
        model_profile_hash=canonical_hash(manifest.model_profile.as_object()),
        profiles_hash=canonical_hash(manifest.profiles.as_object()),
        environment_profile_hash=manifest.environment_profile_hash, configuration_consistent=True)


def version_case():
    frozen, manifest, row, item, _ = fixture()
    context = EvaluationContext(context_id="context-1", tenant_id="test", subject_id="requester", application_id="app",
        agent_id=manifest.spec.agent_id, candidate_release_id=manifest.spec.release_id,
        manifest_hash=manifest.content_hash(), dataset_hash=frozen.dataset_hash, gate_policy_version="m5-gates@1",
        environment_profile_hash=manifest.environment_profile_hash, resource_scopes=frozenset({"synthetic"}),
        expires_at=datetime.now(UTC) + timedelta(hours=1))
    version = synthetic_version(manifest).model_dump(mode="json")
    base = {"schema_version": 1, "run_id": "run-1", "timestamp": version["observed_at"]}
    events = [item["events"][0],
        {**base, "seq": 2, "type": "run.version_observed", "payload": {
            "phase": "start", "execution_identity": "worker-1", "observation": deepcopy(version), "error_code": None}},
        {**base, "seq": 3, "type": "run.started", "payload": {"execution_identity": "worker-1",
            "version_evidence_seq": 2, "observed_manifest_hash": manifest.content_hash(),
            "observed_environment_profile_hash": manifest.environment_profile_hash}},
        {**base, "seq": 4, "type": "run.version_observed", "payload": {
            "phase": "terminal", "execution_identity": "worker-1", "observation": deepcopy(version), "error_code": None}},
        {**base, "seq": 5, "type": "run.failed", "payload": {"code": "EXPECTED_SYNTHETIC_FAILURE",
            "version_evidence_seq": 4, "observed_manifest_hash": manifest.content_hash(),
            "observed_environment_profile_hash": manifest.environment_profile_hash}}]
    item["events"] = events
    item["accepted"].update({"release_id": manifest.spec.release_id, "manifest_hash": manifest.content_hash(),
                             "environment_profile_hash": manifest.environment_profile_hash})
    item["final"].update(item["accepted"])
    item["ledger_evidence"].update({"event_seq": 5, "event_hashes": [canonical_hash(event) for event in events]})
    item["external_observation"]["event_seq"] = 5
    collection = {"candidate_manifest": manifest.model_dump(mode="json"), "evaluation_context": context.model_dump(mode="json"),
        "freeze_hash": frozen.content_hash(), "dataset_hash": frozen.dataset_hash, "gate_policy_version": "m5-gates@1",
        "cases": [item]}
    return frozen, manifest, row, collection


def compare(collection):
    frozen, manifest, row, _ = version_case()
    return compare_versions(collection, frozen, (row,), manifest)


def test_all_recorded_actual_versions_must_match():
    _, _, _, collection = version_case()
    assert compare(collection)["outcome"] == "passed"


@pytest.mark.parametrize("field", ["source_tree_hash", "dependency_lock_hash", "prompt_hash",
    "tool_schema_hashes_hash", "model_profile_hash", "profiles_hash", "environment_profile_hash"])
def test_actual_inventory_drift_is_failed_even_if_manifest_label_is_unchanged(field):
    _, _, _, collection = version_case()
    collection["cases"][0]["events"][3]["payload"]["observation"][field] = "f" * 64
    assert compare(collection)["outcome"] == "failed"


@pytest.mark.parametrize("where", ["accepted", "final", "ledger_evidence"])
def test_client_candidate_a_cannot_use_b_version_receipts(where):
    _, _, _, collection = version_case()
    collection["cases"][0][where]["manifest_hash"] = "f" * 64
    assert compare(collection)["outcome"] == "failed"


def test_missing_version_fields_are_incomplete_without_copying_candidate():
    _, _, _, collection = version_case()
    for event in collection["cases"][0]["events"]:
        if event["type"] == "run.version_observed":
            event["payload"]["observation"] = None
            event["payload"]["error_code"] = "Unavailable"
    item = collection["cases"][0]
    item["ledger_evidence"]["event_hashes"] = [canonical_hash(event) for event in item["events"]]
    report = compare(collection)
    assert report["outcome"] == "incomplete"
    assert "version_observation_unavailable" in report["cases"][0]["reasons"]


def test_terminal_cannot_reuse_start_proof_or_another_worker():
    for mutation in ("reference", "worker"):
        _, _, _, collection = version_case()
        item = collection["cases"][0]
        if mutation == "reference":
            item["events"][-1]["payload"]["version_evidence_seq"] = 2
        else:
            item["events"][3]["payload"]["execution_identity"] = "another-worker"
        item["ledger_evidence"]["event_hashes"] = [canonical_hash(event) for event in item["events"]]
        assert compare(collection)["outcome"] == "failed"


def test_first_started_drift_is_not_hidden_by_later_good_version():
    _, _, _, collection = version_case()
    item = collection["cases"][0]
    item["events"][1]["payload"]["observation"]["configuration_consistent"] = False
    assert compare(collection)["outcome"] == "failed"


def test_resume_must_validate_every_started_not_only_last_owner():
    _, _, _, collection = version_case()
    item = collection["cases"][0]
    original = item["events"]
    resumed_proof = deepcopy(original[1])
    resumed_proof["seq"] = 4
    resumed_proof["payload"]["execution_identity"] = "worker-2"
    resumed_start = deepcopy(original[2])
    resumed_start.update({"seq": 5})
    resumed_start["payload"].update({"execution_identity": "worker-2", "version_evidence_seq": 4})
    terminal_proof = deepcopy(original[3])
    terminal_proof["seq"] = 6
    terminal_proof["payload"]["execution_identity"] = "worker-2"
    terminal = deepcopy(original[4])
    terminal["seq"] = 7
    terminal["payload"]["version_evidence_seq"] = 6
    item["events"] = original[:3] + [resumed_proof, resumed_start, terminal_proof, terminal]
    item["events"][1]["payload"]["observation"]["source_tree_hash"] = "f" * 64
    item["ledger_evidence"].update({"event_seq": 7,
        "event_hashes": [canonical_hash(event) for event in item["events"]]})
    assert compare(collection)["outcome"] == "failed"


def test_declared_git_revision_without_live_git_evidence_is_not_treated_as_verified():
    frozen, manifest, row, collection = version_case()
    revised = manifest.model_copy(update={"source_revision": "claimed-only"})
    collection["candidate_manifest"] = revised.model_dump(mode="json")
    report = compare_versions(collection, frozen, (row,), revised)
    assert report["outcome"] != "passed"
