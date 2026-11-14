"""独立观测合并与绑定反例；使用合成材料。"""

from copy import deepcopy

import pytest

from harness.contracts.evaluation_observations import ObservationBundle
from harness.evaluation.observations import merge_observations
from tests.unit.test_evaluation_reports import report_case


def observation_case():
    frozen, blobs, manifest, collection = report_case()
    item = collection["cases"][0]
    ledger = item["ledger_evidence"]
    observation = {
        "source_ref": "independent-observer@1", "case_id": item["case_id"], "run_id": ledger["run_id"],
        "freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
        "result_hash": ledger["result_hash"], "event_seq": ledger["event_seq"],
        "ranked_chunk_aliases": ["c02"],
        "costs": {"model": {"state": "unknown"}},
    }
    bundle = {"freeze_hash": frozen.content_hash(), "manifest_hash": manifest.content_hash(),
              "cases": [{"case_id": item["case_id"], "metric_observation": observation}]}
    return frozen, blobs, manifest, collection, bundle


def merge(frozen, blobs, manifest, collection, bundle):
    return merge_observations(collection, frozen, blobs, "holdout", manifest,
                              ObservationBundle.model_validate(bundle))


def test_merge_preserves_original_and_identical_replay_is_idempotent():
    frozen, blobs, manifest, collection, bundle = observation_case()
    original = deepcopy(collection)
    merged = merge(frozen, blobs, manifest, collection, bundle)
    assert collection == original
    assert merged["cases"][0]["metric_observation"]["source_ref"] == "independent-observer@1"
    assert merge(frozen, blobs, manifest, merged, bundle) == merged


@pytest.mark.parametrize("field,value", [
    ("run_id", "other-run"), ("freeze_hash", "0" * 64), ("manifest_hash", "0" * 64),
    ("result_hash", "0" * 64), ("event_seq", 999), ("case_id", "other-case"),
])
def test_metric_observation_binding_mismatch_is_rejected(field, value):
    frozen, blobs, manifest, collection, bundle = observation_case()
    bundle["cases"][0]["metric_observation"][field] = value
    with pytest.raises(ValueError, match="指标观测"):
        merge(frozen, blobs, manifest, collection, bundle)


def test_merge_rejects_conflicts_and_does_not_modify_the_collection():
    frozen, blobs, manifest, collection, bundle = observation_case()
    merged = merge(frozen, blobs, manifest, collection, bundle)
    before = deepcopy(merged)
    bundle["cases"][0]["metric_observation"]["ranked_chunk_aliases"] = ["different"]
    with pytest.raises(ValueError, match="不能覆盖"):
        merge(frozen, blobs, manifest, merged, bundle)
    assert merged == before


def test_merge_rejects_unbound_events_and_duplicate_cases():
    frozen, blobs, manifest, collection, bundle = observation_case()
    collection["cases"][0]["events"][-1]["payload"]["result_hash"] = "0" * 64
    with pytest.raises(ValueError, match="持久事件"):
        merge(frozen, blobs, manifest, collection, bundle)
    bundle["cases"].append(deepcopy(bundle["cases"][0]))
    with pytest.raises(ValueError, match="重复用例"):
        ObservationBundle.model_validate(bundle)


def test_judge_receipt_cannot_be_imported_when_judge_is_unconfigured():
    frozen, blobs, manifest, collection, bundle = observation_case()
    ledger = collection["cases"][0]["ledger_evidence"]
    bundle["cases"][0] = {
        "case_id": collection["cases"][0]["case_id"],
        "judge_receipt": {"run_id": ledger["run_id"], "freeze_hash": frozen.content_hash(),
            "result_hash": ledger["result_hash"], "policy_version": "test@1", "implementation": "test@1",
            "prompt_sha256": "a" * 64, "model_binding_ref": "test@1", "model_call_id": "call",
            "receipt_ref": "receipt", "state": "succeeded", "scores": {"quality": 0.8}},
    }
    with pytest.raises(ValueError, match="Judge 回执"):
        merge(frozen, blobs, manifest, collection, bundle)
