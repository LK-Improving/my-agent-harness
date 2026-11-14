"""Judge 顺序采集与中断边界的合成测试定义；不代表真实依赖验收。"""

import json
from copy import deepcopy

import pytest

from harness.contracts.evaluation import CollectionFault
from harness.contracts.evaluation_gates import JudgeInput, JudgeModelResponse, RunGateEvidence
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import RunView
from harness.evaluation.assets import freeze_assets
from harness.evaluation.judge import prepare_judge_input
from harness.evaluation.judge_collector import collect_judges
from harness.evaluation.observations import merge_observations
from tests.unit.test_evaluation_judge_evidence import governed_judge_case
from tests.unit.test_evaluation_reports import report_case


def judge_batch_case():
    policy, receipt, _, _, row, target, _, proof = governed_judge_case()
    _, blobs, target, collection = report_case(target)
    blobs["judge.json"] = json.dumps(policy.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
    frozen = freeze_assets(blobs)
    collection["freeze_hash"] = frozen.content_hash()
    request = prepare_judge_input(collection, frozen, blobs, "holdout", target, row.case_id,
                                  "synthetic-material@1", {"text": "合成资料"})
    proof = deepcopy(proof)
    proof["input"] = request.model_dump(mode="json")
    answer = canonical_json({"version": "evaluation-judge-output@1", "case_id": request.case_id,
        "target_run_id": request.target_run_id, "target_result_hash": request.target_result_hash,
        "freeze_hash": frozen.content_hash(), "scores": receipt["scores"]})
    model_response = canonical_json({"answer": answer, "answer_kind": "answer", "task_outcome": "achieved"})
    proof["model_response"] = model_response
    proof["final"]["result"]["answer"] = answer
    digest = canonical_hash(proof["final"]["result"])
    for event in proof["events"]:
        if event["type"] == "answer.delta":
            event["payload"]["delta"] = answer
        elif event["type"] == "run.completed":
            event["payload"]["result_hash"] = digest
    ledger = proof["ledger"]
    ledger["request_hash"] = canonical_hash({"message": canonical_json(request.model_dump(mode="json")),
                                               "attachment_versions": [], "business_options": {}})
    ledger["result_hash"] = digest
    ledger["model_calls"][0]["response_content_hash"] = canonical_hash(model_response)
    ledger["event_hashes"] = [canonical_hash(event) for event in proof["events"]]
    return frozen, blobs, target, collection, ReleaseManifest.model_validate(proof["manifest"]), request, proof


class JudgeAPI:
    def __init__(self, proof):
        self.proof = proof
        self.calls = []
        self.wrong_identity = False
        self.unknown_create = False
        self.response_denied = False

    async def create_context(self, request):
        self.calls.append("context")
        context = EvaluationContext.model_validate(self.proof["context"])
        if self.wrong_identity:
            context = context.model_copy(update={"subject_id": "other"})
        return context

    async def create_session(self, request):
        self.calls.append("session")
        return self.proof["accepted"]["session_id"]

    async def create_run(self, session_id, request):
        self.calls.append("run")
        if self.unknown_create:
            raise CollectionFault("CREATE_OUTCOME_UNKNOWN")
        assert request.message == canonical_json(self.proof["input"])
        return RunView.model_validate(self.proof["accepted"])

    async def events(self, run_id):
        for event in self.proof["events"]:
            yield RuntimeEvent.model_validate(event)

    async def get_run(self, run_id):
        return RunView.model_validate(self.proof["final"])

    async def evaluation_evidence(self, run_id):
        return RunGateEvidence.model_validate(self.proof["ledger"])

    async def judge_model_response(self, run_id, call_id):
        if self.response_denied:
            raise PermissionError("synthetic sensitive error")
        content = self.proof["model_response"]
        return JudgeModelResponse(run_id=run_id, model_call_id=call_id,
                                  content=content, content_hash=canonical_hash(content))


async def test_judge_batch_collects_bound_score_and_mergeable_execution_evidence():
    frozen, blobs, target, collection, judge, request, proof = judge_batch_case()
    api = JudgeAPI(proof)
    journal = []
    result, bundle = await collect_judges(api, frozen, blobs, "holdout", collection, target, judge,
                                          (request,), "batch-judge", journal.append)
    assert api.calls == ["context", "session", "run"]
    assert result["outcome"] == "passed" and bundle is not None
    merged = merge_observations(collection, frozen, blobs, "holdout", target, bundle)
    assert merged["cases"][0]["judge_receipt"]["model_call_id"] == "judge-call"
    assert merged["cases"][0]["judge_execution"]["model_response"] == proof["model_response"]


async def test_judge_preflight_rejects_changed_input_before_any_api_creation():
    frozen, blobs, target, collection, judge, request, proof = judge_batch_case()
    api = JudgeAPI(proof)
    bad = JudgeInput.model_validate({**request.model_dump(mode="json"), "expected": {"other": True}})
    with pytest.raises(ValueError, match="输入"):
        await collect_judges(api, frozen, blobs, "holdout", collection, target, judge,
                             (bad,), "batch-judge", lambda record: None)
    assert api.calls == []


@pytest.mark.parametrize("failure", ["wrong_identity", "unknown_create", "response_denied"])
async def test_judge_interruption_preserves_incomplete_without_repeated_creation(failure):
    frozen, blobs, target, collection, judge, request, proof = judge_batch_case()
    api = JudgeAPI(proof)
    setattr(api, failure, True)
    journal = []
    result, bundle = await collect_judges(api, frozen, blobs, "holdout", collection, target, judge,
                                          (request,), "batch-judge", journal.append)
    assert result["outcome"] == "incomplete" and bundle is None
    assert api.calls.count("context") == 1 and api.calls.count("run") <= 1
    if failure == "wrong_identity":
        assert api.calls == ["context"]
    assert "synthetic sensitive error" not in str(journal)
