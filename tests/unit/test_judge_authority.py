"""Judge 冻结策略和输入边界的合成测试定义。"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from harness.adapters.postgres.judge_authority import PostgresJudgeAuthority
from harness.contracts.hashing import RunRequest, canonical_json
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from tests.unit.test_evaluation_judge_collector import judge_batch_case


def authority_case():
    frozen, blobs, _, _, judge, request, _ = judge_batch_case()

    async def read_material(conn, actor, run, session):
        return {"text": "合成资料"}

    authority = PostgresJudgeAuthority(None, frozen, blobs, read_material, "synthetic-material@1")
    actor = Principal(tenant_id="test", subject_id="requester", application_id="app",
                      roles=frozenset({"evaluator"}), resource_scopes=frozenset({"synthetic"}))
    run_request = RunRequest(request_id="judge-test", expected_revision=0,
                             message=canonical_json(request.model_dump(mode="json")))
    run = ClaimedRun(run_id="judge-run", session_id="judge-session", principal=actor,
                     request=run_request, manifest=judge, execution_deadline=datetime.now(UTC) + timedelta(minutes=5),
                     history=JsonSnapshot.capture([]))
    return authority, run, request


def test_judge_authority_requires_frozen_exact_tool_free_manifest():
    authority, run, request = authority_case()
    assert authority.ready_for(run.manifest)
    assert authority.input_for(run) == request
    changed = run.manifest.model_copy(update={"prompt_hash": "f" * 64})
    assert not authority.ready_for(changed)
    with pytest.raises(RuntimeFault, match="JUDGE_POLICY_OR_INPUT_MISMATCH"):
        authority.input_for(run.model_copy(update={"manifest": changed}))


@pytest.mark.parametrize("mutation", ["noncanonical", "freeze", "expected", "dimensions"])
def test_judge_authority_rejects_drift_before_provider_dispatch(mutation):
    authority, run, request = authority_case()
    body = request.model_dump(mode="json")
    if mutation == "freeze":
        body["freeze_hash"] = "f" * 64
    elif mutation == "expected":
        body["expected"] = {"injected": True}
    elif mutation == "dimensions":
        body["dimensions"] = ["other"]
    message = json.dumps(body, ensure_ascii=False, indent=2) if mutation == "noncanonical" else canonical_json(body)
    changed = run.model_copy(update={"request": run.request.model_copy(update={"message": message})})
    with pytest.raises(RuntimeFault, match="JUDGE_POLICY_OR_INPUT_MISMATCH"):
        authority.input_for(changed)
