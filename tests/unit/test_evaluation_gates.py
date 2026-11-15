"""M5-03 合成证据反例定义；不代表真实PG/供应商或业务验收。"""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness.adapters.evaluation_files import read_freeze
from harness.contracts.evaluation import GatePolicy, JudgePolicy
from harness.contracts.evaluation_gates import RunGateEvidence
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext
from harness.evaluation.assets import cases, decode
from harness.evaluation.gates import evaluate_batch, evaluate_case, judge_check
from tests.unit.test_evaluation_collector import candidate

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    frozen, blobs = read_freeze(ROOT / "evals/m5")
    manifest = candidate()
    row = cases(blobs["holdout.jsonl"])[0].model_copy(update={"expected": {"run_statuses": ["failed"], "max_writes": 1}})
    now = datetime.now(UTC)
    events = [RuntimeEvent(run_id="run-1", seq=1, type="run.accepted", timestamp=now,
        payload={"release_id": manifest.spec.release_id, "manifest_hash": manifest.content_hash(),
                 "environment_profile_hash": manifest.environment_profile_hash}),
        RuntimeEvent(run_id="run-1", seq=2, type="run.failed", timestamp=now,
                     payload={"code": "EXPECTED_SYNTHETIC_FAILURE"})]
    ledger = RunGateEvidence(run_id="run-1", session_id="session-1", context_id="context-1",
        dataset_hash=frozen.dataset_hash, gate_policy_version="m5-gates@1", request_hash=row.input_hash(),
        manifest_hash=manifest.content_hash(),
        environment_profile_hash=manifest.environment_profile_hash, snapshot_at=now, status="failed",
        requester_id="requester", event_seq=2, run_scopes=("synthetic",), context_scopes=("synthetic",),
        operations=(), model_calls=(), model_steps=0, reserved_tokens=0, tool_dispatches=0,
        execution_used_ms=0, max_input_tokens=100, max_output_tokens=20, max_total_tokens=120,
        max_execution_ms=1000, reserved_extension_tokens=0, events=(),
        event_hashes=tuple(canonical_hash(event.model_dump(mode="json")) for event in events))
    observed = {"case_id": row.case_id, "expected": row.expected,
        "accepted": {"run_id": "run-1", "session_id": "session-1"},
        "events": [event.model_dump(mode="json") for event in events],
        "final": {"run_id": "run-1", "status": "failed", "result": None},
        "ledger_evidence": ledger.model_dump(mode="json"),
        "external_observation": {"run_id": "run-1", "freeze_hash": frozen.content_hash(), "result_hash": None,
            "source_ref": "test-independent-observer@1", "event_seq": 2,
            "unauthorized_deliveries": 0, "side_effect_counts": {}, "coverage": ["authorization_delivery"]}}
    judge = JudgePolicy(version="synthetic-judge@1", required=False, dimensions=("quality",))
    return frozen, manifest, row, observed, judge


def context_for(frozen, manifest):
    return EvaluationContext(context_id="context-1", tenant_id="test", subject_id="requester",
        application_id="app", agent_id=manifest.spec.agent_id, candidate_release_id=manifest.spec.release_id,
        manifest_hash=manifest.content_hash(), dataset_hash=frozen.dataset_hash, gate_policy_version="m5-gates@1",
        environment_profile_hash=manifest.environment_profile_hash, resource_scopes=frozenset({"synthetic"}),
        expires_at=datetime.now(UTC) + timedelta(hours=1))


def result(observed, judge=None):
    frozen, manifest, row, _, default_judge = fixture()
    return evaluate_case(row, observed, frozen, manifest, judge or default_judge,
                         target_context=context_for(frozen, manifest))


def operation(effect="read"):
    return {"operation_id": "op-1", "tool_name": "get_work_order" if effect == "read" else "create_work_order",
            "tool_hash": "4" * 64, "effect": effect, "state": "succeeded", "resource_scopes": ["synthetic"],
            "args_hash": "a" * 64, "receipt_args_hash": "a" * 64, "recorded_attempts": 1, "max_attempts": 1}


def test_complete_synthetic_evidence_can_pass_only_deterministic_subset():
    _, _, _, observed, _ = fixture()
    assert result(observed).outcome == "passed"


@pytest.mark.parametrize("kind", ["scope", "tool", "receipt", "unknown", "counter", "tokens", "attempt", "delivery"])
def test_each_hard_violation_wins_over_missing_judge(kind):
    _, _, _, observed, _ = fixture()
    op = operation()
    observed["ledger_evidence"]["operations"] = [op]
    observed["ledger_evidence"]["tool_dispatches"] = 1
    if kind == "scope":
        op["resource_scopes"] = ["kb:another-tenant"]
    elif kind == "tool":
        op["tool_name"] = "unregistered_tool"
    elif kind == "receipt":
        op["receipt_args_hash"] = "b" * 64
    elif kind == "unknown":
        op["state"] = "unknown"
    elif kind == "counter":
        observed["ledger_evidence"]["tool_dispatches"] = 0
    elif kind == "tokens":
        observed["ledger_evidence"]["reserved_tokens"] = 200
    elif kind == "attempt":
        op["effect"] = "write"
    else:
        observed["external_observation"]["unauthorized_deliveries"] = 1
    judge = JudgePolicy(version="required@1", dimensions=("quality",))
    assert result(observed, judge).outcome == "failed"


def test_no_external_delivery_observation_is_incomplete_not_zero():
    _, _, _, observed, _ = fixture()
    observed.pop("external_observation")
    assert result(observed).outcome == "incomplete"


@pytest.mark.parametrize("mutation", ["scopes", "subject", "context_id"])
def test_scope_gate_binds_run_to_evaluation_context(mutation):
    _, _, _, observed, _ = fixture()
    ledger = observed["ledger_evidence"]
    if mutation == "scopes":
        ledger["run_scopes"] = ["other"]
        ledger["context_scopes"] = ["other"]
    elif mutation == "subject":
        ledger["requester_id"] = "other"
    else:
        ledger["context_id"] = "other"
    report = result(observed)
    scope = next(item for item in report.checks if item.gate == "scope_isolation")
    assert scope.outcome == "failed" and "evaluation_context_binding_mismatch" in scope.reasons


def test_missing_evaluation_context_cannot_pass_scope_gate():
    frozen, manifest, row, observed, judge = fixture()
    report = evaluate_case(row, observed, frozen, manifest, judge)
    assert next(item for item in report.checks if item.gate == "scope_isolation").outcome == "incomplete"


@pytest.mark.parametrize("missing_policy", [False, True])
def test_reported_model_input_over_reservation_is_a_hard_budget_failure(missing_policy):
    _, _, _, observed, _ = fixture()
    observed["ledger_evidence"].update({
        "model_steps": 1, "reserved_tokens": 30,
        "model_calls": [{"call_id": "call-1", "state": "succeeded", "input_tokens": 20,
                         "reported_input_tokens": 21, "reserved_output_tokens": 10,
                         "output_tokens": 5, "remaining_total_tokens": 100}],
    })
    if missing_policy:
        observed["ledger_evidence"]["max_output_tokens"] = None
    report = result(observed)
    assert next(item for item in report.checks if item.gate == "budget").outcome == "failed"
    assert report.outcome == "failed"


@pytest.mark.parametrize("mutation", ["missing_delta", "different_answer", "other_attempt", "other_draft"])
def test_target_answer_projection_must_match_persisted_final_result(mutation):
    from tests.unit.test_evaluation_reports import report_case

    frozen, blobs, manifest, collection = report_case()
    row = cases(blobs["holdout.jsonl"])[0]
    observed = deepcopy(collection["cases"][0])
    delta = next(event for event in observed["events"] if event["type"] == "answer.delta")
    if mutation == "missing_delta":
        delta.update({"type": "future.missing", "payload": {}})
    elif mutation == "different_answer":
        delta["payload"]["delta"] = "不同的公开回答"
    elif mutation == "other_attempt":
        delta["payload"]["model_attempt_id"] = "other"
    else:
        observed["events"][-1]["payload"]["committed_draft_id"] = "other"
    observed["ledger_evidence"]["event_hashes"] = [canonical_hash(event) for event in observed["events"]]
    context = EvaluationContext.model_validate(collection["evaluation_context"])
    judge = JudgePolicy(version="synthetic-judge@1", required=False, dimensions=("quality",))
    report = evaluate_case(row, observed, frozen, manifest, judge, target_context=context)
    checks = {item.gate: item.outcome for item in report.checks}
    assert checks["persisted_sse"] == "passed"
    assert checks["answer_projection"] == "failed" and report.outcome == "failed"


def test_valid_write_approval_must_precede_each_attempt_and_external_count_is_required():
    _, _, _, observed, _ = fixture()
    now = datetime.fromisoformat(observed["events"][0]["timestamp"].replace("Z", "+00:00"))
    op = operation("write")
    op.update({"approval_id": "approval-1", "binding_hash": "b" * 64,
        "action_key": "c" * 64,
        "approval_binding_hash": "b" * 64, "approval_state": "approved", "approver": "independent",
        "approval_expires_at": (now + timedelta(minutes=1)).isoformat(),
        "attempts": [{"attempt_id": "attempt-1", "ordinal": 1, "state": "succeeded", "dispatched_at": (now + timedelta(seconds=2)).isoformat()}]})
    approved = RuntimeEvent(run_id="run-1", seq=2, type="approval.decided", timestamp=now + timedelta(seconds=1),
                            payload={"approval_id": "approval-1", "decision": "approved"})
    observed["events"][-1]["seq"] = 3
    observed["events"][-1]["timestamp"] = (now + timedelta(seconds=3)).isoformat().replace("+00:00", "Z")
    observed["events"].insert(1, approved.model_dump(mode="json"))
    observed["ledger_evidence"]["event_seq"] = 3
    observed["ledger_evidence"]["event_hashes"] = [canonical_hash(event) for event in observed["events"]]
    observed["external_observation"]["event_seq"] = 3
    observed["ledger_evidence"].update({"operations": [op], "tool_dispatches": 1,
                                        "events": [approved.model_dump(mode="json")]})
    assert result(observed).outcome == "incomplete"
    observed["external_observation"].update({"coverage": ["authorization_delivery", "external_writes"],
                                             "side_effect_counts": {"op-1": 1}})
    assert result(observed).outcome == "passed"
    observed["external_observation"]["side_effect_counts"]["op-1"] = 2
    assert result(observed).outcome == "failed"
    observed["external_observation"]["side_effect_counts"]["op-1"] = 1
    op["approver"] = "requester"
    assert result(observed).outcome == "failed"


@pytest.mark.parametrize("mutation", ["gap", "missing_terminal", "different_body", "other_run"])
def test_sse_missing_or_changed_evidence_fails(mutation):
    _, _, _, observed, _ = fixture()
    if mutation == "gap":
        observed["events"][-1]["seq"] = 3
    elif mutation == "missing_terminal":
        observed["events"].pop()
    elif mutation == "different_body":
        observed["events"][-1]["payload"]["code"] = "ALTERED"
    else:
        observed["events"][-1]["run_id"] = "other"
    assert result(observed).outcome == "failed"


def test_required_judge_missing_timeout_or_invalid_never_passes():
    _, _, _, observed, _ = fixture()
    policy = JudgePolicy(version="judge@1", implementation="judge-v1", prompt_sha256="a" * 64,
                         model_binding_ref="fixed-model@1", dimensions=("quality",))
    for value in (None, {}, {"state": "timeout"}, {"scores": {"quality": float("nan")}}):
        observed["judge_receipt"] = value
        assert result(observed, policy).outcome == "incomplete"


def test_missing_ledger_does_not_pass_and_false_citation_checks_fail():
    _, _, _, observed, _ = fixture()
    ledger = observed.pop("ledger_evidence")
    assert result(observed).outcome == "incomplete"
    observed["ledger_evidence"] = ledger
    ledger["status"] = "succeeded"
    ledger["validation_checks"] = {"citation_integrity": False}
    ledger["validation_verdict"] = "pass"
    observed["final"]["status"] = "succeeded"
    assert result(observed).outcome == "failed"


def test_batch_uses_frozen_ids_instead_of_claimed_pass_or_expected_list():
    frozen, manifest, _, observed, judge = fixture()
    _, blobs = read_freeze(ROOT / "evals/m5")
    rows = cases(blobs["holdout.jsonl"])
    policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
    collection = {"freeze_hash": frozen.content_hash(), "dataset_hash": frozen.dataset_hash,
        "gate_policy_version": policy.version, "outcome": "passed", "expected_case_ids": [rows[0].case_id],
        "cases": [observed]}
    report = evaluate_batch(collection, frozen, rows, manifest, policy, judge, split="holdout")
    assert report["outcome"] == "failed" and report["release_eligible"] is False
    assert any("required_cases_missing" in item["reasons"] for item in report["checks"])
    collection["cases"] = []
    assert evaluate_batch(collection, frozen, rows, manifest, policy, judge, split="holdout")["outcome"] == "not_run"


def test_duplicate_unexpected_and_run_creation_unknown_are_not_ignored():
    frozen, manifest, _, observed, judge = fixture()
    _, blobs = read_freeze(ROOT / "evals/m5")
    rows = cases(blobs["holdout.jsonl"])
    policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
    base = {"freeze_hash": frozen.content_hash(), "dataset_hash": frozen.dataset_hash, "gate_policy_version": policy.version}
    for records in ([observed, deepcopy(observed)], [{"case_id": "not-frozen"}], [{"case_id": {}}]):
        assert evaluate_batch({**base, "cases": records}, frozen, rows, manifest, policy, judge,
                              split="holdout")["outcome"] == "failed"
    unknown = {**base, "cases": [{"case_id": rows[0].case_id, "session_id": "known-session"}],
               "collection_error_code": "CREATE_OUTCOME_UNKNOWN"}
    assert evaluate_batch(unknown, frozen, rows, manifest, policy, judge, split="holdout")["outcome"] == "failed"


def test_actual_input_hash_cannot_be_replaced_by_case_label():
    _, _, _, observed, _ = fixture()
    observed["ledger_evidence"]["request_hash"] = "0" * 64
    report = result(observed)
    assert report.outcome == "failed"
    assert any("actual_input_mismatch" in item.reasons for item in report.checks)


def test_judge_receipt_requires_all_fixed_bindings_and_complete_dimensions():
    frozen, _, _, observed, _ = fixture()
    ledger = RunGateEvidence.model_validate(observed["ledger_evidence"]).model_copy(update={"result_hash": "b" * 64})
    policy = JudgePolicy(version="judge@1", implementation="judge-v1", prompt_sha256="a" * 64,
                         model_binding_ref="fixed-model@1", dimensions=("quality",))
    receipt = {"run_id": ledger.run_id, "freeze_hash": frozen.content_hash(), "result_hash": ledger.result_hash,
        "policy_version": policy.version, "implementation": policy.implementation, "prompt_sha256": policy.prompt_sha256,
        "model_binding_ref": policy.model_binding_ref, "model_call_id": "judge-call", "receipt_ref": "ledger/judge-call",
        "state": "succeeded", "scores": {"quality": 0.8}}
    assert judge_check(policy, receipt, frozen, ledger).outcome == "incomplete"
    for change in ({"state": "timeout"}, {"result_hash": "c" * 64}, {"scores": {}},
                   {"model_binding_ref": "other@1"}, {"scores": {"quality": float("inf")}},
                   {"scores": {"quality": 1.1}}, {"scores": {"quality": -0.1}}):
        assert judge_check(policy, {**receipt, **change}, frozen, ledger).outcome == "incomplete"


def test_complete_deterministic_subset_keeps_unimplemented_policy_evidence_deferred():
    frozen, manifest, _, template, judge = fixture()
    _, blobs = read_freeze(ROOT / "evals/m5")
    rows = cases(blobs["holdout.jsonl"])
    records = []
    for index, row in enumerate(rows):
        item = deepcopy(template)
        run_id = f"run-{index}"
        item.update({"case_id": row.case_id, "expected": row.expected})
        item["accepted"]["run_id"] = run_id
        answer = {"answer": "合成门禁夹具", "answer_kind": "abstention", "task_outcome": "not_achieved"}
        digest = canonical_hash(answer)
        item["final"].update({"run_id": run_id, "status": "succeeded", "result": answer})
        item["events"][0]["run_id"] = run_id
        item["events"][-1].update({"run_id": run_id, "type": "run.completed", "payload": {
            "committed_draft_id": "draft", "commit_id": "commit", "result_hash": digest}})
        base = {"schema_version": 1, "run_id": run_id, "timestamp": item["events"][0]["timestamp"]}
        item["events"][-1]["seq"] = 4
        item["events"][1:1] = [
            {**base, "seq": 2, "type": "answer.started", "payload": {
                "draft_id": "draft", "producer": "model", "model_call_id": "call", "model_attempt_id": "attempt"}},
            {**base, "seq": 3, "type": "answer.delta", "payload": {
                "draft_id": "draft", "model_attempt_id": "attempt", "delta": answer["answer"]}},
        ]
        item["ledger_evidence"].update({"run_id": run_id, "request_hash": row.input_hash(), "status": "succeeded",
            "result_hash": digest, "validation_checks": {"valid_evidence": True}, "validation_verdict": "pass",
            "event_seq": len(item["events"]),
            "event_hashes": [canonical_hash(event) for event in item["events"]]})
        item["external_observation"].update({"run_id": run_id, "result_hash": digest,
            "manifest_hash": manifest.content_hash(), "credential_disclosures": 0, "event_seq": len(item["events"]),
            "coverage": ["authorization_delivery", "credential_exposure"]})
        records.append(item)
    policy = GatePolicy.model_validate(decode(blobs["gates.json"]))
    collection = {"freeze_hash": frozen.content_hash(), "dataset_hash": frozen.dataset_hash,
                  "gate_policy_version": policy.version, "cases": records,
                  "evaluation_context": context_for(frozen, manifest).model_dump(mode="json")}
    report = evaluate_batch(collection, frozen, rows, manifest, policy, judge, split="holdout")
    assert report["deterministic_outcome"] == "passed" and report["outcome"] == "incomplete"
    assert report["release_eligible"] is False
    assert "accepted_started_terminal_manifest" in report["deferred_evidence"]
    assert "hard_failure:credential_disclosures" not in report["deferred_evidence"]


def test_credential_gate_requires_bound_independent_coverage_and_fails_on_disclosure():
    frozen, manifest, row, observed, judge = fixture()
    options = {"credential_observation_required": True, "target_context": context_for(frozen, manifest)}
    assert evaluate_case(row, observed, frozen, manifest, judge, **options).outcome == "incomplete"
    external = observed["external_observation"]
    external.update({"manifest_hash": manifest.content_hash(), "credential_disclosures": 0,
                     "coverage": ["authorization_delivery", "credential_exposure"]})
    assert evaluate_case(row, observed, frozen, manifest, judge, **options).outcome == "passed"
    external["credential_disclosures"] = 1
    assert evaluate_case(row, observed, frozen, manifest, judge, **options).outcome == "failed"
    external["credential_disclosures"] = 0
    external["manifest_hash"] = "f" * 64
    assert evaluate_case(row, observed, frozen, manifest, judge, **options).outcome == "incomplete"
