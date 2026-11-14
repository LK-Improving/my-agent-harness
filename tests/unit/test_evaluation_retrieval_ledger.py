"""检索子账本绑定、预算和未知状态的合成反例。"""

import pytest

from harness.contracts.evaluation_gates import RetrievalChildEvidence, RunGateEvidence
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.specs import KnowledgeBinding
from harness.evaluation.retrieval_ledger import retrieval_checks
from tests.retrieval_pipeline_fixtures import retrieval_ticket
from tests.unit.test_evaluation_versions import version_case


def retrieval_evidence_case():
    ticket, configuration, _ = retrieval_ticket()
    _, base, _, collection = version_case()
    profile = ticket.profile
    spec = base.spec.model_copy(update={"knowledge_bindings": (KnowledgeBinding(
        knowledge_base_id=ticket.pin.knowledge_base_id, snapshot_id=ticket.pin.snapshot_id,
        retrieval_profile_ref=profile.ref),)})
    profiles = base.profiles.as_object()
    profiles["retrieval"] = configuration.model_dump(mode="json")
    manifest = ReleaseManifest.model_validate({**base.model_dump(mode="json"),
        "spec": spec.model_dump(mode="json"), "profiles": profiles})
    context = EvaluationContext.model_validate(collection["evaluation_context"])
    scope = ticket.scope.model_copy(update={"tenant_id": context.tenant_id, "subject_id": context.subject_id,
        "application_id": context.application_id, "owner_id": "run-1", "manifest_hash": manifest.content_hash()})
    call = {"call_id": "query-embedding", "attempt_id": "attempt-1", "request_fingerprint": "a" * 64,
        "binding_hash": scope.embedding_binding_hash, "task_type": "embedding", "input_bytes": 100,
        "state": "succeeded", "result_call_id": "query-embedding", "result_attempt_id": "attempt-1",
        "result_fingerprint": "a" * 64, "result_task_type": "embedding", "usage": {"input_tokens": 20}}
    child = RetrievalChildEvidence(retrieval_id=scope.context_id, knowledge_base_id=scope.knowledge_base_id,
        state="succeeded", ticket_scope=scope, ticket_binding_valid=True, profile_limits=profile.model_limits,
        parent_reserved_tokens=profile.parent_token_reservation(), context_scope=scope, context_binding_valid=True,
        context_limits=profile.model_limits, calls_used=1, input_bytes_used=100, calls=(call,))
    ledger = RunGateEvidence.model_validate(collection["cases"][0]["ledger_evidence"]).model_copy(update={
        "status": "succeeded", "manifest_hash": manifest.content_hash(),
        "reserved_extension_tokens": child.parent_reserved_tokens, "retrieval_children": (child,)})
    assert scope.embedding_binding_hash == canonical_hash(configuration.models.as_object()["embedding"])
    return ledger, manifest, context


def test_child_ledger_checks_state_but_does_not_claim_input_size_or_cost():
    ledger, manifest, context = retrieval_evidence_case()
    bad, missing, unknown, unknown_missing = retrieval_checks(ledger, manifest, context)
    assert not bad and missing == ["retrieval_input_size_unverified"]
    assert not unknown and not unknown_missing
    assert ledger.retrieval_children[0].calls[0].usage.output_tokens is None


@pytest.mark.parametrize("mutation", ["scope", "budget", "counter", "binding", "result", "reserved", "unknown"])
def test_child_ledger_drift_or_unknown_blocks_gate(mutation):
    ledger, manifest, context = retrieval_evidence_case()
    data = ledger.model_dump(mode="json")
    child = data["retrieval_children"][0]
    call = child["calls"][0]
    if mutation == "scope":
        child["ticket_scope"]["subject_id"] = "other"
    elif mutation == "budget":
        child["parent_reserved_tokens"] += 1
    elif mutation == "counter":
        child["calls_used"] += 1
    elif mutation == "binding":
        call["binding_hash"] = "f" * 64
    elif mutation == "result":
        call["result_attempt_id"] = "other"
    elif mutation == "reserved":
        call["state"] = "reserved"
    else:
        call.update({"state": "failed", "dispatch_status": "unknown", "billing_status": "unknown"})
    bad, missing, unknown, unknown_missing = retrieval_checks(RunGateEvidence.model_validate(data), manifest, context)
    assert bad or unknown


def test_missing_child_ledger_is_incomplete_not_zero_or_passed():
    ledger, manifest, context = retrieval_evidence_case()
    ledger = ledger.model_copy(update={"retrieval_children": None})
    bad, missing, unknown, unknown_missing = retrieval_checks(ledger, manifest, context)
    assert not bad and not unknown and missing and unknown_missing
