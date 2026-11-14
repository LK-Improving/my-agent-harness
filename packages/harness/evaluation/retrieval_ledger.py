"""独立复核检索子账本及父预留；不推断费用或外部副作用。"""

from harness.contracts.evaluation_gates import RunGateEvidence
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.retrieval import RetrievalRelease


def retrieval_checks(
    ledger: RunGateEvidence, manifest: ReleaseManifest, context: EvaluationContext | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    budget_bad: list[str] = []
    budget_missing: list[str] = []
    unknown_bad: list[str] = []
    unknown_missing: list[str] = []
    children = ledger.retrieval_children
    if children is None:
        if ledger.reserved_extension_tokens or manifest.spec.knowledge_bindings or "retrieval_child_ledger" in ledger.missing:
            budget_missing.append("retrieval_child_ledger_missing")
            unknown_missing.append("retrieval_unknown_unverified")
        return budget_bad, budget_missing, unknown_bad, unknown_missing
    expected = {item.knowledge_base_id: item for item in manifest.spec.knowledge_bindings}
    if (len({item.retrieval_id for item in children}) != len(children)
            or len({item.knowledge_base_id for item in children}) != len(children)
            or (ledger.reserved_extension_tokens is not None
                and sum(item.parent_reserved_tokens for item in children) != ledger.reserved_extension_tokens)):
        budget_bad.append("retrieval_reservation_mismatch")
    if ledger.status == "succeeded" and set(item.knowledge_base_id for item in children) != set(expected):
        budget_missing.append("retrieval_knowledge_coverage_missing")
        unknown_missing.append("retrieval_unknown_unverified")
    if not children:
        return budget_bad, budget_missing, unknown_bad, unknown_missing
    try:
        release = RetrievalRelease.model_validate(manifest.profiles.as_object().get("retrieval"))
        models = release.models.as_object()
        embedding = canonical_hash(models["embedding"])
        rerank = canonical_hash(models["rerank"])
    except (TypeError, ValueError, KeyError):
        budget_missing.append("retrieval_release_profile_missing")
        unknown_missing.append("retrieval_binding_unverified")
        return budget_bad, budget_missing, unknown_bad, unknown_missing
    if context is None:
        budget_missing.append("retrieval_context_identity_missing")
        unknown_missing.append("retrieval_binding_unverified")
    for child in children:
        scope = child.ticket_scope
        binding = expected.get(child.knowledge_base_id)
        profile = next((item for item in release.profiles
                        if binding is not None and item.ref == binding.retrieval_profile_ref), None)
        pin = next((item for item in release.snapshots if item.knowledge_base_id == child.knowledge_base_id), None)
        if (not child.ticket_binding_valid or scope.context_id != child.retrieval_id
                or scope.owner_kind != "run" or scope.owner_id != ledger.run_id
                or scope.knowledge_base_id != child.knowledge_base_id
                or scope.manifest_hash != manifest.content_hash()
                or scope.subject_id != ledger.requester_id or scope.embedding_binding_hash != embedding
                or scope.rerank_binding_hash != rerank or binding is None or profile is None or pin is None
                or (binding is not None and pin is not None and binding.snapshot_id != pin.snapshot_id)
                or (profile is not None and child.profile_limits != profile.model_limits)
                or child.parent_reserved_tokens != child.profile_limits.max_input_bytes + 512 * child.profile_limits.max_calls
                or (context is not None and (scope.tenant_id, scope.subject_id, scope.application_id)
                    != (context.tenant_id, context.subject_id, context.application_id))):
            budget_bad.append("retrieval_identity_or_reservation_mismatch")
        if child.state in {"queued", "running"}:
            unknown_bad.append("unresolved_retrieval")
        if child.context_scope is None:
            if (child.context_binding_valid is not None or child.context_limits is not None or child.calls
                    or child.calls_used is not None or child.input_bytes_used is not None):
                budget_bad.append("retrieval_context_inconsistent")
            if child.state != "failed":
                budget_missing.append("retrieval_model_context_missing")
                unknown_missing.append("retrieval_unknown_unverified")
            continue
        if (child.context_binding_valid is not True or child.context_scope != scope
                or child.context_limits != child.profile_limits or child.calls_used != len(child.calls)
                or child.input_bytes_used != sum(call.input_bytes for call in child.calls)
                or len({call.call_id for call in child.calls}) != len(child.calls)
                or len({call.attempt_id for call in child.calls}) != len(child.calls)
                or len({call.task_type for call in child.calls}) != len(child.calls)
                or len(child.calls) > child.profile_limits.max_calls
                or sum(call.input_bytes for call in child.calls) > child.profile_limits.max_input_bytes):
            budget_bad.append("retrieval_child_budget_or_identity_mismatch")
        if child.state == "succeeded" and not any(call.task_type == "embedding" for call in child.calls):
            budget_missing.append("retrieval_embedding_call_missing")
            unknown_missing.append("retrieval_unknown_unverified")
        if child.calls:
            # 当前账本没有持久化请求正文，不能离线重算请求指纹与输入字节数的关系。
            budget_missing.append("retrieval_input_size_unverified")
        for call in child.calls:
            expected_binding = embedding if call.task_type == "embedding" else rerank
            if call.binding_hash != expected_binding:
                budget_bad.append("retrieval_model_binding_mismatch")
            if call.state == "reserved":
                unknown_bad.append("unresolved_retrieval_model")
            elif call.state == "succeeded":
                result_identity = (call.result_call_id, call.result_attempt_id, call.result_fingerprint, call.result_task_type)
                if any(item is None for item in result_identity):
                    budget_missing.append("retrieval_result_identity_missing")
                elif result_identity != (call.call_id, call.attempt_id, call.request_fingerprint, call.task_type):
                    budget_bad.append("retrieval_result_identity_mismatch")
            else:
                if call.dispatch_status is None or call.billing_status is None:
                    unknown_missing.append("retrieval_failure_status_missing")
                elif call.dispatch_status == "unknown":
                    unknown_bad.append("unresolved_retrieval_model")
                elif call.billing_status == "not_billable" and call.dispatch_status != "not_sent":
                    budget_bad.append("retrieval_failure_billing_invalid")
    return budget_bad, budget_missing, unknown_bad, unknown_missing
