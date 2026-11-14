"""候选、接纳、实际起始/终止及PG版本证据的纯比较。"""

from harness.contracts.evaluation import EvaluationCase, EvaluationFreeze
from harness.contracts.evaluation_gates import GateCheck, RunGateEvidence
from harness.contracts.evaluation_versions import ExecutionVersion
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import EvaluationContext, ReleaseManifest


def compare_versions(collection: dict[str, object], frozen: EvaluationFreeze,
                     rows: tuple[EvaluationCase, ...], candidate: ReleaseManifest) -> dict[str, object]:
    failures: list[str] = []
    missing: list[str] = []
    expected = candidate.content_hash()
    try:
        declared = ReleaseManifest.model_validate(collection.get("candidate_manifest"))
        if declared.content_hash() != expected:
            failures.append("candidate_declaration_mismatch")
    except (ValueError, TypeError):
        missing.append("candidate_manifest_missing")
    context = None
    try:
        context = EvaluationContext.model_validate(collection.get("evaluation_context"))
        if (context.manifest_hash != expected or context.candidate_release_id != candidate.spec.release_id
                or context.environment_profile_hash != candidate.environment_profile_hash
                or context.agent_id != candidate.spec.agent_id or context.dataset_hash != frozen.dataset_hash
                or context.gate_policy_version != collection.get("gate_policy_version")):
            failures.append("evaluation_context_version_mismatch")
        if context.application_id is None:
            missing.append("context_application_version_missing")
    except (ValueError, TypeError):
        missing.append("evaluation_context_missing")
    records = collection.get("cases")
    records = records if isinstance(records, list) else []
    reports = []
    executed = False
    for row in rows:
        record = next((record for record in records if isinstance(record, dict)
                       and record.get("case_id") == row.case_id), {})
        accepted = record.get("accepted")
        if not isinstance(accepted, dict):
            continue
        executed = True
        bad: list[str] = []
        absent: list[str] = []
        run_id = accepted.get("run_id")

        def match(value: object, name: str, env_field: str = "environment_profile_hash") -> None:
            if not isinstance(value, dict):
                absent.append(name + "_missing")
                return
            for key, target in (("manifest_hash", expected), (env_field, candidate.environment_profile_hash),
                                ("release_id", candidate.spec.release_id)):
                if key not in value:
                    absent.append(name + "_" + key + "_missing")
                elif value[key] != target:
                    bad.append(name + "_" + key + "_mismatch")

        match(accepted, "api_accepted")
        try:
            raw_events = record.get("events")
            if not isinstance(raw_events, list):
                raise ValueError("缺事件")
            events = [RuntimeEvent.model_validate(event) for event in raw_events]
        except (ValueError, TypeError):
            events = []
            absent.append("version_events_invalid")
        accepted_events = [event for event in events if event.type == "run.accepted"]
        if len(accepted_events) != 1:
            absent.append("persisted_accepted_missing")
        else:
            match(accepted_events[0].payload, "persisted_accepted")
        started = [event for event in events if event.type == "run.started"]
        terminals = [event for event in events if event.type in {
            "run.completed", "run.failed", "run.cancelled", "run.expired"}]
        proofs = {event.seq: event for event in events if event.type == "run.version_observed"}
        for proof in proofs.values():
            try:
                actual = ExecutionVersion.model_validate(proof.payload.get("observation"))
                bad.extend("observed_" + field + "_mismatch" for field in actual.differences(candidate))
            except (ValueError, TypeError):
                absent.append("version_observation_unavailable")
        if not started:
            absent.append("actual_started_missing")
        used: set[int] = set()
        details = []
        for event, phase in [(event, "start") for event in started] + [(event, "terminal") for event in terminals]:
            reference = event.payload.get("version_evidence_seq")
            if event.payload.get("observed_manifest_hash") is None:
                absent.append(phase + "_observed_manifest_missing")
            elif event.payload["observed_manifest_hash"] != expected:
                bad.append(phase + "_observed_manifest_mismatch")
            if event.payload.get("observed_environment_profile_hash") is None:
                absent.append(phase + "_observed_environment_missing")
            elif event.payload["observed_environment_profile_hash"] != candidate.environment_profile_hash:
                bad.append(phase + "_observed_environment_mismatch")
            if type(reference) is not int or reference not in proofs:
                absent.append(phase + "_version_reference_missing")
                continue
            proof = proofs[reference]
            owner = event.payload.get("execution_identity") if phase == "start" else (
                started[-1].payload.get("execution_identity") if started else None)
            if (reference in used or proof.seq >= event.seq or proof.run_id != run_id
                    or proof.payload.get("phase") != phase
                    or proof.payload.get("execution_identity") != owner
                    or (phase == "terminal" and started and proof.seq <= started[-1].seq)):
                bad.append("version_reference_binding_mismatch")
            used.add(reference)
            details.append({"phase": phase, "event_seq": event.seq, "proof_seq": proof.seq,
                            "execution_identity": owner, "observation": proof.payload.get("observation")})
        if len(terminals) != 1:
            absent.append("terminal_version_missing")
        final = record.get("final")
        match(final, "final_view")
        if isinstance(final, dict) and final.get("run_id") != run_id:
            bad.append("final_run_mismatch")
        try:
            ledger = RunGateEvidence.model_validate(record.get("ledger_evidence"))
            if (ledger.run_id != run_id or ledger.manifest_hash != expected
                    or ledger.environment_profile_hash != candidate.environment_profile_hash
                    or (context is not None and ledger.context_id != context.context_id)):
                bad.append("ledger_version_mismatch")
            if tuple(canonical_hash(event.model_dump(mode="json")) for event in events) != ledger.event_hashes:
                bad.append("version_evidence_not_persisted")
        except (ValueError, TypeError):
            absent.append("version_ledger_missing")
        if candidate.source_revision is not None:
            # 当前组合根提供工作树hash，不把客户端Git SHA当作实际读取的HEAD。
            absent.append("declared_git_revision_unverified")
        case_outcome = "failed" if bad else "incomplete" if absent else "passed"
        reports.append({"case_id": row.case_id, "outcome": case_outcome,
            "reasons": list(dict.fromkeys(bad + absent)), "observations": details})
        failures.extend(bad)
        missing.extend(absent)
    outcome = "failed" if failures else "not_run" if not executed else "incomplete" if missing else "passed"
    return {"version": "execution-version-comparison@1", "outcome": outcome,
            "expected_manifest_hash": expected, "expected_source_identity": {"kind": "working_tree",
                "source_revision": candidate.source_revision, "source_tree_hash": candidate.source_tree_hash},
            "checks": [GateCheck(gate="execution_versions", outcome="failed" if failures else "incomplete" if missing else "passed",
                                 reasons=tuple(dict.fromkeys(failures + missing))).model_dump(mode="json")],
            "cases": reports}
