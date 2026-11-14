"""顺序逐例采集；只产出证据，不授予质量通过或发布资格。"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from harness.contracts.evaluation import CollectionFault, EvaluationAPI, EvaluationCase, EvaluationFreeze
from harness.contracts.evaluation_contexts import EvaluationLimits
from harness.contracts.evaluation_gates import RunGateEvidence
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import EvaluationRequest, SessionRequest


async def collect(
    api: EvaluationAPI, frozen: EvaluationFreeze, rows: tuple[EvaluationCase, ...],
    candidate: ReleaseManifest, gate_version: str, batch_id: str,
    resource_scopes: frozenset[str], emit: Callable[[dict[str, object]], None],
    *, limits: EvaluationLimits | None = None, lifetime_seconds: int = 3600,
    expected_identity: tuple[str, str, str] | None = None,
) -> dict[str, object]:
    """每次创建前落意图；结果不明及审批挂起时停止后续业务创建。"""
    async def record(body: dict[str, object]) -> None:
        # 本地落盘含fsync，必须等待成功再继续创建，同时避免阻塞异步事件循环。
        await asyncio.to_thread(emit, body)

    async def ledger_evidence(entry: dict[str, object]) -> None:
        reader = getattr(api, "evaluation_evidence", None)
        accepted = entry.get("accepted")
        if reader is None or not isinstance(accepted, dict):
            entry["ledger_evidence_error"] = "EVIDENCE_READER_NOT_INSTALLED"
            return
        try:
            evidence = RunGateEvidence.model_validate(await reader(accepted["run_id"]))
            entry["ledger_evidence"] = evidence.model_dump(mode="json")
            await record({"phase": "ledger_evidence", "case_id": entry["case_id"],
                          "evidence": entry["ledger_evidence"]})
        except Exception as error:
            # 撤销/到期或依赖失败不通过旁路补正文；保留缺口，不覆盖原始采集错误。
            entry["ledger_evidence_error"] = type(error).__name__

    report: dict[str, object] = {
        "version": "evaluation-collection@1", "batch_id": batch_id,
        "freeze_hash": frozen.content_hash(), "dataset_hash": frozen.dataset_hash,
        "candidate_manifest": candidate.model_dump(mode="json"),
        "gate_policy_version": gate_version, "baseline": None,
        "expected_case_ids": [row.case_id for row in rows],
        "outcome": "not_run", "collection_complete": False,
        "unverified": ["deterministic_gates", "manifest_comparison", "required_judge",
                       "quality_thresholds", "cost_coverage", "release_eligibility"],
    }
    await record({"phase": "batch_intent", **report})
    results: list[dict[str, object]] = []
    report["cases"] = results
    try:
        if lifetime_seconds < 1 or lifetime_seconds > 86400:
            raise ValueError("EVALUATION_LIFETIME_INVALID")
        selected_limits = limits or EvaluationLimits()
        context = await api.create_context(EvaluationRequest(
            candidate_release_id=candidate.spec.release_id, dataset_hash=frozen.dataset_hash,
            gate_policy_version=gate_version, resource_scopes=resource_scopes,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=lifetime_seconds),
            limits=selected_limits,
        ))
        await record({"phase": "context_created", "context": context.model_dump(mode="json")})
        report["evaluation_context"] = context.model_dump(mode="json")
        if (context.manifest_hash != candidate.content_hash()
                or context.dataset_hash != frozen.dataset_hash
                or context.gate_policy_version != gate_version
                or context.candidate_release_id != candidate.spec.release_id
                or context.agent_id != candidate.spec.agent_id
                or context.limits != selected_limits
                or context.environment_profile_hash != candidate.environment_profile_hash
                or context.resource_scopes != resource_scopes
                or (expected_identity is not None and
                    (context.tenant_id, context.subject_id, context.application_id) != expected_identity)):
            raise ValueError("EVALUATION_BINDING_MISMATCH")
        for row in rows:
            entry: dict[str, object] = {"case_id": row.case_id, "expected": row.expected,
                                        "outcome": "incomplete", "events": []}
            results.append(entry)
            request_id = "eval-" + canonical_hash({"batch": batch_id, "case": row.case_id})[:40]
            await record({"phase": "case_intent", "case_id": row.case_id, "request_id": request_id})
            session_id = await api.create_session(SessionRequest(
                agent_id=candidate.spec.agent_id, evaluation_context_id=context.context_id))
            entry["session_id"] = session_id
            await record({"phase": "session_created", "case_id": row.case_id, "session_id": session_id})
            report["outcome"] = "incomplete"
            accepted = await api.create_run(session_id, RunRequest(
                request_id=request_id, expected_revision=0, message=row.message,
                attachment_versions=row.attachment_versions, business_options=row.business_options))
            entry["accepted"] = accepted.model_dump(mode="json")
            await record({"phase": "run_created", "case_id": row.case_id, "accepted": entry["accepted"]})
            events: list[dict[str, object]] = []
            entry["events"] = events
            async for event in api.events(accepted.run_id):
                body = event.model_dump(mode="json")
                events.append(body)
                await record({"phase": "event", "case_id": row.case_id, "event": body})
                if event.type == "run.started":
                    entry["started"] = body
                if event.type in {"run.completed", "run.failed", "run.cancelled", "run.expired"}:
                    entry["terminal"] = body
            view = await api.get_run(accepted.run_id)
            entry["final"] = view.model_dump(mode="json")
            await ledger_evidence(entry)
            if "terminal" not in entry:
                raise ValueError("TERMINAL_MISSING")
            await record({"phase": "case_collected", **entry})
        report["collection_complete"] = len(results) == len(rows)
    except Exception as error:
        # 异常类型足以定位采集阶段；不序列化带 URL/认证/响应正文的异常文本。
        report["collection_error"] = type(error).__name__
        code = str(error) if isinstance(error, CollectionFault) else type(error).__name__
        report["collection_error_code"] = code
        if results and "accepted" in results[-1] and "ledger_evidence" not in results[-1]:
            await ledger_evidence(results[-1])
        await record({"phase": "collection_interrupted", "error_type": type(error).__name__, "code": code})
    return report
