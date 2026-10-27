"""基于本轮授权持久回执验证引用与候选身份；不声称语义蕴含已由Judge验证。"""

import re
from datetime import UTC, datetime
from typing import Protocol

from harness.contracts.hashing import canonical_hash
from harness.contracts.artifacts import ArtifactPage
from harness.contracts.retrieval import Evidence, KnowledgeRetrieval
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, Principal, RuntimeFault
from harness.contracts.state import ToolReceipt, ValidationReport

from domain_tools.knowledge.retrieval import KnowledgeOutput, receipt_output


class ReceiptSource(Protocol):
    async def read_receipts(self, actor: Principal, run_id: str) -> tuple[ToolReceipt, ...]: ...


def knowledge_receipts(receipts: tuple[ToolReceipt, ...]) -> tuple[ToolReceipt, ...]:
    """内置分页回执不是新知识检索；只能从真实检索回执验证完成。"""
    result = []
    for receipt in receipts:
        if receipt.cost_status == "not_billable":
            ArtifactPage.model_validate(receipt.data)
        else:
            result.append(receipt)
    return tuple(result)


def reference(evidence: Evidence) -> str:
    return f"{evidence.snapshot_id}/{evidence.version_id}/{evidence.chunk_id}"


def validate_knowledge(run: ClaimedRun, candidate: AnswerCandidate,
                       outputs: tuple[KnowledgeOutput, ...], receipts: tuple[ToolReceipt, ...]) -> ValidationReport:
    evidence = {reference(e): e for output in outputs for result in output.results for e in result.evidence}
    cited = tuple(re.findall(r"\[\[source:([^\]\s]+)\]\]", candidate.answer))
    refs = candidate.task.evidence_refs
    checks = {
        "task_source": candidate.task.source_turn_id == run.run_id,
        "authorized_receipt": len(receipts) == 1 and len(outputs) == 1,
        "all_bound_kbs_returned": bool(outputs) and len(outputs[0].results) == len(run.manifest.spec.knowledge_bindings),
        "no_unfinished_items": not candidate.task.pending_items and not candidate.task.blockers,
        "citation_integrity": len(set(refs)) == len(refs) and set(refs) == set(cited) and set(refs) <= evidence.keys(),
    }
    if candidate.task_outcome == "achieved":
        checks["answer_has_current_evidence"] = candidate.answer_kind == "answer" and bool(refs) and bool(evidence)
    else:
        checks["safe_outcome"] = (candidate.answer_kind, candidate.task_outcome) in {
            ("clarification", "needs_input"), ("abstention", "not_achieved"),
        }
    if not evidence:
        checks["no_evidence_no_answer"] = candidate.answer_kind != "answer" and candidate.task_outcome != "achieved" and not refs
    degraded = {reason for output in outputs for result in output.results for reason in result.degraded}
    if degraded:
        checks["degradation_disclosed"] = any(text in candidate.answer for text in ("降级", "部分检索", "重排不可用", "部分通路"))
    passed = all(checks.values())
    return ValidationReport(validator_ref=run.manifest.spec.completion_validator_ref,
                             input_hash=canonical_hash(candidate.validation_input()),
                             verdict=("needs_clarification" if candidate.answer_kind == "clarification" else "pass") if passed else "fail",
                             checks=checks, evidence_refs=tuple(r.operation_id for r in receipts) + refs,
                             checked_at=datetime.now(UTC))


class KnowledgeCompletionValidator:
    def __init__(self, receipts: ReceiptSource, service: KnowledgeRetrieval) -> None:
        self.receipts, self.service = receipts, service

    async def authorized(self, run: ClaimedRun, owner: str) -> tuple[tuple[ToolReceipt, ...], tuple[KnowledgeOutput, ...]]:
        receipts = knowledge_receipts(await self.receipts.read_receipts(run.principal, run.run_id))
        if len(receipts) != 1:
            raise RuntimeFault("KNOWLEDGE_TOOL_RECEIPT_MISSING", 409)
        outputs = tuple(receipt_output(run, receipt) for receipt in receipts)
        for output in outputs:
            for result in output.results:
                await self.service.authorize_result(run.principal, owner, run.run_id, result)
        return receipts, outputs

    async def validate(self, run: ClaimedRun, owner: str, candidate: AnswerCandidate) -> ValidationReport:
        receipts, outputs = await self.authorized(run, owner)
        return validate_knowledge(run, candidate, outputs, receipts)
