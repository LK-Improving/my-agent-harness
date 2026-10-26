"""工单完成验证；只相信当前run的授权持久回执，拒绝模型自报成功。"""

from datetime import UTC, datetime

from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import AnswerCandidate, Principal
from harness.contracts.state import ValidationReport
from harness.contracts.writes import WriteLedger

from domain_tools.operations.work_orders import CreateWorkOrderInput, WorkOrderOutput


class WorkOrderCompletionValidator:
    def __init__(self, ledger: WriteLedger) -> None:
        self.ledger = ledger

    async def validate(
        self,
        actor: Principal,
        run_id: str,
        expected: CreateWorkOrderInput,
        candidate: AnswerCandidate,
        validator_ref: str,
    ) -> ValidationReport:
        # expected来自引擎固定的用户任务，不从模型答案提取；回执必须属于当前run。
        receipts = await self.ledger.write_receipts(actor, run_id)
        matching = []
        answer_matches = False
        for receipt in receipts:
            if receipt.args_hash != canonical_hash(expected.model_dump(mode="json")):
                continue
            try:
                output = WorkOrderOutput.model_validate(receipt.data, strict=True)
            except ValueError:
                continue
            if (
                output.order_id == expected.order_id
                and output.work_order_id == receipt.external_reference
            ):
                matching.append(receipt.operation_id)
                answer_matches |= (
                    output.order_id in candidate.answer and output.work_order_id in candidate.answer
                )
        checks = {
            "task_source": candidate.task.source_turn_id == run_id,
            "authorized_receipt": bool(matching),
            "answer_matches_receipt": answer_matches,
            "no_unfinished_items": not candidate.task.pending_items and not candidate.task.blockers,
            "candidate_outcome": candidate.answer_kind == "answer"
            and candidate.task_outcome == "achieved",
        }
        return ValidationReport(
            validator_ref=validator_ref,
            input_hash=canonical_hash(candidate.validation_input()),
            verdict="pass" if all(checks.values()) else "fail",
            checks=checks,
            evidence_refs=tuple(matching),
            checked_at=datetime.now(UTC),
        )
