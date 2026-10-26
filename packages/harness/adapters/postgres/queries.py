"""已授权范围内的证据查询；调用者传入同一事务连接，不在此重新连接。"""

from harness.adapters.postgres.connection import Connection, Row
from harness.adapters.postgres.mapping import write_operation
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import AnswerCandidate, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.state import ToolReceipt


class PostgresEvidenceQueries:
    @staticmethod
    async def read_receipts(conn: Connection, run_id: str) -> tuple[ToolReceipt, ...]:
        rows = await (
            await conn.execute(
                "SELECT receipt FROM harness.read_operations WHERE run_id=%s AND state='succeeded'",
                (run_id,),
            )
        ).fetchall()
        return tuple(ToolReceipt.model_validate(row["receipt"]) for row in rows)

    @staticmethod
    async def write_receipts(conn: Connection, run_id: str) -> tuple[ToolReceipt, ...]:
        rows = await (
            await conn.execute(
                (
                    "SELECT receipt FROM harness.operations WHERE run_id=%s AND "
                    "state='succeeded' AND output_error IS NULL ORDER BY operation_id"
                ),
                (run_id,),
            )
        ).fetchall()
        return tuple(
            ToolReceipt.model_validate(row["receipt"]) for row in rows if row["receipt"] is not None
        )

    @staticmethod
    async def candidate(conn: Connection, run_id: str) -> AnswerCandidate | None:
        draft = await (
            await conn.execute("SELECT * FROM harness.answer_drafts WHERE run_id=%s", (run_id,))
        ).fetchone()
        if draft is None:
            return None
        if draft["state"] == "invalid":
            raise RuntimeFault("ANSWER_DRAFT_INVALIDATED")
        candidate = AnswerCandidate.model_validate(draft["candidate"])
        if canonical_hash(candidate.validation_input()) != draft["candidate_hash"]:
            raise RuntimeFault("CANDIDATE_SNAPSHOT_CORRUPT")
        return candidate

    @staticmethod
    async def operation_state(conn: Connection, op: Row) -> tuple[JsonSnapshot, JsonSnapshot]:
        attempts = await (
            await conn.execute(
                (
                    "SELECT attempt_id,ordinal,requires_reconciliation FROM "
                    "harness.tool_attempts WHERE operation_id=%s ORDER BY ordinal"
                ),
                (op["operation_id"],),
            )
        ).fetchall()
        return JsonSnapshot.capture(
            write_operation(op).model_dump(mode="json")
        ), JsonSnapshot.capture(attempts)
