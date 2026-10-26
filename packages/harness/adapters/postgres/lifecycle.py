"""终止事务中的显式协作顺序；unknown 留待核对，外部 I/O 不进入本模块。"""

from dataclasses import dataclass

from harness.adapters.postgres.connection import Connection, Row
from harness.adapters.postgres.events import PostgresEventJournal


@dataclass(frozen=True, slots=True)
class PostgresRunLifecycle:
    events: PostgresEventJournal

    async def terminate(self, conn: Connection, row: Row, status: str, code: str) -> None:
        if self.events.drafts:
            draft = await (
                await conn.execute(
                    (
                        "UPDATE harness.answer_drafts SET state='invalid' WHERE run_id=%s "
                        "AND state='published' RETURNING draft_id"
                    ),
                    (row["run_id"],),
                )
            ).fetchone()
            if draft:
                await self.events.append(
                    conn,
                    row["run_id"],
                    "answer.invalidated",
                    {"draft_id": draft["draft_id"], "reason": code},
                )
        if self.events.writes:
            unknown = await (
                await conn.execute(
                    (
                        "SELECT operation_id FROM harness.operations WHERE run_id=%s AND "
                        "state IN ('dispatched','unknown')"
                    ),
                    (row["run_id"],),
                )
            ).fetchall()
            if unknown:
                await conn.execute(
                    (
                        "UPDATE harness.operations SET state='unknown' WHERE run_id=%s "
                        "AND state='dispatched'"
                    ),
                    (row["run_id"],),
                )
                await conn.execute(
                    (
                        "UPDATE harness.runs SET "
                        "status='reconciling',error_code=%s,execution_used_ms=execution_us"
                        "ed_ms+CASE WHEN execution_started_at IS NULL THEN 0 ELSE "
                        "greatest(0,extract(epoch FROM "
                        "clock_timestamp()-execution_started_at)*1000)::bigint "
                        "END,execution_started_at=NULL,revision=revision+1 WHERE run_id=%s"
                    ),
                    (code, row["run_id"]),
                )
                await self.events.append(
                    conn,
                    row["run_id"],
                    "run.reconciling",
                    {"code": code, "operations": [item["operation_id"] for item in unknown]},
                )
                return
        await conn.execute(
            (
                "UPDATE harness.runs SET "
                "status=%s,error_code=%s,revision=revision+1,execution_used_ms=CAS"
                "E WHEN execution_started_at IS NULL THEN execution_used_ms ELSE "
                "execution_used_ms+greatest(0,extract(epoch FROM "
                "clock_timestamp()-execution_started_at)*1000)::bigint "
                "END,execution_started_at=NULL WHERE run_id=%s"
            ),
            (status, code, row["run_id"]),
        )
        await conn.execute(
            (
                "UPDATE harness.sessions SET active_run_id=NULL WHERE "
                "session_id=%s AND active_run_id=%s"
            ),
            (row["session_id"], row["run_id"]),
        )
        await self.events.append(conn, row["run_id"], f"run.{status}", {"code": code})
        if self.events.writes:
            # 只有权威未执行证据允许终态释放动作guard；成功和unknown均保留。
            await conn.execute(
                (
                    "DELETE FROM harness.business_action_guards g USING "
                    "harness.operations o WHERE g.operation_id=o.operation_id AND "
                    "o.run_id=%s AND o.state='not_executed'"
                ),
                (row["run_id"],),
            )
