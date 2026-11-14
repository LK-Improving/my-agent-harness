"""显式组合事件持久化与投影检查；从不创建或提交事务。"""

from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import Connection, observes_event, one, remember_observation
from harness.contracts.events import RuntimeEvent
from harness.contracts.runtime import AnswerCandidate, RuntimeFault


@dataclass(frozen=True, slots=True)
class PostgresEventJournal:
    writes: bool = False
    drafts: bool = False

    async def append(
        self, conn: Connection, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        if self.drafts and event_type in {"answer.started", "answer.delta", "run.completed"}:
            draft = await (
                await conn.execute("SELECT * FROM harness.answer_drafts WHERE run_id=%s", (run_id,))
            ).fetchone()
            if draft:
                if event_type != "run.completed":
                    return
                if (
                    draft["state"] != "published"
                    or payload["committed_draft_id"] != draft["draft_id"]
                ):
                    raise RuntimeFault("COMMITTED_DRAFT_MISMATCH")
                candidate = AnswerCandidate.model_validate(draft["candidate"])
                if draft["published_offset"] != len(candidate.answer):
                    raise RuntimeFault("ANSWER_PUBLICATION_INCOMPLETE")
                await conn.execute(
                    "UPDATE harness.answer_drafts SET state='committed' WHERE run_id=%s", (run_id,)
                )
        if self.writes and event_type in {"run.failed", "run.cancelled", "run.expired"}:
            effects = await (
                await conn.execute(
                    (
                        "SELECT operation_id,state,receipt->>'external_reference' AS "
                        "external_reference FROM harness.operations WHERE run_id=%s AND "
                        "attempts>0"
                    ),
                    (run_id,),
                )
            ).fetchall()
            payload = {**payload, "side_effects": effects}
        await self._persist(conn, run_id, event_type, payload)

    async def append_published(
        self, conn: Connection, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        # 仅供已锁定/校验草稿的发布流程使用，避免最终提交时再次生成整段 delta。
        if not self.drafts or event_type not in {"answer.started", "answer.delta"}:
            raise ValueError("仅草稿发布路径允许写入开始/增量事件")
        await self._persist(conn, run_id, event_type, payload)

    @staticmethod
    async def _persist(
        conn: Connection, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        observed = observes_event(conn, event_type)
        returning = "RETURNING event_seq,clock_timestamp() AS now"
        if observed:
            returning += """, EXISTS (
                SELECT 1 FROM harness.sessions s WHERE s.session_id=harness.runs.session_id
                  AND s.revoked_at IS NULL AND s.purged_at IS NULL
                  AND s.retention_policy->'external_trace_copies'='true'::jsonb
                ) AS external_trace_allowed"""
        row = await one(
            conn,
            "UPDATE harness.runs SET event_seq=event_seq+1 WHERE run_id=%s " + returning,
            (run_id,),
        )
        event = RuntimeEvent(
            run_id=run_id,
            seq=row["event_seq"],
            type=event_type,
            timestamp=row["now"],
            payload=payload,
        )
        await conn.execute(
            "INSERT INTO harness.events(run_id,seq,type,payload,timestamp) VALUES(%s,%s,%s,%s,%s)",
            (run_id, event.seq, event.type, Jsonb(event.payload), event.timestamp),
        )
        if observed and row["external_trace_allowed"]:
            remember_observation(conn, event)
