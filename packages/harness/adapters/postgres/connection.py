"""业务事务唯一连接入口；内部协作者只接收已有连接。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from harness.contracts.events import RuntimeEvent
from harness.contracts.observability import NoopObservationSink, Observation, ObservationSink
from harness.contracts.runtime import RuntimeFault

Row = dict[str, Any]
Connection = AsyncConnection[Row]
_observations: ContextVar[tuple[Connection, list[Observation]] | None] = ContextVar(
    "postgres_transaction_observations", default=None,
)
OBSERVED_EVENT_TYPES = frozenset({
    "run.accepted", "run.started", "run.completed", "run.failed", "run.cancelled", "run.expired",
    "approval.required", "approval.decided", "answer.invalidated",
})


def observes_event(conn: Connection, event_type: str) -> bool:
    batch = _observations.get()
    return batch is not None and batch[0] is conn and len(batch[1]) < 64 and event_type in OBSERVED_EVENT_TYPES


def remember_observation(conn: Connection, event: RuntimeEvent) -> None:
    batch = _observations.get()
    if batch is not None and observes_event(conn, event.type):
        trace_id = sha256(("agent-harness/run/" + event.run_id).encode()).hexdigest()[:32]
        batch[1].append(Observation(trace_id, event.type, event.seq, event.timestamp))


async def one(conn: Connection, sql: str, args: tuple[Any, ...] = ()) -> Row:
    row = await (await conn.execute(sql, args)).fetchone()
    if row is None:
        raise RuntimeFault("NOT_FOUND", 404)
    return row


@dataclass(frozen=True, slots=True)
class PostgresUnitOfWork:
    dsn: str = field(repr=False)
    observation_sink: ObservationSink = field(default_factory=NoopObservationSink, repr=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        async with await AsyncConnection.connect(
            self.dsn, row_factory=dict_row, connect_timeout=5
        ) as conn:
            batch: list[Observation] = []
            token = _observations.set(
                None if isinstance(self.observation_sink, NoopObservationSink) else (conn, batch),
            )
            try:
                async with conn.transaction():
                    await conn.execute("SET LOCAL statement_timeout = '5s'")
                    yield conn
            finally:
                _observations.reset(token)
        for observation in batch:
            try:
                self.observation_sink.offer(observation)
            except Exception:
                # 已完成提交，诊断端的异常不能改变业务返回。
                break
