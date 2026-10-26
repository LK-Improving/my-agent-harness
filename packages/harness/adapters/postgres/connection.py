"""业务事务唯一连接入口；内部协作者只接收已有连接。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from harness.contracts.runtime import RuntimeFault

Row = dict[str, Any]
Connection = AsyncConnection[Row]


async def one(conn: Connection, sql: str, args: tuple[Any, ...] = ()) -> Row:
    row = await (await conn.execute(sql, args)).fetchone()
    if row is None:
        raise RuntimeFault("NOT_FOUND", 404)
    return row


@dataclass(frozen=True, slots=True)
class PostgresUnitOfWork:
    dsn: str = field(repr=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        async with await AsyncConnection.connect(
            self.dsn, row_factory=dict_row, connect_timeout=5
        ) as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL statement_timeout = '5s'")
                yield conn
