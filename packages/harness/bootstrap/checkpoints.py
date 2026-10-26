"""官方saver独立schema与显式setup；启动Worker时不自动迁移。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo

from harness.bootstrap.event_loop import run
from harness.bootstrap.server import load_server_settings


@asynccontextmanager
async def saver(dsn: str) -> AsyncIterator[AsyncPostgresSaver]:
    scoped = make_conninfo(
        dsn,
        options="-c search_path=harness_checkpoint -c statement_timeout=5000",
        connect_timeout=5,
    )
    async with AsyncPostgresSaver.from_conn_string(scoped) as checkpoint:
        yield checkpoint


async def setup(root: Path) -> None:
    dsn = load_server_settings(root).database_dsn.get_secret_value()
    async with await AsyncConnection.connect(dsn, autocommit=True, connect_timeout=5) as conn:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS harness_checkpoint")
    async with saver(dsn) as checkpoint:
        await checkpoint.setup()


if __name__ == "__main__":
    run(setup(Path.cwd().resolve()))
