"""显式安装可选检索模型账本；无导入时副作用，不随core迁移自动运行。"""

from pathlib import Path

from psycopg import AsyncConnection

from harness.bootstrap.event_loop import run
from harness.bootstrap.providers import model_values


async def migrate_retrieval(root: Path) -> None:
    dsn = model_values(root).get("HARNESS_DATABASE_DSN", "")
    if not dsn:
        raise ValueError("可选检索迁移需要本项目HARNESS_DATABASE_DSN")
    async with await AsyncConnection.connect(
        dsn, connect_timeout=5
    ) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(184031,7)")
            exists = await (await conn.execute(
                "SELECT to_regclass('harness_rag.schema_versions')"
            )).fetchone()
            if exists is not None and exists[0] is not None:
                rows = await (await conn.execute(
                    "SELECT version FROM harness_rag.schema_versions ORDER BY version"
                )).fetchall()
                versions = [row[0] for row in rows]
                if versions not in ([1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]):
                    raise ValueError("可选检索迁移版本不匹配")
            else:
                versions = []
            for version, filename in (
                (1, "0001_retrieval_models.sql"), (2, "0002_ingestion.sql"),
                (3, "0003_index_snapshots.sql"),
                (4, "0004_retrieval.sql"),
                (5, "0005_lifecycle.sql"),
            ):
                if version not in versions:
                    await conn.execute(
                        (root / "migrations/rag" / filename).read_text("utf-8"), prepare=False,
                    )


if __name__ == "__main__":
    run(migrate_retrieval(Path.cwd().resolve()))
