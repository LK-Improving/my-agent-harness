"""显式安装本项目业务迁移；不会自动执行或修改官方 saver。"""

from pathlib import Path

from psycopg import AsyncConnection

from harness.bootstrap.event_loop import run
from harness.bootstrap.server import load_server_settings


async def migrate(root: Path) -> None:
    settings = load_server_settings(root)
    async with await AsyncConnection.connect(
        settings.database_dsn.get_secret_value(), connect_timeout=5
    ) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(184031,1)")
            exists = await (
                await conn.execute("SELECT to_regclass('harness.schema_versions')")
            ).fetchone()
            versions: list[int] = []
            if exists is not None and exists[0] is not None:
                installed = await (
                    await conn.execute(
                        "SELECT version FROM harness.schema_versions ORDER BY version"
                    )
                ).fetchall()
                versions = [row[0] for row in installed]
                if versions not in [list(range(1, end + 1)) for end in range(1, 11)]:
                    raise ValueError("业务迁移版本不匹配；请使用兼容构建")
            for version, name in (
                (1, "0001_runtime.sql"),
                (2, "0002_writes.sql"),
                (3, "0003_execution.sql"),
                (4, "0004_extensions.sql"),
                (5, "0005_artifacts.sql"),
                (6, "0006_artifact_retention.sql"),
                (7, "0007_context_purpose.sql"),
                (8, "0008_context_summaries.sql"),
                (9, "0009_retention.sql"),
                (10, "0010_evaluation_contexts.sql"),
            ):
                if version not in versions:
                    await conn.execute(
                        (root / "migrations" / name).read_text(encoding="utf-8"), prepare=False
                    )


if __name__ == "__main__":
    run(migrate(Path.cwd().resolve()))
