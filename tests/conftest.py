"""测试只连接显式配置的本项目隔离库，不猜测其他项目连接。"""

import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from dotenv import load_dotenv
from harness.bootstrap.event_loop import postgres_loop
from psycopg import sql

from tests.integration.test_m1_runtime import prepared as prepared

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


class RedactedDSN(str):
    """断言错误展示参数时不把测试连接中的密码写进报告。"""

    def __repr__(self) -> str:
        return "<本项目测试连接已脱敏>"


@pytest.fixture
def pg_dsn() -> str:
    dsn = os.environ.get("HARNESS_TEST_DSN")
    if not dsn:
        pytest.fail("必须显式设置本项目 HARNESS_TEST_DSN，集成测试不能跳过依赖")
    parsed = urlparse(dsn)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path != "/harness_m0":
        pytest.fail("M0 探针仅允许本机 harness_m0 隔离测试数据库")
    return RedactedDSN(dsn)


@pytest.fixture
def pg_schema(pg_dsn: str) -> Iterator[tuple[str, str]]:
    namespace = "poc_" + uuid4().hex
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    scoped = psycopg.conninfo.make_conninfo(pg_dsn, options=f"-c search_path={namespace}")
    try:
        yield RedactedDSN(scoped), namespace
    finally:
        # 仅清理本用例刚创建且具有随机前缀的隔离 schema。
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))


def pytest_asyncio_loop_factories(config, item):
    return {"postgres": postgres_loop}
