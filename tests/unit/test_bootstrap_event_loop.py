"""防止 Windows PG 循环与 Uvicorn 自定义入口再次失配。"""

import asyncio
import sys

import uvicorn
from harness.bootstrap.event_loop import postgres_loop, run


def test_uvicorn_custom_factory_returns_actual_postgres_loop():
    config = uvicorn.Config("unused:app", loop="harness.bootstrap.event_loop:postgres_loop")
    factory = config.get_loop_factory()
    assert factory is not None
    loop = factory()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
        if sys.platform == "win32":
            assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_cli_uses_same_factory_and_propagates_result():
    async def identify():
        return isinstance(asyncio.get_running_loop(), type(postgres_loop_for_comparison))

    postgres_loop_for_comparison = postgres_loop()
    try:
        assert run(identify())
    finally:
        postgres_loop_for_comparison.close()
