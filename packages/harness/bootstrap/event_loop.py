"""本机 PG 入口统一使用兼容事件循环；Windows 不启动异步子进程。"""

import asyncio
import selectors
import sys
from collections.abc import Callable, Coroutine
from typing import Any


def postgres_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def uvicorn_loop_factory(use_subprocess: bool = False) -> Callable[[], asyncio.AbstractEventLoop]:
    return postgres_loop


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, loop_factory=postgres_loop)
