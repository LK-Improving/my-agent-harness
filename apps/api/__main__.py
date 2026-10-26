"""在工作区根目录运行 python -m apps.api；仅监听本机。"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "harness.bootstrap.server:application",
        factory=True,
        loop="harness.bootstrap.event_loop:postgres_loop",
        host="127.0.0.1",
        port=18080,
        access_log=False,
        log_level="critical",
    )
