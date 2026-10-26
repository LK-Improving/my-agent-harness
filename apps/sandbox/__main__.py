"""显式启动运营沙箱，默认不随核心服务启动。"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "harness.bootstrap.sandbox:application",
        factory=True,
        loop="harness.bootstrap.event_loop:postgres_loop",
        host="127.0.0.1",
        port=18081,
        access_log=False,
        log_level="critical",
    )
