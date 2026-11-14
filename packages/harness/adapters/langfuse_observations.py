"""通过有界队列导出已脱敏的 Langfuse OTLP 事件。"""

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from logging import getLogger

import httpx

from harness.contracts.observability import Observation

logger = getLogger(__name__)


class LangfuseObservationSink:
    def __init__(self, endpoint: str, public_key: str, secret_key: str) -> None:
        self.endpoint = endpoint
        self.client = httpx.AsyncClient(
            auth=(public_key, secret_key), timeout=2, follow_redirects=False, trust_env=False,
            headers={"x-langfuse-ingestion-version": "4"},
        )
        self.queue: asyncio.Queue[Observation] = asyncio.Queue(maxsize=128)
        self.task: asyncio.Task[None] | None = None
        self.closed = False
        self.failed = 0
        self.dropped = 0

    def offer(self, observation: Observation) -> None:
        if self.closed or self.queue.full():
            self.dropped += 1
            return
        self.queue.put_nowait(observation)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while not self.queue.empty():
            event = self.queue.get_nowait()
            try:
                async with asyncio.timeout(2), self.client.stream(
                    "POST", self.endpoint, json=self._payload(event),
                ) as response:
                    response.raise_for_status()
                    # 追踪结果不返回业务层，也不读取远端正文。
            except Exception:
                self.failed += 1
                if self.failed == 1:
                    logger.warning("Langfuse 追踪导出失败")

    @staticmethod
    def _payload(event: Observation) -> dict[str, object]:
        elapsed = event.timestamp - datetime(1970, 1, 1, tzinfo=UTC)
        nanos = (elapsed.days * 86400 + elapsed.seconds) * 10**9 + elapsed.microseconds * 1000
        span_id = sha256(f"{event.trace_id}/{event.seq}".encode()).hexdigest()[:16]
        span = {"traceId": event.trace_id, "spanId": span_id, "name": event.event_type, "kind": 1,
                "startTimeUnixNano": str(nanos), "endTimeUnixNano": str(nanos),
                "attributes": [{"key": "harness.event.seq", "value": {"intValue": str(event.seq)}}]}
        return {"resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "agent-harness"}}]},
            "scopeSpans": [{"scope": {"name": "harness.persisted-events"}, "spans": [span]}],
        }]}

    async def aclose(self) -> None:
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        try:
            async with asyncio.timeout(2):
                await self.client.aclose()
        except Exception:
            logger.warning("Langfuse 追踪客户端关闭失败")
