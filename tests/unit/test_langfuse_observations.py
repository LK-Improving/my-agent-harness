"""离线验证有界导出、协议形状和故障隔离。"""

import json
from datetime import UTC, datetime

import httpx

from harness.adapters.langfuse_observations import LangfuseObservationSink
from harness.contracts.observability import Observation


async def test_export_queue_and_transport_failure(monkeypatch):
    requests = []

    def receive(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200)

    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: async_client(transport=httpx.MockTransport(receive), **kwargs))
    sink = LangfuseObservationSink("https://example.invalid/api/public/otel/v1/traces", "test", "secret")
    try:
        for seq in range(130):
            sink.offer(Observation("a" * 32, "run.started", seq + 1, datetime(2026, 11, 7, tzinfo=UTC)))
        assert sink.dropped == 2 and sink.task is not None
        await sink.task
        assert sink.failed == 1 and len(requests) == 128
        span = json.loads(requests[0].content)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert set(span) == {"traceId", "spanId", "name", "kind", "startTimeUnixNano",
                             "endTimeUnixNano", "attributes"}
        assert span["attributes"] == [{"key": "harness.event.seq", "value": {"intValue": "1"}}]
        assert "secret" not in requests[0].content.decode()
    finally:
        await sink.aclose()
