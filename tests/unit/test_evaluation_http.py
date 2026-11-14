"""SSE 传输与单次创建的协议测试定义；不能代替真实 API 验收。"""

import json

import httpx
import pytest

from harness.adapters.evaluation_http import HTTPEvaluationAPI
from harness.contracts.evaluation import CollectionFault
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import SessionRequest


def frame(seq, kind="run.failed", **changes):
    body = {"run_id": "run-test", "seq": seq, "type": kind,
            "timestamp": "2026-10-27T00:00:00Z", "payload": {"code": "TEST_FAILED"}}
    body.update(changes)
    return f"id: {seq}\nevent: {kind}\ndata: {json.dumps(body)}\n\n"


async def test_creation_unknown_never_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadError("synthetic secret must not be reported")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        api = HTTPEvaluationAPI(client)
        with pytest.raises(CollectionFault, match="CREATE_OUTCOME_UNKNOWN"):
            await api.create_session(SessionRequest(agent_id="test"))
    assert len(calls) == 1


async def test_reconnect_uses_cursor_and_deduplicates_exact_replay():
    cursors = []
    first = frame(1, "test.audit")

    def handler(request):
        cursors.append(request.headers["last-event-id"])
        body = first if len(cursors) == 1 else first + frame(2)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        events = [event async for event in HTTPEvaluationAPI(client).events("run-test")]
    assert [event.seq for event in events] == [1, 2]
    assert cursors == ["0", "1"]


@pytest.mark.parametrize("body,code", [
    (frame(2), "SSE_SEQUENCE_GAP"),
    (frame(1, run_id="other"), "SSE_IDENTITY_MISMATCH"),
    (frame(1, "test.audit") + frame(1), "SSE_REPLAY_CONFLICT"),
])
async def test_protocol_corruption_is_not_silently_accepted(body, code):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        with pytest.raises(CollectionFault, match=code):
            _ = [event async for event in HTTPEvaluationAPI(client).events("run-test")]


async def test_expired_cursor_and_byte_limit_stop_collection():
    for status, body, code in [(410, "", "SSE_REJECTED_410"),
                               (200, frame(1), "SSE_BYTES_EXCEEDED")]:
        def handler(request):
            return httpx.Response(status, headers={"content-type": "text/event-stream"}, text=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
            with pytest.raises(CollectionFault, match=code):
                _ = [event async for event in HTTPEvaluationAPI(client, max_bytes=1).events("run-test")]


async def test_judge_response_checks_path_identity_and_content_hash():
    body = {"run_id": "run-test", "model_call_id": "call-test", "content": "合成响应",
            "content_hash": canonical_hash("合成响应")}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        api = HTTPEvaluationAPI(client)
        assert (await api.judge_model_response("run-test", "call-test")).content == "合成响应"
        body["run_id"] = "different"
        with pytest.raises(CollectionFault, match="JUDGE_RESPONSE_IDENTITY_MISMATCH"):
            await api.judge_model_response("run-test", "call-test")
        body["run_id"] = "run-test"
        body["content_hash"] = "0" * 64
        with pytest.raises(ValueError):
            await api.judge_model_response("run-test", "call-test")
        with pytest.raises(ValueError, match="请求路径"):
            await api.judge_model_response("../other", "call-test")
    assert len(requests) == 3


async def test_judge_source_material_checks_run_and_material_hash():
    material = {"evidence": "合成资料"}
    body = {"run_id": "run-test", "context_id": "context", "dataset_hash": "a" * 64,
            "manifest_hash": "b" * 64, "input_hash": "c" * 64, "result_hash": "d" * 64,
            "event_seq": 4, "source_ref": "knowledge-read@1", "material": material,
            "material_hash": canonical_hash(material)}

    def handler(request):
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        api = HTTPEvaluationAPI(client)
        assert (await api.judge_source_material("run-test")).material == material
        body["run_id"] = "other"
        with pytest.raises(CollectionFault, match="JUDGE_MATERIAL_IDENTITY_MISMATCH"):
            await api.judge_source_material("run-test")
        body["run_id"] = "run-test"
        body["material_hash"] = "0" * 64
        with pytest.raises(ValueError):
            await api.judge_source_material("run-test")
