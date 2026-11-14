"""真实 HTTP/SSE 采集，创建请求单次发送，断流只续读事件。"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from harness.contracts.events import RuntimeEvent
from harness.contracts.evaluation import CollectionFault
from harness.contracts.evaluation_gates import JudgeModelResponse, JudgeSourceMaterial, RunGateEvidence
from harness.contracts.hashing import RunRequest
from harness.contracts.release import EvaluationContext
from harness.contracts.runtime import EvaluationRequest, RunView, SessionRequest

TERMINALS = {"run.completed", "run.failed", "run.cancelled", "run.expired"}


class HTTPEvaluationAPI:
    def __init__(self, client: httpx.AsyncClient, *, timeout_seconds: float = 180,
                 max_events: int = 10000, max_bytes: int = 8388608,
                 reconnects: int = 3) -> None:
        if timeout_seconds <= 0 or max_events < 1 or max_bytes < 1 or reconnects < 0:
            raise ValueError("采集上限必须有效")
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.reconnects = reconnects

    async def _post(self, path: str, body: dict[str, object]) -> httpx.Response:
        try:
            response = await self.client.post(path, json=body)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as error:
            raise CollectionFault(f"API_REJECTED_{error.response.status_code}") from None
        except httpx.TransportError:
            # 不能证明服务端未写入，不自动重发 POST。
            raise CollectionFault("CREATE_OUTCOME_UNKNOWN") from None

    async def create_context(self, request: EvaluationRequest) -> EvaluationContext:
        body = request.model_dump(mode="json")
        body["resource_scopes"] = sorted(request.resource_scopes)
        response = await self._post("/api/v1/evaluation-contexts", body)
        return EvaluationContext.model_validate(response.json())

    async def create_session(self, request: SessionRequest) -> str:
        response = await self._post("/api/v1/sessions", request.model_dump(mode="json"))
        session_id = response.json()["session_id"]
        if not isinstance(session_id, str):
            raise CollectionFault("SESSION_RESPONSE_INVALID")
        # 只使用服务端已校验的 ID，避免响应值改变后续请求路径。
        from pydantic import TypeAdapter
        from harness.contracts.specs import Identifier

        return TypeAdapter(Identifier).validate_python(session_id)

    async def create_run(self, session_id: str, request: RunRequest) -> RunView:
        response = await self._post(f"/api/v1/sessions/{session_id}/runs",
                                    request.model_dump(mode="json"))
        body = response.json()
        body.pop("events_url", None)
        return RunView.model_validate(body)

    async def get_run(self, run_id: str) -> RunView:
        response = await self.client.get(f"/api/v1/runs/{run_id}")
        response.raise_for_status()
        return RunView.model_validate(response.json())

    async def evaluation_evidence(self, run_id: str) -> RunGateEvidence:
        response = await self.client.get(f"/api/v1/runs/{run_id}/evaluation-evidence")
        response.raise_for_status()
        return RunGateEvidence.model_validate(response.json())

    async def judge_model_response(self, run_id: str, call_id: str) -> JudgeModelResponse:
        from pydantic import TypeAdapter
        from harness.contracts.specs import Identifier

        run_id = TypeAdapter(Identifier).validate_python(run_id)
        call_id = TypeAdapter(Identifier).validate_python(call_id)
        if "/" in run_id or "/" in call_id or run_id in {".", ".."} or call_id in {".", ".."}:
            raise ValueError("运行和模型调用身份不能改变请求路径")
        response = await self.client.get(f"/api/v1/runs/{run_id}/evaluation-judge-responses/{call_id}")
        response.raise_for_status()
        result = JudgeModelResponse.model_validate(response.json())
        if result.run_id != run_id or result.model_call_id != call_id:
            raise CollectionFault("JUDGE_RESPONSE_IDENTITY_MISMATCH")
        return result

    async def judge_source_material(self, run_id: str) -> JudgeSourceMaterial:
        from pydantic import TypeAdapter
        from harness.contracts.specs import Identifier

        run_id = TypeAdapter(Identifier).validate_python(run_id)
        if "/" in run_id or run_id in {".", ".."}:
            raise ValueError("运行身份不能改变请求路径")
        response = await self.client.get(f"/api/v1/runs/{run_id}/evaluation-judge-material")
        response.raise_for_status()
        result = JudgeSourceMaterial.model_validate(response.json())
        if result.run_id != run_id:
            raise CollectionFault("JUDGE_MATERIAL_IDENTITY_MISMATCH")
        return result

    async def events(self, run_id: str) -> AsyncIterator[RuntimeEvent]:
        cursor, size, count = 0, 0, 0
        seen: dict[int, RuntimeEvent] = {}
        deadline = time.monotonic() + self.timeout_seconds
        async with asyncio.timeout(self.timeout_seconds):
            for attempt in range(self.reconnects + 1):
                try:
                    async with self.client.stream("GET", f"/api/v1/runs/{run_id}/events",
                                                  headers={"Last-Event-ID": str(cursor)}) as response:
                        response.raise_for_status()
                        if not response.headers.get("content-type", "").startswith("text/event-stream"):
                            raise CollectionFault("SSE_CONTENT_TYPE_INVALID")
                        pending = b""
                        frame: list[str] = []
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > self.max_bytes:
                                raise CollectionFault("SSE_BYTES_EXCEEDED")
                            pending += chunk
                            while b"\n" in pending:
                                raw, pending = pending.split(b"\n", 1)
                                line = raw.rstrip(b"\r").decode("utf-8")
                                if line:
                                    frame.append(line)
                                    continue
                                fields: dict[str, list[str]] = {}
                                for part in frame:
                                    if part.startswith(":"):
                                        continue
                                    key, _, value = part.partition(":")
                                    fields.setdefault(key, []).append(value.removeprefix(" "))
                                frame = []
                                if "data" not in fields:
                                    continue
                                event = RuntimeEvent.model_validate(json.loads("\n".join(fields["data"])))
                                if (fields.get("id") != [str(event.seq)]
                                        or fields.get("event") != [event.type]
                                        or event.run_id != run_id):
                                    raise CollectionFault("SSE_IDENTITY_MISMATCH")
                                if event.seq <= cursor:
                                    if seen.get(event.seq) != event:
                                        raise CollectionFault("SSE_REPLAY_CONFLICT")
                                    continue
                                if event.seq != cursor + 1:
                                    raise CollectionFault("SSE_SEQUENCE_GAP")
                                count += 1
                                if count > self.max_events:
                                    raise CollectionFault("SSE_EVENTS_EXCEEDED")
                                cursor = event.seq
                                seen[cursor] = event
                                yield event
                                if event.type in TERMINALS:
                                    return
                except httpx.HTTPStatusError as error:
                    raise CollectionFault(f"SSE_REJECTED_{error.response.status_code}") from None
                except httpx.TransportError:
                    pass
                if attempt < self.reconnects and time.monotonic() < deadline:
                    await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
            raise CollectionFault("SSE_TERMINAL_MISSING")
