"""API 组合根；只接收请求和回放事件，不创建或执行图。"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, Header, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from harness.contracts.activation import ActivationRequest
from harness.contracts.artifacts import ArtifactDeletionRequest, ArtifactRead, ArtifactReader, ArtifactRetention
from harness.contracts.hashing import RunRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.evaluation_contexts import EvaluationRevocationRequest
from harness.contracts.retention import CleanupRequest, RetentionManager
from harness.contracts.runtime import (
    CancelRequest,
    EvaluationRequest,
    Principal,
    RuntimeFault,
    RuntimeStore,
    SessionRequest,
)
from harness.contracts.specs import Identifier
from harness.contracts.writes import (
    ApprovalDecisionRequest,
    ReconcileRequest,
    WriteLedger,
    WriteOperation,
)


class BodyLimitMiddleware:
    """在 JSON 解析前限制实际字节数，包含没有 Content-Length 的分块请求。"""

    def __init__(self, app: ASGIApp, max_bytes: int = 262144) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks: list[Message] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                response = JSONResponse({"code": "REQUEST_TOO_LARGE"}, status_code=413)
                await response(scope, receive, send)
                return
            chunks.append(message)
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if chunks:
                return chunks.pop(0)
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    store: RuntimeStore,
    authenticate: Callable[[str | None], Principal],
    writes: WriteLedger | None = None,
    reconcile: Callable[[Principal, str, ReconcileRequest], Awaitable[WriteOperation]]
    | None = None,
    model_ready: Callable[[], bool] | None = None,
    artifacts: ArtifactReader | None = None,
    artifact_retention: ArtifactRetention | None = None,
    retention: RetentionManager | None = None,
    activate: Callable[[Principal, str, ActivationRequest], Awaitable[dict[str, str]]] | None = None,
    close_observations: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if close_observations is not None:
                await close_observations()

    app = FastAPI(title="Agent Harness", version="0.1.0", debug=False, lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(RuntimeFault)
    async def runtime_error(request: Request, error: RuntimeFault) -> JSONResponse:
        return JSONResponse({"code": error.code}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: RequestValidationError) -> JSONResponse:
        # FastAPI 默认错误包含原始 input，这里只公开字段位置和错误类型。
        return JSONResponse(
            {
                "code": "INVALID_REQUEST",
                "fields": [
                    {"loc": list(item["loc"]), "type": item["type"]} for item in error.errors()
                ],
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"code": "DEPENDENCY_OR_INTERNAL_FAILURE"}, status_code=503)

    async def principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
        return await asyncio.to_thread(authenticate, authorization)

    Actor = Annotated[Principal, Depends(principal)]
    ResourceId = Annotated[Identifier, Path()]
    ArtifactId = Annotated[str, Path(pattern=r"^artifact-[a-f0-9]{32}$")]

    @app.post("/api/v1/sessions/{session_id}/cleanup-requests")
    async def cleanup_request(session_id: ResourceId, body: CleanupRequest, actor: Actor) -> Any:
        if retention is None:
            raise RuntimeFault("RETENTION_MANAGER_NOT_INSTALLED", 503)
        return await retention.request(actor, session_id, body)

    @app.get("/api/v1/cleanup-requests/{job_id}")
    async def cleanup_view(job_id: ResourceId, actor: Actor) -> Any:
        if retention is None:
            raise RuntimeFault("RETENTION_MANAGER_NOT_INSTALLED", 503)
        await store.admit_read(actor)
        return await retention.inspect(actor, job_id)

    @app.get("/api/v1/sessions/{session_id}/artifacts/{artifact_ref}/retention")
    async def artifact_retention_view(session_id: ResourceId, artifact_ref: ArtifactId, actor: Actor,
                                       response: Response) -> Any:
        if artifact_retention is None:
            raise RuntimeFault("ARTIFACT_RETENTION_NOT_INSTALLED", 503)
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await artifact_retention.inspect(actor, session_id, artifact_ref)

    @app.post("/api/v1/sessions/{session_id}/artifacts/{artifact_ref}/deletion-requests", status_code=202)
    async def artifact_deletion(session_id: ResourceId, artifact_ref: ArtifactId, body: ArtifactDeletionRequest,
                                actor: Actor, response: Response) -> Any:
        if artifact_retention is None:
            raise RuntimeFault("ARTIFACT_RETENTION_NOT_INSTALLED", 503)
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await artifact_retention.request_deletion(actor, session_id, artifact_ref, body)

    @app.get("/api/v1/sessions/{session_id}/artifacts/{artifact_ref}")
    async def artifact(session_id: ResourceId, artifact_ref: ResourceId, actor: Actor, response: Response,
                       offset: int = 0, limit: int = 2000) -> Any:
        if artifacts is None:
            raise RuntimeFault("ARTIFACT_READER_NOT_INSTALLED", 503)
        from pydantic import ValidationError

        try:
            page = ArtifactRead(artifact_ref=artifact_ref, offset=offset, limit=limit)
        except ValidationError:
            raise RuntimeFault("ARTIFACT_PAGE_INVALID", 422) from None
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await artifacts.read(actor, session_id, page)

    @app.get("/api/health/live")
    async def live() -> dict[str, bool]:
        return {"live": True}

    @app.get("/api/health/ready")
    async def ready() -> JSONResponse:
        health = await store.health()
        ok = all(
            health.get(key) is True
            for key in ("database", "schema", "worker_registered", "worker_fresh", "checkpoint")
        )
        if model_ready is not None:
            ok = ok and await asyncio.to_thread(model_ready)
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.post("/api/v1/agents/{agent_id}/releases", status_code=201)
    async def release(agent_id: ResourceId, body: ReleaseManifest, actor: Actor) -> dict[str, str]:
        if agent_id != body.spec.agent_id:
            raise RuntimeFault("AGENT_ID_MISMATCH", 422)
        return {
            "release_id": body.spec.release_id,
            "manifest_hash": await store.register_release(actor, body),
        }

    @app.post("/api/v1/agents/{agent_id}/activations")
    async def activation(agent_id: ResourceId, body: ActivationRequest, actor: Actor) -> dict[str, str]:
        if activate is None:
            raise RuntimeFault("ACTIVATION_NOT_CONFIGURED", 503)
        return await activate(actor, agent_id, body)

    @app.post("/api/v1/evaluation-contexts", status_code=201)
    async def evaluation(body: EvaluationRequest, actor: Actor) -> Any:
        return await store.create_evaluation(actor, body)

    @app.get("/api/v1/evaluation-contexts/{context_id}")
    async def evaluation_view(context_id: ResourceId, actor: Actor, response: Response) -> Any:
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await store.inspect_evaluation(actor, context_id)

    @app.post("/api/v1/evaluation-contexts/{context_id}/revocations")
    async def evaluation_revoke(context_id: ResourceId, body: EvaluationRevocationRequest,
                                actor: Actor, response: Response) -> Any:
        response.headers["Cache-Control"] = "no-store"
        return await store.revoke_evaluation(actor, context_id, body)

    @app.post("/api/v1/sessions", status_code=201)
    async def session(body: SessionRequest, actor: Actor) -> dict[str, str | int]:
        return {"session_id": await store.create_session(actor, body), "revision": 0}

    @app.post("/api/v1/sessions/{session_id}/runs", status_code=202)
    async def run(session_id: ResourceId, body: RunRequest, actor: Actor) -> dict[str, Any]:
        result = await store.create_run(actor, session_id, body)
        return {
            **result.model_dump(mode="json"),
            "events_url": f"/api/v1/runs/{result.run_id}/events",
        }

    @app.get("/api/v1/runs/{run_id}")
    async def read_run(run_id: ResourceId, actor: Actor) -> Any:
        await store.admit_read(actor)
        return await store.get_run(actor, run_id)

    @app.get("/api/v1/runs/{run_id}/evaluation-evidence")
    async def evaluation_run_evidence(run_id: ResourceId, actor: Actor, response: Response) -> Any:
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await store.evaluation_evidence(actor, run_id)

    @app.get("/api/v1/runs/{run_id}/evaluation-judge-responses/{call_id}")
    async def evaluation_judge_response(run_id: ResourceId, call_id: ResourceId,
                                        actor: Actor, response: Response) -> Any:
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await store.judge_model_response(actor, run_id, call_id)

    @app.get("/api/v1/runs/{run_id}/evaluation-judge-material")
    async def evaluation_judge_material(run_id: ResourceId, actor: Actor, response: Response) -> Any:
        await store.admit_read(actor)
        response.headers["Cache-Control"] = "no-store"
        return await store.judge_source_material(actor, run_id)

    @app.get("/api/v1/approvals")
    async def approvals(actor: Actor, after: str = "", limit: int = 50) -> Any:
        if writes is None:
            raise RuntimeFault("WRITE_CAPABILITY_NOT_INSTALLED", 503)
        if len(after) > 128:
            raise RuntimeFault("INVALID_PAGE_CURSOR", 422)
        await store.admit_read(actor)
        return {"items": await writes.list_approvals(actor, after, limit)}

    @app.post("/api/v1/approvals/{approval_id}/decisions")
    async def decision(approval_id: ResourceId, body: ApprovalDecisionRequest, actor: Actor) -> Any:
        if writes is None:
            raise RuntimeFault("WRITE_CAPABILITY_NOT_INSTALLED", 503)
        return await writes.decide_approval(actor, approval_id, body)

    @app.post("/api/v1/runs/{run_id}/reconciliations")
    async def reconciliation(run_id: ResourceId, body: ReconcileRequest, actor: Actor) -> Any:
        if reconcile is None:
            raise RuntimeFault("RECONCILIATION_ADAPTER_NOT_INSTALLED", 503)
        operation = await reconcile(actor, run_id, body)
        # 不把内部主体、原始参数和供应商正文返回给核对接口。
        return {
            "operation_id": operation.operation_id,
            "state": operation.state,
            "output_error": operation.output_error,
        }

    @app.post("/api/v1/runs/{run_id}/cancel")
    async def cancel(run_id: ResourceId, body: CancelRequest, actor: Actor) -> Any:
        return await store.cancel_run(actor, run_id, body)

    @app.get("/api/v1/runs/{run_id}/events")
    async def events(
        run_id: ResourceId,
        request: Request,
        actor: Actor,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        if last_event_id is not None and (
            not last_event_id.isascii() or not last_event_id.isdecimal() or len(last_event_id) > 18
        ):
            raise RuntimeFault("INVALID_EVENT_CURSOR", 422)
        cursor = int(last_event_id or "0")
        await store.admit_read(actor)
        # 建立响应前先检查权限与过期游标，保证可返回标准 403/404/410。
        initial = await store.events(actor, run_id, cursor)
        lease = await store.acquire_sse(actor)

        async def stream() -> AsyncIterator[str]:
            nonlocal cursor
            batch = initial
            try:
                while not await request.is_disconnected():
                    current_actor = await asyncio.to_thread(
                        authenticate, request.headers.get("authorization")
                    )
                    await store.renew_sse(current_actor, lease)
                    view = await store.get_run(current_actor, run_id)
                    for event in batch:
                        # 已加载的同一批事件也逐条复查，撤权后不继续公开缓存正文。
                        current_actor = await asyncio.to_thread(
                            authenticate, request.headers.get("authorization")
                        )
                        await store.renew_sse(current_actor, lease)
                        await store.get_run(current_actor, run_id)
                        validate_cursor = getattr(store, "validate_event_cursor", None)
                        if validate_cursor is not None:
                            await validate_cursor(current_actor, run_id, cursor)
                        cursor = int(str(event["seq"]))
                        yield (
                            f"id: {cursor}\nevent: {event['type']}\n"
                            f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        )
                        if event["type"] in {
                            "run.completed",
                            "run.failed",
                            "run.cancelled",
                            "run.expired",
                        }:
                            return
                    if (
                        view.status in {"succeeded", "failed", "cancelled", "expired", "partial"}
                        and not batch
                    ):
                        return
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1)
                    batch = await store.events(current_actor, run_id, cursor)
            except RuntimeFault:
                return
            finally:
                try:
                    # ASGI断线会取消整个生成器；屏蔽该取消，限时释放只读连接名额。
                    with anyio.CancelScope(shield=True):
                        async with asyncio.timeout(5):
                            await store.release_sse(lease)
                except Exception:
                    # 连接占用有独立短租期；此处失败不释放运行Worker执行权。
                    pass

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return app
