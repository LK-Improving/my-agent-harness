"""保留边界、管理API及固定trace文件定义；本批不运行。"""

from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from harness.adapters.cleanup_copies import ManagedTraceFiles, OfficialCheckpointCopies
from harness.bootstrap.api import create_app
from harness.contracts.retention import CleanupRequest, CleanupView, RetentionPolicy
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from tests.unit.test_knowledge_runtime import values


def test_retention_windows_and_exact_targets_are_bounded():
    policy = RetentionPolicy()
    assert (policy.session_seconds, policy.event_seconds, policy.trace_seconds, policy.private_seconds) == (
        30 * 86400, 7 * 86400, 7 * 86400, 90 * 86400)
    for update in ({"private_seconds": 1}, {"max_attempts": 4}, {"max_runs": 1001}, {"max_jobs": 1001}):
        with pytest.raises(ValidationError):
            RetentionPolicy.model_validate({**policy.model_dump(), **update})
    for update in ({"kind": "artifact"}, {"kind": "trace", "reason": "delete"},
                   {"kind": "session", "artifact_ref": "artifact-" + "a" * 32}):
        with pytest.raises(ValidationError):
            CleanupRequest(request_id="test", expected_revision=0, **update)
    for thread in ("../other", "harness-" + "a" * 63):
        with pytest.raises(RuntimeFault, match="CLEANUP_TARGET_INVALID"):
            OfficialCheckpointCopies._threads(JsonSnapshot.capture({"kind": "session", "threads": [thread]}))


async def test_managed_trace_cleanup_is_exact_idempotent_and_preserves_other_files(tmp_path):
    traces = ManagedTraceFiles(tmp_path)
    directory = tmp_path / ".runtime" / "traces"
    directory.mkdir(parents=True)
    thread = "harness-" + "a" * 64
    target = directory / (thread + ".jsonl")
    target.write_text('{"diagnostic":"合成诊断"}\n', encoding="utf-8")
    unrelated = directory / "other.jsonl"
    unrelated.write_text("保留", encoding="utf-8")
    assert not await traces.absent((thread,))
    await traces.erase((thread,))
    await traces.erase((thread,))
    assert await traces.absent((thread,)) and unrelated.read_text("utf-8") == "保留"
    with pytest.raises(RuntimeFault, match="CLEANUP_THREAD_INVALID"):
        await traces.erase(("../other",))


async def test_cleanup_api_only_registers_and_returns_metadata():
    run, *_ = values()
    calls = []

    class Manager:
        async def request(self, actor, session_id, request):
            calls.append((actor, session_id, request))
            return CleanupView(job_id="job-1", session_id=session_id, kind=request.kind,
                state="blocked", phase="copies", attempts=0, reasons=("unresolved",),
                revoked_at=datetime.now(UTC), request_hash="a" * 64)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(object(), lambda token: run.principal,
        retention=Manager())), base_url="http://test") as client:
        response = await client.post("/api/v1/sessions/session-1/cleanup-requests", json={
            "request_id": "delete", "kind": "session", "reason": "delete", "expected_revision": 0})
    assert response.status_code == 200 and len(calls) == 1
    assert response.json()["state"] == "blocked" and response.json()["revoked_at"] is not None
    assert not {"request", "target", "body", "candidate", "decision", "principal"} & response.json().keys()


async def test_loaded_sse_batch_stops_after_event_floor_moves():
    run, *_ = values()

    class Store:
        checks, released = 0, False
        async def admit_read(self, actor): pass
        async def acquire_sse(self, actor): return "lease"
        async def release_sse(self, lease): self.released = True
        async def renew_sse(self, actor, lease): pass
        async def get_run(self, actor, run_id): return type("View", (), {"status": "running"})()
        async def events(self, actor, run_id, after):
            return [{"seq": 1, "type": "answer.delta", "delta": "允许"},
                    {"seq": 2, "type": "answer.delta", "delta": "清理后禁止回放"}]
        async def validate_event_cursor(self, actor, run_id, after):
            self.checks += 1
            if self.checks > 1:
                raise RuntimeFault("EVENT_CURSOR_EXPIRED", 410)

    store = Store()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(store, lambda token: run.principal)),
                                 base_url="http://test") as client:
        response = await client.get("/api/v1/runs/run-1/events")
    assert "允许" in response.text and "清理后禁止回放" not in response.text and store.released
