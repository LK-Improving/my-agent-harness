"""上下文预算错误与父账本/公开错误路径的真实PG定义；本批不运行。"""

import httpx
import pytest

from harness.bootstrap.api import create_app
from harness.contracts.hashing import RunRequest
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.runtime.worker import Worker
from tests.integration.test_context_pipeline_postgres import context_case as context_case

pytestmark = pytest.mark.integration


async def test_ledger_distinguishes_input_capacity_from_cumulative_budget_without_reserving(context_case):
    store, _, owner, run, execution, *_ = context_case
    oversized = JsonSnapshot.capture([{"role": "user", "content": "x" * execution.budget.max_input_tokens}])
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED") as error:
        await store.reserve_model(run, owner, oversized, JsonSnapshot.capture([]), execution)
    assert error.value.status == 429
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT model_steps,model_reserved_tokens FROM harness.runs WHERE run_id=%s", (run.run_id,))
        assert row == {"model_steps": 0, "model_reserved_tokens": 0}
        await conn.execute("UPDATE harness.runs SET model_reserved_tokens=%s WHERE run_id=%s", (execution.budget.max_total_tokens - 1, run.run_id))
    with pytest.raises(RuntimeFault, match="MODEL_BUDGET_EXHAUSTED"):
        await store.reserve_model(run, owner, JsonSnapshot.capture([{"role": "user", "content": "短输入"}]), JsonSnapshot.capture([]), execution)
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT model_steps,model_reserved_tokens FROM harness.runs WHERE run_id=%s", (run.run_id,))
        assert row == {"model_steps": 0, "model_reserved_tokens": execution.budget.max_total_tokens - 1}
        assert await (await conn.execute("SELECT 1 FROM harness.model_calls WHERE run_id=%s", (run.run_id,))).fetchone() is None


async def test_uncompressible_current_task_surfaces_same_error_in_result_and_sse(context_case):
    store, actor, owner, original, _, _, _, _, gateway, transport, _ = context_case
    await store.fail(owner, original.run_id, "PREPARE_BOUNDARY_CASE")
    await store.finish_slot(owner)
    queued = await store.create_run(actor, original.session_id, RunRequest(request_id="oversized-context",
        message="x" * 13000, expected_revision=3))
    run = await store.claim(owner)
    assert run is not None and run.run_id == queued.run_id

    class Engine:
        def observed_manifest_hash(self, release):
            return release.content_hash()

        async def execute(self, current):
            messages = JsonSnapshot.capture([{"role": "system", "content": "固定规则"}, *current.history.as_object_array(),
                {"role": "user", "content": current.request.message}])
            await gateway.complete(current, owner, messages, JsonSnapshot.capture([]))
            raise AssertionError("超限输入不应得到候选")

    try:
        with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED") as error:
            await Worker(store, Engine(), identity=owner)._execute(run)
        # 与Worker.serve相同的分类收尾；不启动后台serve或额外执行者。
        await store.fail(owner, run.run_id, error.value.code)
        assert transport.calls == 0
        app = create_app(store, lambda token: actor)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = await client.get(f"/api/v1/runs/{run.run_id}")
            assert result.status_code == 200 and result.json()["status"] == "failed"
            assert result.json()["error_code"] == "CONTEXT_BUDGET_EXCEEDED" and result.json()["result"] is None
            async with client.stream("GET", f"/api/v1/runs/{run.run_id}/events") as response:
                body = (await response.aread()).decode("utf-8")
                assert response.status_code == 200 and "CONTEXT_BUDGET_EXCEEDED" in body
        async with store._transaction() as conn:
            session = await store._one(conn, "SELECT revision,summary_cursor_hash FROM harness.sessions WHERE session_id=%s", (run.session_id,))
            assert session == {"revision": 3, "summary_cursor_hash": None}
            counters = await store._one(conn, "SELECT model_steps,model_reserved_tokens FROM harness.runs WHERE run_id=%s", (run.run_id,))
            assert counters == {"model_steps": 0, "model_reserved_tokens": 0}
            assert await (await conn.execute("SELECT 1 FROM harness.summary_candidates WHERE run_id=%s", (run.run_id,))).fetchone() is None
    finally:
        await store.fail(owner, run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
