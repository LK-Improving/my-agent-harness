"""摘要暂存到期拒绝与父失败失效定义；未运行。"""

import asyncio

import pytest

from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from tests.integration.test_context_pipeline_postgres import context_case as context_case

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("context_case", ["persist_expire"], indirect=True)
async def test_expired_candidate_is_not_refreshed_or_committed(context_case):
    store, _, owner, run, _, _, _, pipeline, gateway, transport, messages = context_case
    built = await pipeline.build(run, owner, messages, JsonSnapshot.capture([]))
    async with store._transaction() as conn:
        before = await store._one(conn, "SELECT expires_at,state FROM harness.summary_candidates WHERE run_id=%s", (run.run_id,))
    await asyncio.sleep(1.05)
    with pytest.raises(RuntimeFault, match="SUMMARY_STAGE_UNAVAILABLE"):
        await pipeline.summaries.stage(run, owner, built.candidate)
    with pytest.raises(RuntimeFault, match="SUMMARY_STAGE_UNAVAILABLE"):
        await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    assert transport.calls == 1
    async with store._transaction() as conn:
        assert await store._one(conn, "SELECT expires_at,state FROM harness.summary_candidates WHERE run_id=%s", (run.run_id,)) == before
        assert (await store._one(conn, "SELECT revision,summary_cursor_hash FROM harness.sessions WHERE session_id=%s", (run.session_id,))) == {"revision": 3, "summary_cursor_hash": None}
    await store.fail(owner, run.run_id, "STAGE_EXPIRED")
    async with store._transaction() as conn:
        assert (await store._one(conn, "SELECT state FROM harness.summary_candidates WHERE run_id=%s", (run.run_id,)))["state"] == "invalid"
