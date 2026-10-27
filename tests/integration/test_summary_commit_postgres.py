"""真实PG摘要暂存/成功提交/CAS与故障回滚定义；全部未运行。"""

import pytest
from psycopg.types.json import Jsonb

from harness.adapters.summaries_postgres import PostgresSummaryStore
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.runtime import CancelRequest, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from tests.integration.test_context_pipeline_postgres import candidate, context_case as context_case

pytestmark = [pytest.mark.integration, pytest.mark.parametrize("context_case", ["persist"], indirect=True)]


async def final_answer(case):
    _, _, owner, run, _, _, _, _, gateway, _, messages = case
    result = await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    return candidate(run, answer=result.content, call=result.model_call_id, attempt=result.model_attempt_id, draft=result.draft_id)


async def state(case):
    store, _, _, run, *_ = case
    async with store._transaction() as conn:
        session = await store._one(conn, "SELECT revision,summary_cursor_hash FROM harness.sessions WHERE session_id=%s", (run.session_id,))
        staged = await (await conn.execute("SELECT state,expires_at,candidate,candidate_hash FROM harness.summary_candidates WHERE run_id=%s", (run.run_id,))).fetchone()
        committed = await (await conn.execute("SELECT covered_revision,committed_revision,cursor_hash FROM harness.session_summaries WHERE session_id=%s", (run.session_id,))).fetchone()
        return session, staged, committed


async def test_stage_is_not_visible_until_atomic_success_and_replay_is_idempotent(context_case):
    store, _, owner, run, _, _, _, pipeline, _, transport, _ = context_case
    answer = await final_answer(context_case)
    before, staged, committed = await state(context_case)
    assert before == {"revision": 3, "summary_cursor_hash": None} and staged["state"] == "staged" and committed is None
    assert await pipeline.summaries.load_committed(run, owner) is None
    await store.complete(owner, run, answer)
    session, stage, summary = await state(context_case)
    assert session["revision"] == 4 and stage["state"] == "committed"
    assert summary["covered_revision"] == 1 and summary["committed_revision"] == 4
    assert summary["cursor_hash"] == session["summary_cursor_hash"]
    await store.complete(owner, run, answer)
    assert await state(context_case) == (session, stage, summary) and transport.calls == 2
    async with store._transaction() as conn:
        assert (await store._one(conn, "SELECT count(*) AS n FROM harness.commits WHERE run_id=%s", (run.run_id,)))["n"] == 1
        assert (await store._one(conn, "SELECT count(*) AS n FROM harness.turns WHERE run_id=%s", (run.run_id,)))["n"] == 1


async def test_completion_event_failure_rolls_back_summary_cursor_turn_and_commit(context_case, monkeypatch):
    store, _, owner, run, _, _, _, _, _, transport, _ = context_case
    answer = await final_answer(context_case)
    await store.stage_answer(owner, run, answer)
    before = await state(context_case)
    original = store._event

    async def broken(conn, run_id, event_type, payload):
        await original(conn, run_id, event_type, payload)
        if run_id == run.run_id and event_type == "run.completed":
            raise RuntimeError("AFTER_COMPLETION_EVENT")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_event", broken)
        with pytest.raises(RuntimeError, match="AFTER_COMPLETION_EVENT"):
            await store.complete(owner, run, answer)
    assert await state(context_case) == before
    async with store._transaction() as conn:
        for table in ("harness.turns", "harness.commits"):
            # 表名固定测试常量；不接收外部SQL或用户标识。
            assert await (await conn.execute(f"SELECT 1 FROM {table} WHERE run_id=%s", (run.run_id,))).fetchone() is None
        assert await (await conn.execute("SELECT 1 FROM harness.events WHERE run_id=%s AND type='run.completed'", (run.run_id,))).fetchone() is None
    await store.complete(owner, run, answer)
    assert (await state(context_case))[0]["revision"] == 4 and transport.calls == 2


@pytest.mark.parametrize("conflict", ["revision", "cursor"])
async def test_source_revision_or_cursor_conflict_never_publishes_candidate(context_case, conflict):
    store, _, owner, run, *_ = context_case
    answer = await final_answer(context_case)
    async with store._transaction() as conn:
        if conflict == "revision":
            await conn.execute("UPDATE harness.sessions SET revision=revision+1 WHERE session_id=%s", (run.session_id,))
        else:
            await conn.execute("UPDATE harness.sessions SET summary_cursor_hash=%s WHERE session_id=%s", ("f" * 64, run.session_id))
    with pytest.raises(RuntimeFault, match="SUMMARY_SOURCE_CAS_CONFLICT"):
        await store.complete(owner, run, answer)
    _, staged, summary = await state(context_case)
    assert staged["state"] == "staged" and summary is None
    async with store._transaction() as conn:
        # 还原本用例注入的元数据冲突；不修改其他会话。
        await conn.execute("UPDATE harness.sessions SET revision=3,summary_cursor_hash=NULL WHERE session_id=%s", (run.session_id,))


@pytest.mark.parametrize("failure", ["generation", "parent_failed", "cancelled"])
async def test_failed_or_cancelled_parent_never_exposes_summary_to_next_run(context_case, failure):
    store, actor, owner, run, _, _, _, _, gateway, transport, messages = context_case
    if failure == "generation":
        transport.fail_summary = True
        with pytest.raises(RuntimeFault):
            await gateway.complete(run, owner, messages, JsonSnapshot.capture([]))
    else:
        await final_answer(context_case)
    if failure == "cancelled":
        view = await store.get_run(actor, run.run_id)
        await store.cancel_run(actor, run.run_id, CancelRequest(request_id="cancel", expected_run_revision=view.revision))
    await store.fail(owner, run.run_id, "CONTROLLED_FAILURE")
    session, staged, summary = await state(context_case)
    assert session == {"revision": 3, "summary_cursor_hash": None} and summary is None
    if failure == "generation":
        assert staged is None
    else:
        assert staged["state"] == "invalid"
    await store.finish_slot(owner)
    await store.create_run(actor, run.session_id, RunRequest(request_id="after-failure", message="下一轮纠错", expected_revision=3))
    next_run = await store.claim(owner)
    assert next_run is not None
    try:
        assert await PostgresSummaryStore(store).load_committed(next_run, owner) is None
    finally:
        await store.fail(owner, next_run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)


async def test_next_run_reads_only_committed_summary_and_rechecks_source_permissions(context_case, monkeypatch):
    store, actor, owner, run, _, profile, source, _, _, transport, _ = context_case
    answer = await final_answer(context_case)
    await store.complete(owner, run, answer)
    await store.finish_slot(owner)
    await store.create_run(actor, run.session_id, RunRequest(request_id="after-success", message="主题切换并纠正旧条件", expected_revision=4))
    next_run = await store.claim(owner)
    assert next_run is not None
    try:
        summaries = PostgresSummaryStore(store)
        committed = await summaries.load_committed(next_run, owner)
        assert committed.covered_revision == 1 and committed.committed_revision == 4 and transport.calls == 2
        for update in ({"tenant_id": "other"}, {"subject_id": "other"}, {"application_id": "other"}):
            with pytest.raises(RuntimeFault):
                await summaries.load_committed(next_run.model_copy(update={"principal": actor.model_copy(update=update)}), owner)
        snapshot = await source.load(next_run, owner, profile)
        target = snapshot.turns[0].run_id
        original = store._authorize_public_resource

        async def revoked(conn, principal, row, session):
            if row["run_id"] == target:
                raise RuntimeFault("COVERED_SOURCE_REVOKED", 403)
            await original(conn, principal, row, session)

        monkeypatch.setattr(store, "_authorize_public_resource", revoked)
        with pytest.raises(RuntimeFault, match="COVERED_SOURCE_REVOKED"):
            await summaries.load_committed(next_run, owner)
    finally:
        await store.fail(owner, next_run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)


async def test_candidate_must_match_real_summary_receipt_and_cannot_be_replaced(context_case):
    store, _, owner, run, _, _, _, pipeline, _, _, _ = context_case
    await final_answer(context_case)
    before = await state(context_case)
    from harness.contracts.context import SummaryCandidate

    value = SummaryCandidate.model_validate(before[1]["candidate"])
    forged = value.model_copy(update={"summary": "捏造已确认事实"})
    forged = forged.model_copy(update={"summary_hash": canonical_hash({"purpose": forged.purpose.model_dump(mode="json"), "summary": forged.summary})})
    with pytest.raises(RuntimeFault, match="SUMMARY_MODEL_CONTENT_MISMATCH"):
        await pipeline.summaries.stage(run, owner, forged)
    await pipeline.summaries.stage(run, owner, value)
    assert await state(context_case) == before


@pytest.mark.parametrize("outcome", ["commit", "fail"])
async def test_later_summary_cursor_advances_only_with_the_later_success(context_case, outcome):
    store, actor, owner, run, _, _, _, _, gateway, transport, _ = context_case
    first = await final_answer(context_case)
    await store.complete(owner, run, first)
    old_cursor = (await state(context_case))[0]["summary_cursor_hash"]
    await store.finish_slot(owner)
    await store.create_run(actor, run.session_id, RunRequest(request_id="later-summary", message="纠正条件并切换主题", expected_revision=4))
    next_run = await store.claim(owner)
    assert next_run is not None
    try:
        messages = JsonSnapshot.capture([{"role": "system", "content": "固定规则"}, *next_run.history.as_object_array(),
            {"role": "user", "content": next_run.request.message}])
        result = await gateway.complete(next_run, owner, messages, JsonSnapshot.capture([]))
        answer = candidate(next_run, answer=result.content, call=result.model_call_id,
            attempt=result.model_attempt_id, draft=result.draft_id)
        if outcome == "commit":
            await store.complete(owner, next_run, answer)
        else:
            await store.fail(owner, next_run.run_id, "LATER_PARENT_FAILED")
        async with store._transaction() as conn:
            session = await store._one(conn, "SELECT revision,summary_cursor_hash FROM harness.sessions WHERE session_id=%s", (run.session_id,))
            summary = await store._one(conn, "SELECT covered_revision,committed_revision FROM harness.session_summaries WHERE session_id=%s", (run.session_id,))
            staged = await store._one(conn, "SELECT state FROM harness.summary_candidates WHERE run_id=%s", (next_run.run_id,))
        assert transport.calls == 4
        if outcome == "commit":
            assert session["revision"] == 5 and session["summary_cursor_hash"] != old_cursor
            assert summary == {"covered_revision": 2, "committed_revision": 5} and staged["state"] == "committed"
        else:
            assert session == {"revision": 4, "summary_cursor_hash": old_cursor}
            assert summary == {"covered_revision": 1, "committed_revision": 4} and staged["state"] == "invalid"
    finally:
        await store.fail(owner, next_run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)


@pytest.mark.parametrize("tamper", ["source_turn", "model_receipt"])
async def test_source_or_model_receipt_change_after_staging_blocks_commit(context_case, tamper):
    store, _, owner, run, *_ = context_case
    answer = await final_answer(context_case)
    _, staged, _ = await state(context_case)
    if tamper == "source_turn":
        target = staged["candidate"]["purpose"]["covered_turn_ids"][0]
        async with store._transaction() as conn:
            saved = await store._one(conn, "SELECT messages FROM harness.turns WHERE run_id=%s", (target,))
            await conn.execute("UPDATE harness.turns SET messages=%s WHERE run_id=%s", (Jsonb([
                {"role": "user", "content": "已改变的原始来源"}, {"role": "assistant", "content": "新正文"}]), target))
    else:
        target = staged["candidate"]["model_call_id"]
        async with store._transaction() as conn:
            saved = await store._one(conn, "SELECT result FROM harness.model_calls WHERE model_call_id=%s", (target,))
            changed = {**saved["result"], "content": '{"summary":"被篡改的摘要"}'}
            await conn.execute("UPDATE harness.model_calls SET result=%s WHERE model_call_id=%s", (Jsonb(changed), target))
    try:
        with pytest.raises(RuntimeFault):
            await store.complete(owner, run, answer)
        session, stage, committed = await state(context_case)
        assert session == {"revision": 3, "summary_cursor_hash": None} and stage["state"] == "staged" and committed is None
    finally:
        async with store._transaction() as conn:
            if tamper == "source_turn":
                await conn.execute("UPDATE harness.turns SET messages=%s WHERE run_id=%s", (Jsonb(saved["messages"]), target))
            else:
                await conn.execute("UPDATE harness.model_calls SET result=%s WHERE model_call_id=%s", (Jsonb(saved["result"]), target))
