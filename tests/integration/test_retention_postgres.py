"""隔离真实PG清理事务/重试/撤权定义；本批不运行、不声称实际删净。"""

import asyncio
from uuid import uuid4

import httpx
import pytest
from psycopg.types.json import Jsonb
from psycopg.errors import RaiseException

from harness.adapters.retention_postgres import PostgresRetention
from harness.bootstrap.api import create_app
from harness.contracts.hashing import RunRequest
from harness.contracts.retention import CleanupRequest, RetentionPolicy
from harness.contracts.runtime import RuntimeFault

pytestmark = pytest.mark.integration


class Copies:
    calls = 0
    present = False
    fail = False

    async def erase(self, target):
        self.calls += 1
        if self.fail:
            raise RuntimeFault("CONTROLLED_COPY_FAILURE", 503)
        self.present = False

    async def absent(self, target):
        return not self.present


async def seed(prepared, *, days=100, policy=None):
    store, actor, manifest, _ = prepared
    session_id, run_id = uuid4().hex, uuid4().hex
    policy = policy or RetentionPolicy()
    request = RunRequest(request_id="seed", message="合成私有原始输入", expected_revision=0)
    async with store._transaction() as conn:
        # 仅插入本用例合成记录，不改已有会话的不可变创建时间或策略。
        await conn.execute("""INSERT INTO harness.sessions(session_id,tenant_id,subject_id,application_id,agent_id,
            release_id,revision,created_at,retention_policy) VALUES(%s,%s,%s,%s,%s,%s,1,
            clock_timestamp()-%s*interval '1 day',%s)""", (session_id, actor.tenant_id, actor.subject_id,
            actor.application_id, manifest.spec.agent_id, manifest.spec.release_id, days, Jsonb(policy.model_dump(mode="json"))))
        await conn.execute("""INSERT INTO harness.runs(run_id,session_id,request_id,request_hash,request,manifest_hash,
            environment_profile_hash,principal,status,event_seq,created_at,queue_deadline,expires_at,result)
            VALUES(%s,%s,'seed',%s,%s,%s,%s,%s,'succeeded',1,clock_timestamp()-%s*interval '1 day',
            clock_timestamp(),clock_timestamp(),%s)""", (run_id, session_id, request.business_hash(), Jsonb(request.model_dump(mode="json")),
            manifest.content_hash(), manifest.environment_profile_hash, Jsonb(actor.model_dump(mode="json")), days,
            Jsonb({"answer": "合成最终正文"})))
        await conn.execute("""INSERT INTO harness.events VALUES(%s,1,'run.completed',%s,
            clock_timestamp()-%s*interval '1 day')""", (run_id, Jsonb({"answer": "合成事件正文"}), days))
        commit = uuid4().hex
        await conn.execute("""INSERT INTO harness.commits(commit_id,run_id,candidate_hash,validation,created_at)
            VALUES(%s,%s,%s,'{}',clock_timestamp()-%s*interval '1 day')""", (commit, run_id, "a" * 64, days))
        await conn.execute("INSERT INTO harness.turns VALUES(%s,1,%s,%s)", (session_id, run_id,
            Jsonb([{"role": "user", "content": request.message}, {"role": "assistant", "content": "合成历史正文"}])))
    operator = actor.model_copy(update={"roles": actor.roles | {"retention_operator"},
        "resource_scopes": actor.resource_scopes | {f"session:{session_id}:retention"}})
    return store, actor, operator, session_id, run_id


async def test_session_cleanup_repeats_idempotently_and_keeps_minimal_audit(prepared):
    store, actor, operator, session, run_id = await seed(prepared)
    ref = "artifact-" + uuid4().hex
    async with store._transaction() as conn:
        await conn.execute("""INSERT INTO harness.artifacts(artifact_ref,run_id,session_id,operation_id,receipt_hash,
            media_type,body,content_hash,total_chars,total_bytes,expires_at) VALUES(%s,%s,%s,'source',%s,
            'text/plain','private',%s,7,7,clock_timestamp()-interval '1 day')""", (ref, run_id, session, "c" * 64, "d" * 64))
        await conn.execute("""INSERT INTO harness.artifact_deletion_requests
            (session_id,request_id,artifact_ref,request_hash,expected_content_hash,actor) VALUES(%s,'delete-ref',%s,%s,%s,%s)""",
            (session, ref, "e" * 64, "d" * 64, Jsonb(actor.model_dump(mode="json"))))
        await conn.execute("""INSERT INTO harness.extension_decisions VALUES(%s,'ext','1','fingerprint','before_model',
            '{}',%s,'{}','hash',clock_timestamp()-interval '100 day')""", (run_id, Jsonb({"private": "合成扩展参数"})))
    copies = Copies()
    manager = PostgresRetention(store, copies)
    request = CleanupRequest(request_id="delete", kind="session", reason="delete", expected_revision=1)
    job = await manager.request(actor, session, request)
    assert (await manager.request(actor, session, request)).job_id == job.job_id
    with pytest.raises(RuntimeFault, match="CLEANUP_REQUEST_CONFLICT"):
        await manager.request(actor, session, request.model_copy(update={"reason": "ttl"}))
    with pytest.raises(RuntimeFault, match="SESSION_REVOKED"):
        await store.get_run(actor, run_id)
    assert (await manager.advance(operator, job.job_id)).phase == "payloads"
    finished = await manager.advance(operator, job.job_id)
    assert finished.state == "completed" and copies.calls == 1
    assert (await manager.advance(operator, job.job_id)).state == "completed" and copies.calls == 1
    another = await manager.request(actor, session, request.model_copy(update={"request_id": "another-delete"}))
    assert (await manager.advance(operator, another.job_id)).state == "completed" and copies.calls == 1
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT request,result,event_floor,event_seq FROM harness.runs WHERE run_id=%s", (run_id,))
        assert row["request"] == {} and row["result"] is None and row["event_floor"] == row["event_seq"] == 1
        for table in ("events", "turns", "extension_decisions", "model_calls", "read_operations"):
            assert await (await conn.execute(f"SELECT 1 FROM harness.{table} WHERE run_id=%s", (run_id,))).fetchone() is None
        assert await (await conn.execute("SELECT 1 FROM harness.commits WHERE run_id=%s", (run_id,))).fetchone()
        artifact = await store._one(conn, "SELECT body,purged_at,content_hash,total_bytes FROM harness.artifacts WHERE artifact_ref=%s", (ref,))
        assert artifact["body"] is None and artifact["purged_at"] is not None and artifact["total_bytes"] == 7
        assert artifact["content_hash"] == "d" * 64
        assert await (await conn.execute("SELECT 1 FROM harness.artifact_deletion_requests WHERE artifact_ref=%s", (ref,))).fetchone()
        assert await (await conn.execute("SELECT 1 FROM harness.audit WHERE action='retention.completed' AND resource_id=%s", (job.job_id,))).fetchone()


async def test_events_cleanup_expires_cursor_without_deleting_final_snapshot(prepared):
    store, actor, operator, session, run_id = await seed(prepared, days=8)
    manager = PostgresRetention(store)
    job = await manager.request(actor, session, CleanupRequest(request_id="events", kind="events", expected_revision=1))
    assert (await manager.advance(operator, job.job_id)).state == "completed"
    assert (await store.get_run(actor, run_id)).result == {"answer": "合成最终正文"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(store, lambda token: actor)), base_url="http://test") as client:
        assert (await client.get(f"/api/v1/runs/{run_id}/events")).status_code == 410
        assert (await client.get(f"/api/v1/runs/{run_id}/events", headers={"Last-Event-ID": "1"})).status_code == 200


@pytest.mark.parametrize("barrier", ["hold", "evaluation", "model_unknown", "active", "private_window", "external_trace"])
async def test_protection_blocks_without_spending_attempt_or_restoring_access(prepared, barrier):
    policy = RetentionPolicy(external_trace_copies=barrier == "external_trace")
    store, actor, operator, session, run_id = await seed(prepared, days=1 if barrier == "private_window" else 100, policy=policy)
    async with store._transaction() as conn:
        if barrier == "hold":
            await conn.execute("UPDATE harness.sessions SET legal_hold=true WHERE session_id=%s", (session,))
        elif barrier == "evaluation":
            await conn.execute("UPDATE harness.sessions SET evaluation_context_id=(SELECT evaluation_context_id FROM harness.sessions WHERE session_id=%s) WHERE session_id=%s",
                               (prepared[3], session))
        elif barrier == "model_unknown":
            await conn.execute("INSERT INTO harness.model_calls(model_call_id,run_id,input_hash,request_fingerprint,request,state) VALUES(%s,%s,%s,%s,'{}','unknown')",
                               (uuid4().hex, run_id, "a" * 64, "b" * 64))
        elif barrier == "active":
            await conn.execute("UPDATE harness.sessions SET active_run_id=%s WHERE session_id=%s", (run_id, session))
    manager = PostgresRetention(store, Copies())
    job = await manager.request(actor, session, CleanupRequest(request_id="delete", kind="session", reason="delete", expected_revision=1))
    blocked = await manager.advance(operator, job.job_id)
    assert blocked.state == "blocked" and blocked.attempts == 0 and blocked.reasons and blocked.revoked_at is not None
    with pytest.raises(RuntimeFault, match="SESSION_REVOKED"):
        await store.get_run(actor, run_id)
    async with store._transaction() as conn:
        assert await (await conn.execute("SELECT 1 FROM harness.turns WHERE run_id=%s", (run_id,))).fetchone()


async def test_unknown_copy_outcome_requires_explicit_reconciliation_and_is_bounded(prepared):
    store, actor, operator, session, _ = await seed(prepared)
    copies = Copies()
    copies.present, copies.fail = True, True
    manager = PostgresRetention(store, copies)
    job = await manager.request(actor, session, CleanupRequest(request_id="delete", kind="session", reason="delete", expected_revision=1))
    for attempt in range(1, 4):
        result = await manager.advance(operator, job.job_id)
        assert result.state == "running" and result.attempts == attempt
        with pytest.raises(RuntimeFault, match="CLEANUP_RECONCILIATION_REQUIRED"):
            await manager.advance(operator, job.job_id)
        result = await manager.reconcile(operator, job.job_id)
        assert result.state == ("manual" if attempt == 3 else "retry")
    assert copies.calls == 3
    assert (await manager.advance(operator, job.job_id)).state == "manual" and copies.calls == 3
    replacement = await manager.request(actor, session, CleanupRequest(request_id="new-key", kind="session", reason="delete", expected_revision=1))
    assert (await manager.advance(operator, replacement.job_id)).state == "manual" and copies.calls == 3


async def test_payload_failure_rolls_back_content_cursor_and_completion_audit_then_retries(prepared, monkeypatch):
    store, actor, operator, session, run_id = await seed(prepared)
    manager = PostgresRetention(store, Copies())
    job = await manager.request(actor, session, CleanupRequest(request_id="delete", kind="session", reason="delete", expected_revision=1))
    await manager.advance(operator, job.job_id)
    original = store._audit

    async def fail_completion(conn, current, action, resource):
        if action == "retention.completed":
            await conn.execute("SELECT 1/0")
        await original(conn, current, action, resource)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_audit", fail_completion)
        assert (await manager.advance(operator, job.job_id)).state == "retry"
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT request,event_floor FROM harness.runs WHERE run_id=%s", (run_id,))
        assert row["request"]["message"] == "合成私有原始输入" and row["event_floor"] == 0
        assert await (await conn.execute("SELECT 1 FROM harness.turns WHERE run_id=%s", (run_id,))).fetchone()
    assert (await manager.advance(operator, job.job_id)).state == "completed"


async def test_scope_and_identity_changes_cannot_manage_another_session(prepared):
    store, actor, operator, session, _ = await seed(prepared)
    manager = PostgresRetention(store)
    job = await manager.request(actor, session, CleanupRequest(request_id="events", kind="events", expected_revision=1))
    for update in ({"tenant_id": "other"}, {"application_id": "other"}, {"subject_id": "other"}):
        with pytest.raises(RuntimeFault):
            await manager.inspect(actor.model_copy(update=update), job.job_id)
    with pytest.raises(RuntimeFault, match="RETENTION_SCOPE_REQUIRED"):
        await manager.advance(operator.model_copy(update={"resource_scopes": frozenset()}), job.job_id)


async def test_copy_io_holds_no_session_lock_and_new_hold_prevents_payloads(prepared):
    store, actor, operator, session, run_id = await seed(prepared)
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingCopies(Copies):
        async def erase(self, target):
            entered.set()
            await release.wait()

    manager = PostgresRetention(store, BlockingCopies())
    job = await manager.request(actor, session, CleanupRequest(request_id="delete", kind="session", reason="delete", expected_revision=1))
    task = asyncio.create_task(manager.advance(operator, job.job_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        async with store._transaction() as conn:
            await conn.execute("UPDATE harness.sessions SET legal_hold=true WHERE session_id=%s", (session,))
        release.set()
        assert (await task).state == "retry"
        async with store._transaction() as conn:
            assert await (await conn.execute("SELECT 1 FROM harness.turns WHERE run_id=%s", (run_id,))).fetchone()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_deletion_audit_failure_cannot_leave_partial_revocation_or_job(prepared, monkeypatch):
    store, actor, _, session, _ = await seed(prepared)
    manager = PostgresRetention(store)
    original = store._audit

    async def deny(conn, current, action, resource):
        if action == "retention.requested":
            raise RuntimeFault("AUDIT_UNAVAILABLE", 503)
        await original(conn, current, action, resource)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_audit", deny)
        with pytest.raises(RuntimeFault, match="AUDIT_UNAVAILABLE"):
            await manager.request(actor, session, CleanupRequest(request_id="delete", kind="session", reason="delete", expected_revision=1))
    async with store._transaction() as conn:
        assert (await store._one(conn, "SELECT revoked_at FROM harness.sessions WHERE session_id=%s", (session,)))["revoked_at"] is None
        assert await (await conn.execute("SELECT 1 FROM harness.cleanup_jobs WHERE session_id=%s", (session,))).fetchone() is None


async def test_artifact_delete_revokes_but_retained_history_blocks_body_purge(prepared):
    store, actor, operator, session, run_id = await seed(prepared, days=8)
    manager = PostgresRetention(store)
    ref = "artifact-" + uuid4().hex
    async with store._transaction() as conn:
        await conn.execute("""INSERT INTO harness.artifacts(artifact_ref,run_id,session_id,operation_id,receipt_hash,
            media_type,body,content_hash,total_chars,total_bytes,expires_at) VALUES(%s,%s,%s,'source',%s,
            'text/plain','private',%s,7,7,clock_timestamp()-interval '1 day')""", (ref, run_id, session, "a" * 64, "b" * 64))
    request = CleanupRequest(request_id="delete-artifact", kind="artifact", reason="delete", expected_revision=1,
        artifact_ref=ref, expected_content_hash="b" * 64)
    with pytest.raises(RuntimeFault, match="ARTIFACT_DELETE_CONTENT_MISMATCH"):
        await manager.request(actor, session, request.model_copy(update={"expected_content_hash": "c" * 64}))
    job = await manager.request(actor, session, request)
    blocked = await manager.advance(operator, job.job_id)
    assert blocked.state == "blocked" and "artifact_body_copies_retained" in blocked.reasons and blocked.attempts == 0
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT body,revoked_at,purged_at FROM harness.artifacts WHERE artifact_ref=%s", (ref,))
        assert row["body"] == "private" and row["revoked_at"] is not None and row["purged_at"] is None
    with pytest.raises(RaiseException, match="authorized cleanup required"):
        async with store._transaction() as conn:
            await conn.execute("UPDATE harness.artifacts SET body=NULL,purged_at=clock_timestamp() WHERE artifact_ref=%s", (ref,))


async def test_job_capacity_preserves_existing_intents_and_replay(prepared):
    store, actor, _, session, _ = await seed(prepared, policy=RetentionPolicy(max_jobs=1))
    manager = PostgresRetention(store)
    request = CleanupRequest(request_id="first", kind="events", expected_revision=1)
    job = await manager.request(actor, session, request)
    assert (await manager.request(actor, session, request)).job_id == job.job_id
    with pytest.raises(RuntimeFault, match="CLEANUP_JOB_LIMIT"):
        await manager.request(actor, session, request.model_copy(update={"request_id": "second"}))
