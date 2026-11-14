"""M5-02 真实 PG 隔离/额度/撤销与审计定义；本轮未运行。"""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from harness.bootstrap.api import create_app
from harness.contracts.evaluation_contexts import EvaluationAdmission, EvaluationLimits, EvaluationRevocationRequest
from harness.contracts.hashing import RunRequest
from harness.contracts.runtime import EvaluationRequest, RuntimeFault, SessionRequest

pytestmark = pytest.mark.integration


async def new_context(store, actor, manifest, **changes):
    return await store.create_evaluation(actor, EvaluationRequest.model_validate({
        "candidate_release_id": manifest.spec.release_id, "dataset_hash": "d" * 64,
        "gate_policy_version": "evaluation-test@1", "resource_scopes": ["synthetic"],
        "expires_at": datetime.now(UTC) + timedelta(hours=1), **changes}))


async def session(store, actor, manifest, context):
    return await store.create_session(actor, SessionRequest(agent_id=manifest.spec.agent_id,
                                                            evaluation_context_id=context.context_id))


async def test_no_active_candidate_fixed_even_when_active_points_elsewhere(prepared):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    sid = await session(store, actor, manifest, context)
    other = manifest.model_copy(update={"spec": manifest.spec.model_copy(update={"release_id": "other"})})
    await store.register_release(actor, other)
    async with store._transaction() as conn:
        await conn.execute("INSERT INTO harness.active_releases(tenant_id,application_id,agent_id,release_id) "
                           "VALUES(%s,%s,%s,%s)",
                           (actor.tenant_id, actor.application_id, manifest.spec.agent_id, other.spec.release_id))
    run = await store.create_run(actor, sid, RunRequest(request_id="fixed", message="查询", expected_revision=0))
    assert run.release_id == manifest.spec.release_id and run.manifest_hash == context.manifest_hash
    assert run.environment_profile_hash == context.environment_profile_hash


@pytest.mark.parametrize("identity_change", [
    {"tenant_id": "other"}, {"subject_id": "other"}, {"application_id": "other"}])
async def test_context_cannot_be_transferred_or_inspected_across_identity(prepared, identity_change):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    stranger = actor.model_copy(update=identity_change)
    for action in (lambda: store.inspect_evaluation(stranger, context.context_id),
                   lambda: session(store, stranger, manifest, context),
                   lambda: store.revoke_evaluation(stranger, context.context_id,
                       EvaluationRevocationRequest(request_id="wrong", expected_manifest_hash=context.manifest_hash))):
        with pytest.raises(RuntimeFault) as denied:
            await action()
        assert denied.value.status == 404


async def test_scope_candidates_limits_and_lifetime_are_server_owned(prepared):
    store, actor, manifest, _ = prepared
    privileged = actor.model_copy(update={"resource_scopes": frozenset({"synthetic", "kb:production"})})
    with pytest.raises(RuntimeFault, match="EVALUATION_TEST_SCOPE_REQUIRED"):
        await new_context(store, privileged, manifest, resource_scopes=["kb:production"])
    with pytest.raises(RuntimeFault, match="INVALID_EVALUATION_EXPIRY"):
        await new_context(store, actor, manifest, expires_at=datetime.now(UTC) + timedelta(days=2))
    store.profile = store.profile.model_copy(update={"evaluation": EvaluationAdmission(
        limits=EvaluationLimits(max_sessions=2, max_runs=2))})
    with pytest.raises(RuntimeFault, match="EVALUATION_LIMIT_ESCALATION"):
        await new_context(store, actor, manifest)
    store.profile = store.profile.model_copy(update={"evaluation": EvaluationAdmission(candidate_release_ids=frozenset())})
    with pytest.raises(RuntimeFault, match="EVALUATION_CANDIDATE_NOT_ALLOWED"):
        await new_context(store, actor, manifest)


async def test_session_quota_is_atomic_and_counts_roll_back_with_failed_creation(prepared):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest, limits={"max_sessions": 1, "max_runs": 1})
    results = await asyncio.gather(*(session(store, actor, manifest, context) for _ in range(2)),
                                   return_exceptions=True)
    assert sum(isinstance(row, str) for row in results) == 1
    assert sum(isinstance(row, RuntimeFault) and row.code == "EVALUATION_RESOURCE_LIMIT" for row in results) == 1
    view = await store.inspect_evaluation(actor, context.context_id)
    assert view["session_count"] == 1 and view["run_count"] == 0


async def test_run_quota_does_not_recount_same_request_and_cannot_be_reset(prepared):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest, limits={"max_sessions": 2, "max_runs": 1})
    first, second = await asyncio.gather(session(store, actor, manifest, context), session(store, actor, manifest, context))
    request = RunRequest(request_id="one", message="查询", expected_revision=0)
    accepted = await store.create_run(actor, first, request)
    assert (await store.create_run(actor, first, request)).run_id == accepted.run_id
    with pytest.raises(RuntimeFault, match="EVALUATION_RESOURCE_LIMIT"):
        await store.create_run(actor, second, request)
    view = await store.inspect_evaluation(actor, context.context_id)
    assert view["run_count"] == 1


async def test_revocation_is_single_direction_idempotent_and_blocks_api_sse(prepared):
    store, actor, manifest, sid = prepared
    accepted = await store.create_run(actor, sid, RunRequest(request_id="before", message="查询", expected_revision=0))
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT evaluation_context_id FROM harness.sessions WHERE session_id=%s", (sid,))
    context_id = row["evaluation_context_id"]
    request = EvaluationRevocationRequest(request_id="revoke-1", expected_manifest_hash=manifest.content_hash())
    first = await store.revoke_evaluation(actor, context_id, request)
    second = await store.revoke_evaluation(actor, context_id, request)
    assert first == second and first["status"] == "revoked"
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOCATION_CONFLICT"):
        await store.revoke_evaluation(actor, context_id, request.model_copy(update={"reason": "security"}))
    app = create_app(store, lambda token: actor)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path in (f"/api/v1/runs/{accepted.run_id}", f"/api/v1/runs/{accepted.run_id}/events"):
            response = await client.get(path)
            assert response.status_code == 403 and "result" not in response.json()
        metadata = await client.get(f"/api/v1/evaluation-contexts/{context_id}")
        assert metadata.status_code == 200 and metadata.headers["cache-control"] == "no-store"
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await store.create_run(actor, sid, RunRequest(request_id="after", message="查询", expected_revision=0))
    async with store._transaction() as conn:
        count = await store._one(conn, "SELECT count(*) AS used FROM harness.audit WHERE "
                                      "action='evaluation.revoked' AND resource_id=%s", (context_id,))
        assert count["used"] == 1
        row = await store._one(conn, "SELECT status FROM harness.runs WHERE run_id=%s", (accepted.run_id,))
        assert row["status"] == "queued"


async def test_context_limit_is_atomic_for_concurrent_creation(prepared):
    store, actor, manifest, _ = prepared
    store.profile = store.profile.model_copy(update={"evaluation": EvaluationAdmission(
        max_valid_contexts_per_subject=2)})
    results = await asyncio.gather(*(new_context(store, actor, manifest) for _ in range(3)),
                                   return_exceptions=True)
    assert sum(not isinstance(item, Exception) for item in results) == 1
    assert sum(isinstance(item, RuntimeFault) and item.code == "EVALUATION_CONTEXT_LIMIT"
               for item in results) == 2


async def test_revocation_audit_failure_rolls_back_revocation_and_idempotency(prepared, monkeypatch):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    original = store._audit

    async def audit(conn, principal, action, resource):
        if action == "evaluation.revoked":
            raise RuntimeError("synthetic audit outage")
        await original(conn, principal, action, resource)

    monkeypatch.setattr(store, "_audit", audit)
    with pytest.raises(RuntimeError, match="synthetic audit outage"):
        await store.revoke_evaluation(actor, context.context_id, EvaluationRevocationRequest(
            request_id="audit-fails", expected_manifest_hash=context.manifest_hash))
    assert (await store.inspect_evaluation(actor, context.context_id))["status"] == "active"
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT count(*) AS used FROM harness.evaluation_revocations WHERE context_id=%s",
                               (context.context_id,))
        assert row["used"] == 0


async def test_expiry_blocks_execution_but_keeps_metadata_and_revocation(prepared):
    from psycopg.types.json import Jsonb

    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    sid = await session(store, actor, manifest, context)
    expired = context.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.evaluation_contexts SET expires_at=%s,context=%s WHERE context_id=%s",
                           (expired.expires_at, Jsonb(expired.model_dump(mode="json")), context.context_id))
    assert (await store.inspect_evaluation(actor, context.context_id))["status"] == "expired"
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await store.create_run(actor, sid, RunRequest(request_id="expired", message="查询", expected_revision=0))
    view = await store.revoke_evaluation(actor, context.context_id, EvaluationRevocationRequest(
        request_id="after-expiry", expected_manifest_hash=context.manifest_hash))
    assert view["status"] == "revoked"


async def test_run_deadline_is_bounded_by_context_and_revocation_races_admission(prepared):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest,
                                expires_at=datetime.now(UTC) + timedelta(minutes=1))
    sid = await session(store, actor, manifest, context)
    run = await store.create_run(actor, sid, RunRequest(request_id="bounded", message="查询", expected_revision=0))
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT expires_at,queue_deadline FROM harness.runs WHERE run_id=%s", (run.run_id,))
        assert row["expires_at"] <= context.expires_at and row["queue_deadline"] <= context.expires_at
    results = await asyncio.gather(session(store, actor, manifest, context),
        store.revoke_evaluation(actor, context.context_id, EvaluationRevocationRequest(
            request_id="race", expected_manifest_hash=context.manifest_hash)), return_exceptions=True)
    assert isinstance(results[1], dict) and results[1]["status"] == "revoked"
    assert isinstance(results[0], str) or isinstance(results[0], RuntimeFault)
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await session(store, actor, manifest, context)


async def test_failed_run_transaction_does_not_burn_quota(prepared, monkeypatch):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest, limits={"max_sessions": 1, "max_runs": 1})
    sid = await session(store, actor, manifest, context)
    original = store._audit

    async def audit(conn, principal, action, resource):
        if action == "run.created":
            raise RuntimeError("synthetic audit outage")
        await original(conn, principal, action, resource)

    request = RunRequest(request_id="rollback", message="查询", expected_revision=0)
    monkeypatch.setattr(store, "_audit", audit)
    with pytest.raises(RuntimeError):
        await store.create_run(actor, sid, request)
    assert (await store.inspect_evaluation(actor, context.context_id))["run_count"] == 0
    monkeypatch.setattr(store, "_audit", original)
    await store.create_run(actor, sid, request)
    assert (await store.inspect_evaluation(actor, context.context_id))["run_count"] == 1


async def test_shared_authorization_is_not_upgraded_by_context_history_locks(prepared):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    first = await session(store, actor, manifest, context)
    second = await session(store, actor, manifest, context)
    ready = [asyncio.Event(), asyncio.Event()]

    async def reader(sid, index):
        async with store._transaction() as conn:
            await store._session(conn, actor, sid)
            ready[index].set()
            await ready[1 - index].wait()
            # 模拟上下文来源再次锁session，不能升级已有context共享锁。
            await store._session(conn, actor, sid, lock=True)

    async with asyncio.timeout(4):
        await asyncio.gather(reader(first, 0), reader(second, 1))


async def test_ordinary_identity_cannot_bind_context_and_uninstalled_candidate_rejected(prepared):
    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    ordinary = actor.model_copy(update={"roles": frozenset({"user"})})
    with pytest.raises(RuntimeFault) as denied:
        await session(store, ordinary, manifest, context)
    assert denied.value.status == 403
    store.installed_manifest_hashes = frozenset()
    with pytest.raises(RuntimeFault, match="EXECUTABLE_RELEASE_NOT_INSTALLED"):
        await new_context(store, actor, manifest)


async def test_mutated_manifest_binding_and_scope_revocation_are_rechecked(prepared):
    from psycopg.types.json import Jsonb

    store, actor, manifest, _ = prepared
    context = await new_context(store, actor, manifest)
    sid = await session(store, actor, manifest, context)
    with pytest.raises(RuntimeFault, match="EVALUATION_SCOPE_REVOKED"):
        await session(store, actor.model_copy(update={"resource_scopes": frozenset()}), manifest, context)
    corrupted = context.model_copy(update={"manifest_hash": "0" * 64})
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.evaluation_contexts SET context=%s WHERE context_id=%s",
                           (Jsonb(corrupted.model_dump(mode="json")), context.context_id))
    with pytest.raises(RuntimeFault, match="EVALUATION_MANIFEST_MISMATCH"):
        await store.create_run(actor, sid, RunRequest(request_id="corrupted", message="查询", expected_revision=0))
