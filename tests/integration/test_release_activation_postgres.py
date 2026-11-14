"""M5-05 真实 PG 指针和审计用例定义；受信报告验证使用独立单测。"""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from harness.adapters.postgres.activation import PostgresReleaseActivation
from harness.bootstrap.api import create_app
from harness.contracts.activation import ActivationPolicy, ActivationRequest
from harness.contracts.evaluation_contexts import EvaluationRevocationRequest
from harness.contracts.runtime import EvaluationRequest, RuntimeFault, SessionRequest
from tests.integration.test_m1_runtime import prepared

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scenario", ["first", "audit_error", "revoked", "publisher_revoked"])
async def test_activation_transaction(prepared, tmp_path, monkeypatch, scenario):
    store, evaluator, manifest, _ = prepared
    agent_id = manifest.spec.agent_id
    context = await store.create_evaluation(evaluator, EvaluationRequest(
        candidate_release_id=manifest.spec.release_id, dataset_hash="d" * 64,
        gate_policy_version="activation-test@1", expires_at=datetime.now(UTC) + timedelta(minutes=5),
        resource_scopes=frozenset({"synthetic"}),
    ))
    publisher = evaluator.model_copy(update={
        "roles": frozenset({"publisher"}), "resource_scopes": frozenset({f"agent:{agent_id}:publish"}),
    })
    policy = ActivationPolicy(
        tenant_id=publisher.tenant_id, application_id=publisher.application_id, agent_id=agent_id,
        environment_profile_hash=manifest.environment_profile_hash, freeze_hash="f" * 64,
        dataset_hash=context.dataset_hash, gate_policy_version=context.gate_policy_version,
        report_hash="a" * 64,
    )
    request = ActivationRequest(release_id=manifest.spec.release_id, manifest_hash=manifest.content_hash(),
                                report_hash=policy.report_hash, expected_active_release_id=None)
    identity = (lambda _: evaluator) if scenario == "publisher_revoked" else (lambda actor: actor)
    def unexpected_observation(_):
        raise AssertionError("事务用例使用已验证材料夹具")

    activation = PostgresReleaseActivation(store, tmp_path, unexpected_observation, identity)
    monkeypatch.setattr(activation, "_verify", lambda *_: (policy, context))
    with pytest.raises(RuntimeFault, match="FORBIDDEN"):
        await activation.activate(evaluator, agent_id, request)

    if scenario == "revoked":
        await store.revoke_evaluation(evaluator, context.context_id, EvaluationRevocationRequest(
            request_id="revoke-activation", expected_manifest_hash=manifest.content_hash(),
        ))
    if scenario == "audit_error":
        original_audit = store._audit

        async def audit(conn, actor, action, resource):
            if action == "release.activated":
                await conn.execute("SELECT 1 / 0")
            await original_audit(conn, actor, action, resource)

        monkeypatch.setattr(store, "_audit", audit)

    app = create_app(store, lambda _: publisher, activate=activation.activate)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                 base_url="http://test") as client:
        async def submit():
            return await client.post(f"/api/v1/agents/{agent_id}/activations", json=request.model_dump(mode="json"))

        if scenario == "first":
            responses = await asyncio.gather(submit(), submit())
            assert sorted(item.status_code for item in responses) == [200, 409]
        else:
            response = await submit()
            assert response.status_code == {"audit_error": 503, "revoked": 409, "publisher_revoked": 403}[scenario]

    async with store._transaction() as conn:
        scope = (publisher.tenant_id, publisher.application_id, agent_id)
        pointer = await (await conn.execute(
            "SELECT release_id FROM harness.active_releases WHERE tenant_id=%s AND application_id=%s AND agent_id=%s",
            scope,
        )).fetchone()
        audits = await (await conn.execute(
            "SELECT resource_id FROM harness.audit WHERE tenant_id=%s AND action='release.activated'",
            (publisher.tenant_id,),
        )).fetchall()
    if scenario == "first":
        assert pointer["release_id"] == manifest.spec.release_id
        assert len(audits) == 1 and policy.report_hash in audits[0]["resource_id"]
        assert await store.create_session(evaluator, SessionRequest(agent_id=agent_id))
    else:
        assert pointer is None and not audits
