"""真实PG撤权/保留、分页故障和删除意图竞争定义；本批未运行。"""

import asyncio

import httpx
import pytest
from psycopg.errors import RaiseException

from harness.adapters.artifact_retention_postgres import PostgresArtifactRetention
from harness.bootstrap.api import create_app
from harness.contracts.artifacts import ArtifactDeletionRequest, ArtifactRead, ArtifactRetentionPolicy
from harness.contracts.runtime import RuntimeFault
from tests.integration.test_artifacts_postgres import artifact_case as artifact_case

pytestmark = pytest.mark.integration


async def saved(case):
    _, artifacts, reads, _, owner, run, _, _ = case
    receipt = await reads.execute(run, owner, "large_result", "source", {})
    info = (await artifacts.project(run, owner, receipt)).as_object()["artifact"]
    return receipt, info, ArtifactRead(artifact_ref=info["artifact_ref"], offset=2000, limit=100)


async def test_deletion_is_idempotent_revocation_and_retains_all_evidence(artifact_case):
    store, artifacts, reads, actor, owner, run, calls, _ = artifact_case
    source, info, page_request = await saved(artifact_case)
    page = await reads.execute(run, owner, "read_artifact", "page", page_request.model_dump())
    retention = PostgresArtifactRetention(artifacts, ArtifactRetentionPolicy())
    request = ArtifactDeletionRequest(request_id="delete-1", expected_content_hash=info["content_hash"])
    results = await asyncio.gather(*(retention.request_deletion(actor, run.session_id, page_request.artifact_ref, request) for _ in range(2)))
    assert results[0].revoked_at == results[1].revoked_at
    assert all(result.deletion_requested and not result.readable and result.state == "retained" for result in results)
    assert {"source_run_active", "session_active", "source_receipt_retained", "readback_reference_retained",
            "checkpoint_scope_retained", "active_evaluation"} <= set(results[0].retention_reasons)
    for action in (artifacts.read(actor, run.session_id, page_request), artifacts.project(run, owner, page),
                   reads.execute(run, owner, "read_artifact", "page", page_request.model_dump()),
                   artifacts.project(run, owner, source)):
        with pytest.raises(RuntimeFault, match="ARTIFACT_REVOKED"):
            await action
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT count(*) AS n FROM harness.artifact_deletion_requests WHERE artifact_ref=%s", (page_request.artifact_ref,))
        assert row["n"] == 1
        artifact = await store._one(conn, "SELECT body,content_hash,expires_at FROM harness.artifacts WHERE artifact_ref=%s", (page_request.artifact_ref,))
        assert artifact["body"] and artifact["content_hash"] == info["content_hash"]
    with pytest.raises(RaiseException, match="cleanup"):
        async with store._transaction() as conn:
            await conn.execute("DELETE FROM harness.artifacts WHERE artifact_ref=%s", (page_request.artifact_ref,))
    assert any(item.operation_id == source.operation_id and item.data == source.data
               for item in await store.read_receipts(actor, run.run_id))
    assert len(calls) == 1


async def test_management_scope_and_delete_binding_conflicts(artifact_case):
    store, artifacts, _, actor, _, run, _, _ = artifact_case
    _, info, page_request = await saved(artifact_case)
    retention = PostgresArtifactRetention(artifacts, ArtifactRetentionPolicy())
    request = ArtifactDeletionRequest(request_id="delete-1", expected_content_hash=info["content_hash"])
    for update in ({"tenant_id": "other"}, {"subject_id": "other"}, {"application_id": "other"}, {"roles": frozenset()}):
        with pytest.raises(RuntimeFault):
            await retention.request_deletion(actor.model_copy(update=update), run.session_id, page_request.artifact_ref, request)
    with pytest.raises(RuntimeFault):
        await retention.request_deletion(actor, "another-session", page_request.artifact_ref, request)
    with pytest.raises(RuntimeFault, match="ARTIFACT_DELETE_CONTENT_MISMATCH"):
        await retention.request_deletion(actor, run.session_id, page_request.artifact_ref,
                                        request.model_copy(update={"expected_content_hash": "0" * 64}))
    assert (await retention.inspect(actor, run.session_id, page_request.artifact_ref)).readable
    await retention.request_deletion(actor, run.session_id, page_request.artifact_ref, request)
    with pytest.raises(RuntimeFault, match="ARTIFACT_DELETE_REQUEST_CONFLICT"):
        await retention.request_deletion(actor, run.session_id, page_request.artifact_ref,
                                        request.model_copy(update={"expected_content_hash": "0" * 64}))
    async with store._transaction() as conn:
        assert (await store._one(conn, "SELECT count(*) AS n FROM harness.artifact_deletion_requests WHERE session_id=%s", (run.session_id,)))["n"] == 1


async def test_revocation_survives_later_inspection_failure_but_audit_failure_rolls_back(artifact_case, monkeypatch):
    store, artifacts, _, actor, _, run, _, _ = artifact_case
    _, info, page_request = await saved(artifact_case)
    retention = PostgresArtifactRetention(artifacts, ArtifactRetentionPolicy())
    request = ArtifactDeletionRequest(request_id="delete-1", expected_content_hash=info["content_hash"])
    original = store._audit

    async def audit_failure(conn, actor, action, ref):
        if action == "artifact.deletion_requested":
            raise RuntimeFault("AUDIT_UNAVAILABLE", 503)
        await original(conn, actor, action, ref)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_audit", audit_failure)
        with pytest.raises(RuntimeFault, match="AUDIT_UNAVAILABLE"):
            await retention.request_deletion(actor, run.session_id, page_request.artifact_ref, request)
    assert (await retention.inspect(actor, run.session_id, page_request.artifact_ref)).readable

    async def inspect_failure(*args):
        raise RuntimeFault("RETENTION_QUERY_UNAVAILABLE", 503)

    with monkeypatch.context() as patch:
        patch.setattr(retention, "inspect", inspect_failure)
        with pytest.raises(RuntimeFault, match="RETENTION_QUERY_UNAVAILABLE"):
            await retention.request_deletion(actor, run.session_id, page_request.artifact_ref, request)
    with pytest.raises(RuntimeFault, match="ARTIFACT_REVOKED"):
        await artifacts.read(actor, run.session_id, page_request)
    assert (await retention.request_deletion(actor, run.session_id, page_request.artifact_ref, request)).deletion_requested


async def test_delete_serializes_with_a_page_already_being_read(artifact_case, monkeypatch):
    _, artifacts, _, actor, _, run, _, _ = artifact_case
    _, info, page_request = await saved(artifact_case)
    retention = PostgresArtifactRetention(artifacts, ArtifactRetentionPolicy())
    locked, release = asyncio.Event(), asyncio.Event()
    original = artifacts._authorized

    async def paused(*args):
        row = await original(*args)
        locked.set()
        await release.wait()
        return row

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "_authorized", paused)
        read = asyncio.create_task(artifacts.read(actor, run.session_id, page_request))
        try:
            await asyncio.wait_for(locked.wait(), 3)
            deletion = asyncio.create_task(retention.request_deletion(actor, run.session_id, page_request.artifact_ref,
                ArtifactDeletionRequest(request_id="delete-race", expected_content_hash=info["content_hash"])))
            release.set()
            assert (await read).artifact_ref == page_request.artifact_ref
            assert not (await deletion).readable
        finally:
            release.set()
            if not read.done():
                await read
    with pytest.raises(RuntimeFault, match="ARTIFACT_REVOKED"):
        await artifacts.read(actor, run.session_id, page_request)


@pytest.mark.parametrize("artifact_case", ["expire"], indirect=True)
async def test_expiry_blocks_client_cache_and_projection_without_freeing_active_evidence(artifact_case):
    store, artifacts, reads, actor, owner, run, calls, _ = artifact_case
    source, info, page_request = await saved(artifact_case)
    page = await reads.execute(run, owner, "read_artifact", "page", page_request.model_dump())
    await asyncio.sleep(1.05)
    for action in (artifacts.read(actor, run.session_id, page_request), artifacts.project(run, owner, source),
                   artifacts.project(run, owner, page), reads.execute(run, owner, "read_artifact", "page", page_request.model_dump())):
        with pytest.raises(RuntimeFault, match="ARTIFACT_EXPIRED"):
            await action
    retention = PostgresArtifactRetention(artifacts, ArtifactRetentionPolicy())
    view = await retention.inspect(actor, run.session_id, page_request.artifact_ref)
    assert view.state == "retained" and not view.readable and view.read_denial_code == "ARTIFACT_EXPIRED"
    assert {"source_run_active", "source_receipt_retained"} <= set(view.retention_reasons)
    app = create_app(store, lambda token: actor, artifacts=artifacts, artifact_retention=retention)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/sessions/{run.session_id}/artifacts/{page_request.artifact_ref}")
        assert response.status_code == 410 and "text" not in response.json()
    assert len(calls) == 1


async def test_live_identity_and_evaluation_revocation_close_all_artifact_reads(artifact_case, tmp_path):
    from harness.adapters.development_auth import LiveDevelopmentAuthenticator
    import json

    store, artifacts, reads, actor, owner, run, _, _ = artifact_case
    _, _, page_request = await saved(artifact_case)
    page = await reads.execute(run, owner, "read_artifact", "page", page_request.model_dump())
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"a" * 40: actor.model_dump(mode="json")}), "utf-8")
    store.identity_source = LiveDevelopmentAuthenticator(path, "development")
    path.write_text("{}", "utf-8")
    with pytest.raises(RuntimeFault):
        await artifacts.read(actor, run.session_id, page_request)
    with pytest.raises(RuntimeFault):
        await artifacts.project(run, owner, page)
    path.write_text(json.dumps({"a" * 40: actor.model_dump(mode="json")}), "utf-8")
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.evaluation_contexts SET revoked_at=clock_timestamp() WHERE context_id=(SELECT evaluation_context_id FROM harness.sessions WHERE session_id=%s)", (run.session_id,))
    with pytest.raises(RuntimeFault, match="EVALUATION_REVOKED_OR_EXPIRED"):
        await artifacts.read(actor, run.session_id, page_request)
    with pytest.raises(RuntimeFault):
        await artifacts.authorize_context(run, owner)


async def test_changed_source_receipt_blocks_existing_ref_and_model_context(artifact_case):
    store, artifacts, _, actor, owner, run, _, _ = artifact_case
    source, _, page_request = await saved(artifact_case)
    async with store._transaction() as conn:
        await conn.execute("""UPDATE harness.read_operations SET receipt=jsonb_set(receipt,'{data}',
            '{"body":"corrupted-source"}'::jsonb) WHERE operation_id=%s""", (source.operation_id,))
    with pytest.raises(RuntimeFault, match="ARTIFACT_SOURCE_RECEIPT_CHANGED"):
        await artifacts.read(actor, run.session_id, page_request)
    with pytest.raises(RuntimeFault, match="ARTIFACT_SOURCE_RECEIPT_CHANGED"):
        await artifacts.authorize_context(run, owner)
