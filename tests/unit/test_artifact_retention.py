"""删除意图/保留视图与分页故障的用例定义；本批不运行。"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from harness.bootstrap.api import create_app
from harness.context.artifact_retention import retention_view
from harness.contracts.artifacts import ArtifactDeletionRequest, ArtifactRetentionFacts, ArtifactRetentionPolicy
from harness.contracts.runtime import RuntimeFault
from harness.tools.artifacts import registered_artifact_reader
from harness.tools.reads import ReadCoordinator
from tests.unit.test_artifacts import PageReader, ReadLedger
from tests.unit.test_knowledge_runtime import values
from harness.contracts.specs import ToolBinding


def view(facts=None, *, revoked=False, expired=False, deletion_requested=False):
    now = datetime(2026, 10, 27, tzinfo=UTC)
    facts = facts or ArtifactRetentionFacts(**{name: False for name in ArtifactRetentionFacts.model_fields})
    return retention_view(artifact_ref="artifact-" + "a" * 32, facts=facts, policy=ArtifactRetentionPolicy(),
        checked_at=now, expires_at=now + timedelta(seconds=-1 if expired else 60),
        revoked_at=now if revoked else None, deletion_requested=deletion_requested,
        read_denial_code="ARTIFACT_REVOKED" if revoked else "ARTIFACT_EXPIRED" if expired else None)


@pytest.mark.parametrize("reason", tuple(ArtifactRetentionFacts.model_fields))
def test_each_activity_or_copy_blocks_cleanup_even_after_ttl_and_revocation(reason):
    facts = ArtifactRetentionFacts(**{name: name == reason for name in ArtifactRetentionFacts.model_fields})
    for revoked, expired in ((False, False), (False, True), (True, False), (True, True)):
        result = view(facts, revoked=revoked, expired=expired, deletion_requested=revoked)
        assert result.state == "retained" and reason in result.retention_reasons
        assert result.readable == (not revoked and not expired)
        assert result.next_check_at > result.checked_at
        assert "cleanup_executor_not_installed" in result.retention_reasons
    assert view(expired=True).state == "cleanup_pending"
    assert "deleted" not in view(revoked=True).model_dump_json()


def test_deletion_request_requires_exact_hash_and_rejects_scope_injection():
    for update in ({"expected_content_hash": "wrong"}, {"tenant_id": "other"}, {"subject_id": "admin"}):
        with pytest.raises(ValidationError):
            ArtifactDeletionRequest.model_validate({"request_id": "delete-1", "expected_content_hash": "a" * 64, **update})


@pytest.mark.asyncio
@pytest.mark.parametrize("code,status", [("ARTIFACT_EXPIRED", 410), ("RESOURCE_SCOPE_REVOKED", 403), ("ARTIFACT_REVOKED", 403)])
async def test_cached_page_denial_never_reinvokes_original_tool(code, status):
    run, _, _, _ = values()
    reader, ledger = PageReader(), ReadLedger()
    entry = registered_artifact_reader(reader)
    spec = run.manifest.spec.model_copy(update={"allowed_tools": (ToolBinding(name="read_artifact", version="1", max_operations_per_run=4),)})
    run = run.model_copy(update={"manifest": run.manifest.model_copy(update={"spec": spec})})
    reads = ReadCoordinator(ledger, (entry,))
    args = {"artifact_ref": "artifact-" + "a" * 32}
    await reads.execute(run, "owner", "read_artifact", "page", args)

    async def denied(*args):
        raise RuntimeFault(code, status)

    reader.read = denied
    with pytest.raises(RuntimeFault, match=code):
        await reads.execute(run, "owner", "read_artifact", "page", args)
    assert ledger.count == 1


@pytest.mark.asyncio
async def test_management_api_returns_pending_metadata_only_and_validates_ref():
    run, _, _, _ = values()
    calls = []

    class Store:
        async def admit_read(self, actor):
            pass

    class Retention:
        async def inspect(self, actor, session_id, artifact_ref):
            calls.append((actor, session_id, artifact_ref))
            return view()

        async def request_deletion(self, actor, session_id, artifact_ref, request):
            calls.append((actor, session_id, artifact_ref, request))
            return view(revoked=True, deletion_requested=True)

    app = create_app(Store(), lambda token: run.principal, artifact_retention=Retention())
    prefix = f"/api/v1/sessions/{run.session_id}/artifacts/artifact-{'a' * 32}"
    body = {"request_id": "delete-1", "expected_content_hash": "a" * 64}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        result = await client.post(prefix + "/deletion-requests", json=body)
        assert result.status_code == 202 and result.headers["cache-control"] == "no-store"
        assert result.json()["deletion_requested"] and not result.json()["readable"]
        assert not {"text", "body", "content", "source_receipt"} & result.json().keys()
        assert (await client.get(prefix + "/retention")).status_code == 200
        assert (await client.get(prefix.replace("artifact-" + "a" * 32, "bad-ref") + "/retention")).status_code == 422
        assert (await client.post(prefix + "/deletion-requests", json={**body, "subject_id": "admin"})).status_code == 422
    assert all(call[0] == run.principal and call[1] == run.session_id for call in calls)
