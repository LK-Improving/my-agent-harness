"""生命周期边界与外部删除协议定义；本轮未运行，不证明实际PG/Milvus删除。"""

import asyncio
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from harness.adapters.ingestion_sources import LocalSourceStore
from harness.adapters.milvus_cleanup import MilvusIndexCleanup
from harness.contracts.ingestion import SourceRef
from harness.contracts.knowledge_lifecycle import CleanupStep, DeleteDocument
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.ingestion.lifecycle import KnowledgeLifecycle
from tests.indexing_fixtures import index_values
from tests.unit.test_ingestion import actor


async def test_disabled_cli_stops_before_identity_or_optional_backend_loading(tmp_path, monkeypatch):
    from harness.bootstrap.knowledge_lifecycle import execute

    monkeypatch.setattr("harness.bootstrap.providers.model_values", lambda root: {"HARNESS_RAG_ENABLED": "false"})
    with pytest.raises(RuntimeFault, match="RAG_DISABLED"):
        await execute(tmp_path, SimpleNamespace(command="retry", job_id="job-1", max_steps=1))


async def test_source_delete_verifies_store_hash_missing_idempotency_and_path(tmp_path):
    store = LocalSourceStore(tmp_path)
    body = b"private-text"
    ref = SourceRef(object_id="a" * 64, sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
    target = JsonSnapshot.capture({"source": ref.model_dump(mode="json"), "source_store_ref": store.storage_ref()})
    await store.put(ref, body)
    await store.delete(target)
    await store.delete(target)
    assert not (tmp_path / ref.object_id).exists()
    (tmp_path / ref.object_id).write_bytes(b"different-bytes")
    with pytest.raises(RuntimeFault, match="HASH_MISMATCH"):
        await store.delete(target)
    assert (tmp_path / ref.object_id).read_bytes() == b"different-bytes"
    with pytest.raises(RuntimeFault, match="STORE_CHANGED"):
        await LocalSourceStore(tmp_path / "other").delete(target)


def test_explicit_timezone_and_extra_scope_are_rejected():
    with pytest.raises(ValueError):
        DeleteDocument(request_id="delete-1", document_id="doc-1", retain_until=datetime(2026, 10, 27), source_store_ref="a" * 64)
    with pytest.raises(ValueError):
        DeleteDocument(request_id="delete-1", document_id="doc-1", retain_until=datetime.now(UTC),
                       source_store_ref="a" * 64, tenant_id="other")


async def test_service_persists_failure_and_propagates_cancellation():
    step = CleanupStep(step_id="step-1", job_id="job-1", kind="source", attempt=1, target=JsonSnapshot.capture({}))
    recorded = []

    class Repository:
        async def claim(self, *args): return step
        async def fail(self, actor, step, code): recorded.append(code)
        async def finish(self, *args): pytest.fail("取消后不能记成功")

    class Sources:
        async def delete(self, target): raise asyncio.CancelledError()

    service = KnowledgeLifecycle(Repository(), Sources(), None)
    with pytest.raises(asyncio.CancelledError):
        await service.advance(actor(), "job-1")
    assert recorded == ["CLEANUP_CANCELLED"]
    with pytest.raises(RuntimeFault, match="STEP_LIMIT"):
        await service.advance(actor(), "job-1", max_steps=33)


async def test_dual_index_delete_uses_fixed_scope_and_requires_zero_strong_count():
    from harness.contracts.indexing import IndexJob

    _, _, binding, scope = index_values()
    scope = scope.model_copy(update={"manifest_hash": binding.content_hash()})
    job = IndexJob(job_id=scope.owner_id, generation_id="a" * 32, knowledge_base_id=scope.knowledge_base_id,
        version_ids=("v1",), binding=binding, scope=scope, members_hash="b" * 64, chunk_count=1, state="building")
    target = JsonSnapshot.capture({"job": job.model_dump(mode="json"), "chunk_ids": ["c" * 64]})

    class Backend:
        messages = SimpleNamespace(HasCollectionRequest=lambda **kwargs: kwargs, DeleteRequest=lambda **kwargs: kwargs)
        Prepare = SimpleNamespace(query_request=lambda *args, **kwargs: (args, kwargs), load_collection=lambda name: {"name": name})
        calls, remaining = [], 0

        @staticmethod
        def _guard(job): return 'tenant == "tenant-1" and generation == "' + job.generation_id + '"'

        async def request(self, fixed, deadline, method, request):
            assert fixed == binding and deadline > datetime.now(UTC)
            self.calls.append((method, request))
            if method == "HasCollection": return SimpleNamespace(value=True)
            return SimpleNamespace(fields_data=[SimpleNamespace(scalars=SimpleNamespace(long_data=SimpleNamespace(data=[self.remaining])))])

    backend = Backend()
    await MilvusIndexCleanup(backend).delete(target)
    assert [method for method, _ in backend.calls] == ["HasCollection", "LoadCollection", "Delete", "Query"]
    assert '"' + "c" * 64 + '"' in backend.calls[2][1]["expr"]
    assert job.generation_id in backend.calls[2][1]["expr"]
    assert backend.calls[3][1][1]["consistency_level"] == 0
    backend.remaining = 1
    with pytest.raises(RuntimeFault, match="NOT_EMPTY"):
        await MilvusIndexCleanup(backend).delete(target)
