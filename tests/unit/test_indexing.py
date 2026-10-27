"""M3-02边界和编排用例定义；本轮未运行，不以合成后端证明真实双路就绪。"""

import argparse
import asyncio

import pytest
from pydantic import ValidationError

from harness.contracts.indexing import IndexJob, IndexRequest, member_hash
from harness.contracts.retrieval_models import RetrievalLimits
from harness.contracts.runtime import RuntimeFault
from harness.ingestion.indexing import IndexBuildService, embedding_batches
from harness.runtime.retrieval_models import GovernedRetrievalGateway
from tests.indexing_fixtures import ControlledBackend, index_members, index_values
from tests.retrieval_fixtures import Authority, EmbeddingTransport, MemoryLedger, RerankTransport
from tests.unit.test_ingestion import actor


def test_members_and_dimensions_are_immutable_generation_identity():
    _, _, binding, _ = index_values()
    assert member_hash(index_members()) == member_hash(tuple(reversed(index_members())))
    with pytest.raises(ValidationError):
        type(binding).model_validate_json(binding.model_copy(update={
            "profile": binding.profile.model_copy(update={"dimensions": 1024})}).model_dump_json())
    with pytest.raises(ValidationError):
        IndexRequest(request_id="request-1", version_ids=("v2", "v1"))
    with pytest.raises(ValidationError):
        IndexRequest(request_id="request-1", version_ids=("v1", "v1"))


def test_batch_planning_respects_bytes_items_and_total_budget_before_dispatch():
    _, bindings, _, _ = index_values()
    members = index_members()
    assert len(embedding_batches(members, bindings[0].profile, RetrievalLimits())) == 1
    small = bindings[0].profile.model_copy(update={"max_items": 1})
    assert len(embedding_batches(members, small, RetrievalLimits())) == 3
    with pytest.raises(RuntimeFault, match="BUDGET_EXCEEDED"):
        embedding_batches(members, small, RetrievalLimits(max_calls=2))
    with pytest.raises(RuntimeFault, match="INPUT_LIMIT_EXCEEDED"):
        embedding_batches(members, bindings[0].profile.model_copy(update={"max_input_bytes": 1}), RetrievalLimits())


class Repository:
    def __init__(self, binding, scope):
        self.items = index_members()
        scope = scope.model_copy(update={"manifest_hash": binding.content_hash()})
        self.job = IndexJob(job_id=scope.owner_id, generation_id="d" * 32, knowledge_base_id=scope.knowledge_base_id,
                            version_ids=("version-1",), binding=binding, scope=scope, members_hash=member_hash(self.items),
                            chunk_count=len(self.items), state="queued")
        self.failure = None

    async def begin(self, *args):
        return self.job

    async def claim(self, actor, job_id):
        if self.job.state != "queued":
            raise RuntimeFault("INDEX_JOB_BUSY")
        self.job = self.job.model_copy(update={"state": "building"})

    async def plan(self, actor, job_id, count):
        self.job = self.job.model_copy(update={"batches_expected": count})

    async def members(self, *args):
        return self.items

    async def progress(self, actor, job_id, expected):
        assert self.job.batches_written == expected
        self.job = self.job.model_copy(update={"batches_written": expected + 1})

    async def ready(self, actor, job_id, proof):
        if not proof.dense_ready or not proof.sparse_ready:
            raise RuntimeFault("INDEX_READINESS_REJECTED")
        self.job = self.job.model_copy(update={"state": "ready"})
        return self.job

    async def fail(self, actor, job_id, code):
        self.failure = code
        self.job = self.job.model_copy(update={"state": "failed", "error_code": code})


def service():
    config, bindings, binding, scope = index_values()
    repository, backend, transport = Repository(binding, scope), ControlledBackend(), EmbeddingTransport()
    gateway = GovernedRetrievalGateway(ledger=MemoryLedger(), authority=Authority(), embedding=transport,
                                       reranking=RerankTransport(), embedding_binding=bindings[0],
                                       rerank_binding=bindings[1], limits=config.limits, configuration_guard=lambda: None)
    return IndexBuildService(repository, backend, gateway, binding, bindings[0].profile, config.limits), transport


async def test_complete_build_is_not_implicitly_published_or_replayed():
    builder, transport = service()
    request = IndexRequest(request_id="request-1", version_ids=("version-1",))
    job = await builder.build(actor(), "kb-1", request)
    assert job.state == "ready" and job.batches_written == job.batches_expected == 1
    assert await builder.build(actor(), "kb-1", request) == job
    assert transport.calls == 1 and builder.backend.created == 1
    assert builder.backend.inserted == [m.chunk_id for m in builder.repository.items]


async def test_sparse_not_ready_and_unknown_write_never_mark_ready():
    builder, transport = service()
    builder.backend.sparse_ready = False
    with pytest.raises(RuntimeFault, match="READINESS_REJECTED"):
        await builder.build(actor(), "kb-1", IndexRequest(request_id="r1", version_ids=("version-1",)))
    assert builder.repository.job.state == "failed"
    builder, transport = service()
    builder.backend.failure = RuntimeFault("MILVUS_RPC_OUTCOME_UNKNOWN")
    request = IndexRequest(request_id="r1", version_ids=("version-1",))
    with pytest.raises(RuntimeFault, match="OUTCOME_UNKNOWN"):
        await builder.build(actor(), "kb-1", request)
    assert (await builder.build(actor(), "kb-1", request)).state == "failed"
    assert builder.backend.created == 1 and transport.calls == 0


async def test_cancelled_model_preserves_failure_and_propagates_cancellation():
    builder, transport = service()
    transport.cancel = True
    with pytest.raises(asyncio.CancelledError):
        await builder.build(actor(), "kb-1", IndexRequest(request_id="r1", version_ids=("version-1",)))
    assert builder.repository.failure == "INDEX_CANCELLED" and not builder.backend.inserted


async def test_disabled_cli_never_loads_backend_or_identity(tmp_path, monkeypatch):
    from harness.bootstrap.indexing import execute

    monkeypatch.setattr("harness.bootstrap.providers.model_values", lambda root: {})
    with pytest.raises(RuntimeFault, match="RAG_DISABLED"):
        await execute(tmp_path, argparse.Namespace(command="build"))


def test_native_rpc_does_not_replay_and_always_closes_channel():
    from types import SimpleNamespace

    from harness.adapters.milvus_index import MilvusDualIndex

    class RpcError(Exception):
        pass

    calls = []
    options_seen = []
    channel = SimpleNamespace(close=lambda: calls.append("closed"))

    def open_channel(target, options):
        options_seen.extend(options)
        return channel

    def insert(request, timeout, metadata):
        calls.append("insert")
        assert timeout == 3
        raise RpcError()

    backend = object.__new__(MilvusDualIndex)
    backend.grpc = SimpleNamespace(insecure_channel=open_channel, RpcError=RpcError)
    backend.secure, backend.target, backend.database, backend.metadata = False, "localhost:19531", "default", ()
    backend.stub_type = lambda value: SimpleNamespace(Insert=insert)
    with pytest.raises(RuntimeFault, match="RPC_OUTCOME_UNKNOWN"):
        backend._invoke("Insert", SimpleNamespace(db_name=""), 3)
    assert calls == ["insert", "closed"]
    assert ("grpc.enable_retries", 0) in options_seen
