"""正式原生RPC适配器的实际Milvus协议用例；合成向量，不调用或重复供应商质量PoC。"""

import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from pydantic import SecretStr

from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import IndexBinding, IndexJob, IndexProfile, member_hash, sample_members
from harness.contracts.snapshots import JsonSnapshot
from tests.indexing_fixtures import index_members, index_values

pytestmark = [pytest.mark.integration, pytest.mark.rag]


async def test_formal_grpc_schema_insert_full_members_and_both_readiness():
    uri = os.environ.get("HARNESS_MILVUS_URI", "")
    parsed = urlparse(uri)
    if parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.port != 19531:
        pytest.fail("正式适配器协议用例只允许显式配置的本项目本机Milvus19531")
    from harness.adapters.milvus_index import MilvusDualIndex

    _, _, original, scope = index_values()
    # 真实通路仍用正式1024维配置；此用例的模型快照是合成fixture，绝非可信能力证据。
    model_data = original.model_snapshot.as_object()
    embedding = model_data["embedding"]
    embedding["profile"]["dimensions"] = 1024
    embedding["evidence"]["dimensions"] = 1024
    binding = IndexBinding(profile=IndexProfile(), model_snapshot=JsonSnapshot.capture(model_data),
                           embedding_binding_hash=canonical_hash(embedding), rerank_binding_hash=original.rerank_binding_hash,
                           backend_hash="c" * 64)
    job_id, generation = uuid4().hex, uuid4().hex
    scope = scope.model_copy(update={"owner_id": job_id, "manifest_hash": binding.content_hash(),
                                     "embedding_binding_hash": binding.embedding_binding_hash,
                                     "deadline": datetime.now(UTC) + timedelta(minutes=10)})
    members = index_members()
    job = IndexJob(job_id=job_id, generation_id=generation, knowledge_base_id=scope.knowledge_base_id,
                    version_ids=("version-1",), binding=binding, scope=scope, members_hash=member_hash(members),
                    chunk_count=len(members), state="building", batches_expected=1)
    backend = MilvusDualIndex(uri, SecretStr(os.environ.get("HARNESS_MILVUS_TOKEN", "")), "default", "c" * 64)
    # 只清理本用例随机生成的collection，不触碰已有代际或PoC资源。
    try:
        await backend.create(job)
        vectors = tuple(tuple(1.0 if i == ordinal else 0.0 for i in range(1024))
                        for ordinal in range(len(members)))
        await backend.insert(job, members, vectors)
        samples = tuple((m, vectors[members.index(m)]) for m in sample_members(members))
        proof = await backend.seal(job, members, samples)
        assert proof.dense_ready and proof.sparse_ready
        assert proof.members_hash == member_hash(members) and proof.row_count == len(members)
        # 新正式召回通路使用独立run scope期限，不能借用已结束的构建job期限。
        from harness.adapters.milvus_recall import MilvusRecall
        from harness.contracts.retrieval import RetrievalProfile, RetrievalQuery, RetrievalTicket, SnapshotPin
        from harness.contracts.runtime import Principal

        identity = Principal(tenant_id=scope.tenant_id, subject_id=scope.subject_id, application_id=scope.application_id)
        query_scope = scope.model_copy(update={"owner_kind": "run", "owner_id": "synthetic-run-protocol-only"})
        ticket = RetrievalTicket(
            retrieval_id=uuid4().hex, actor=identity, owner="protocol-test", scope=query_scope,
            pin=SnapshotPin(knowledge_base_id=scope.knowledge_base_id, snapshot_id="synthetic-snapshot",
                            snapshot_hash="f" * 64, generation_id=generation, index_binding_hash=binding.content_hash()),
            index_binding=binding, profile=RetrievalProfile(),
            query=RetrievalQuery(request_id="protocol-query", query=members[0].body),
        )
        recall = MilvusRecall(backend)
        dense, sparse = await recall.dense(ticket, vectors[0]), await recall.sparse(ticket)
        allowed = {m.chunk_id for m in members}
        assert dense and sparse and all(h.chunk_id in allowed for h in (*dense, *sparse))
        from harness.adapters.milvus_cleanup import MilvusIndexCleanup

        delete_target = JsonSnapshot.capture({"job": job.model_dump(mode="json"), "chunk_ids": [members[0].chunk_id]})
        cleaner = MilvusIndexCleanup(backend)
        await cleaner.delete(delete_target)
        await cleaner.delete(delete_target)
        guard = backend._guard(job) + ' and id == "' + members[0].chunk_id + '"'
        assert not await backend._search(job, "dense", list(vectors[0]), guard)
        assert not await backend._search(job, "sparse", members[0].body, guard)
        other = backend._guard(job) + ' and id == "' + members[1].chunk_id + '"'
        assert await backend._search(job, "dense", list(vectors[1]), other) == (members[1].chunk_id,)
    finally:
        assert job.collection_name == "harness_g_" + generation
        cleanup_scope = scope.model_copy(update={"deadline": datetime.now(UTC) + timedelta(seconds=30)})
        cleanup = job.model_copy(update={"scope": cleanup_scope})
        await backend._rpc(cleanup, "DropCollection", backend.Prepare.drop_collection_request(job.collection_name))
