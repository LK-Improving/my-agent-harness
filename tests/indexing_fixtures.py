"""索引合成夹具；模型/后端均标记受控，不冒充实际Milvus或供应商证据。"""

from harness.contracts.indexing import IndexBinding, IndexMember, IndexProfile, IndexReadiness, member_hash, sample_members
from harness.contracts.hashing import canonical_hash
from harness.contracts.snapshots import JsonSnapshot
from tests.retrieval_fixtures import retrieval_values


def index_values():
    config, _, bindings, scope = retrieval_values()
    snapshot = JsonSnapshot.capture({
        "identity_version": "retrieval-model-runtime@1",
        "embedding": bindings[0].snapshot.materialize(), "rerank": bindings[1].snapshot.materialize(),
        "limits": config.limits.model_dump(mode="json"), "environment_profile_hash": "a" * 64,
    })
    binding = IndexBinding(profile=IndexProfile(dimensions=2), model_snapshot=snapshot,
                           embedding_binding_hash=bindings[0].snapshot.content_hash,
                           rerank_binding_hash=bindings[1].snapshot.content_hash, backend_hash="c" * 64)
    return config, bindings, binding, scope


def index_members():
    return tuple(sorted((IndexMember(chunk_id=canonical_hash(f"chunk-{i}"), version_id="version-1",
                                    document_id="doc-1", body_hash=canonical_hash(f"合成资料{i}"), body=f"合成资料{i}")
                         for i in range(3)), key=lambda m: m.chunk_id))


def readiness(job, members, **changes):
    return IndexReadiness(generation_id=job.generation_id, binding_hash=job.binding.content_hash(),
                          members_hash=member_hash(members), row_count=len(members), dense_ready=True,
                          sparse_ready=True, sample_ids=tuple(m.chunk_id for m in sample_members(members))).model_copy(
                              update=changes)


class ControlledBackend:
    def __init__(self):
        self.created = 0
        self.inserted = []
        self.failure = None
        self.sparse_ready = True

    async def create(self, job):
        self.created += 1
        if self.failure:
            raise self.failure

    async def insert(self, job, members, vectors):
        self.inserted.extend(m.chunk_id for m in members)

    async def seal(self, job, members, samples):
        return readiness(job, members, sparse_ready=self.sparse_ready)
