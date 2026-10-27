"""同一collection行包含稠密/BM25两路；固定作用域删除并Strong回读确认，不重放写入。"""

import json
from datetime import UTC, datetime, timedelta

from pydantic import Field

from harness.adapters.milvus_index import MilvusDualIndex
from harness.contracts.indexing import IndexJob
from harness.contracts.providers import Digest
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract


class IndexCleanupTarget(Contract):
    job: IndexJob
    chunk_ids: tuple[Digest, ...] = Field(min_length=1, max_length=1000)


class MilvusIndexCleanup:
    def __init__(self, backend: MilvusDualIndex) -> None:
        self.backend = backend

    async def delete(self, target: JsonSnapshot) -> None:
        fixed = IndexCleanupTarget.model_validate_json(target.canonical_text, strict=True)
        if tuple(sorted(set(fixed.chunk_ids))) != fixed.chunk_ids:
            raise RuntimeFault("CLEANUP_INDEX_TARGET_INVALID", 409)
        job, backend = fixed.job, self.backend
        # 清理有自己的有界期限，不能篡改原构建scope来复活模型调用。
        deadline = datetime.now(UTC) + timedelta(seconds=90)
        exists = await backend.request(job.binding, deadline, "HasCollection",
            backend.messages.HasCollectionRequest(collection_name=job.collection_name))
        if not exists.value:
            return
        # 冷代际需先加载才能Strong回读；未就绪时本次失败，不伪造已清空。
        await backend.request(job.binding, deadline, "LoadCollection", backend.Prepare.load_collection(job.collection_name))
        guard = backend._guard(job) + " and id in " + json.dumps(list(fixed.chunk_ids))
        await backend.request(job.binding, deadline, "Delete",
            backend.messages.DeleteRequest(collection_name=job.collection_name, expr=guard))
        remaining = await backend.request(job.binding, deadline, "Query", backend.Prepare.query_request(
            job.collection_name, guard, ["count(*)"], [], use_default_consistency=False, consistency_level=0))
        if not remaining.fields_data or list(remaining.fields_data[0].scalars.long_data.data) != [0]:
            raise RuntimeFault("CLEANUP_INDEX_NOT_EMPTY", 503)
