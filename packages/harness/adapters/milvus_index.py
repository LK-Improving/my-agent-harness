"""Milvus单次原生RPC适配；复用纯协议准备逻辑，绕开SDK重试/Schema重发。"""

import asyncio
import base64
import json
import math
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any
from urllib.parse import urlparse

import anyio
from pydantic import SecretStr

from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import IndexBinding, IndexJob, IndexMember, IndexReadiness, member_hash, sample_members
from harness.contracts.retrieval_models import Vector
from harness.contracts.runtime import RuntimeFault


class MilvusDualIndex:
    def __init__(self, uri: str, token: SecretStr, database: str, backend_hash: str,
                 *, rpc_timeout_seconds: int = 20) -> None:
        parsed = urlparse(uri)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Milvus端点必须为无凭据/路径的显式HTTP(S)地址")
        if version("pymilvus") != "2.6.17":
            raise RuntimeFault("MILVUS_SDK_VERSION_UNSUPPORTED", 503)
        # 仅显式索引装配才导入rag依赖；所有网络调用由_rpc单次派发。
        import grpc
        from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
        from pymilvus.client.prepare import Prepare
        from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc

        self.grpc, self.Prepare = grpc, Prepare
        self.common, self.messages, self.stub_type = common_pb2, milvus_pb2, milvus_pb2_grpc.MilvusServiceStub
        self.CollectionSchema, self.DataType, self.FieldSchema = CollectionSchema, DataType, FieldSchema
        self.Function, self.FunctionType = Function, FunctionType
        self.target = f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 19530)}"
        self.secure = parsed.scheme == "https"
        self.database, self.backend_hash, self.timeout = database, backend_hash, rpc_timeout_seconds
        self.metadata = (("authorization", base64.b64encode(token.get_secret_value().encode()).decode()),) if (
            token.get_secret_value()) else ()

    def _schema(self, job: IndexJob) -> Any:
        fields = [self.FieldSchema("id", self.DataType.VARCHAR, is_primary=True, max_length=64)]
        for name in ("tenant", "application", "kb", "generation", "version_id", "body_hash"):
            fields.append(self.FieldSchema(name, self.DataType.VARCHAR, max_length=128))
        fields.extend((
            self.FieldSchema("text", self.DataType.VARCHAR, max_length=32000, enable_analyzer=True,
                             analyzer_params={"tokenizer": job.binding.profile.tokenizer}),
            self.FieldSchema("dense", self.DataType.FLOAT_VECTOR, dim=job.binding.profile.dimensions),
            self.FieldSchema("sparse", self.DataType.SPARSE_FLOAT_VECTOR),
        ))
        schema = self.CollectionSchema(fields, auto_id=False, enable_dynamic_field=False)
        schema.add_function(self.Function(name="text_bm25", input_field_names=["text"],
                                         output_field_names=["sparse"], function_type=self.FunctionType.BM25))
        schema.verify()
        return schema

    def _invoke(self, method: str, request: Any, timeout: float) -> Any:
        options = (("grpc.enable_retries", 0), ("grpc.max_receive_message_length", 16777216),
                   ("grpc.max_send_message_length", 16777216))
        channel = (self.grpc.secure_channel(self.target, self.grpc.ssl_channel_credentials(), options=options)
                   if self.secure else self.grpc.insecure_channel(self.target, options=options))
        try:
            if hasattr(request, "db_name"):
                request.db_name = self.database
            response = getattr(self.stub_type(channel), method)(request, timeout=timeout, metadata=self.metadata)
            status = response.status if hasattr(response, "status") else response
            if status.error_code != 0 or status.code != 0:
                if status.error_code in {self.common.PermissionDenied, self.common.OperatePrivilegeFailure}:
                    raise RuntimeFault("MILVUS_ACCESS_DENIED", 403)
                if status.code in {401, 403, 1400}:
                    raise RuntimeFault("MILVUS_ACCESS_DENIED", 403)
                if status.error_code in {self.common.RateLimit, self.common.NotReadyServe, self.common.NotReadyCoordActivating}:
                    raise RuntimeFault("MILVUS_DEPENDENCY_UNAVAILABLE", 503)
                raise RuntimeFault("MILVUS_RPC_REJECTED", 503)
            return response
        except self.grpc.RpcError as exc:
            if hasattr(exc, "code") and exc.code() in {
                self.grpc.StatusCode.PERMISSION_DENIED, self.grpc.StatusCode.UNAUTHENTICATED,
            }:
                raise RuntimeFault("MILVUS_ACCESS_DENIED", 403) from None
            if hasattr(exc, "code") and exc.code() not in {
                self.grpc.StatusCode.UNAVAILABLE, self.grpc.StatusCode.DEADLINE_EXCEEDED,
                self.grpc.StatusCode.RESOURCE_EXHAUSTED,
            }:
                raise RuntimeFault("MILVUS_RPC_REJECTED", 503) from None
            raise RuntimeFault("MILVUS_RPC_OUTCOME_UNKNOWN", 503) from None
        finally:
            channel.close()

    async def _rpc(self, job: IndexJob, method: str, request: Any) -> Any:
        return await self.request(job.binding, job.scope.deadline, method, request)

    async def request(self, binding: IndexBinding, deadline: datetime, method: str, request: Any) -> Any:
        """构建与只读召回共用单次RPC边界；调用者提供已经授权的固定binding。"""
        if binding.backend_hash != self.backend_hash:
            raise RuntimeFault("INDEX_BACKEND_BINDING_MISMATCH", 409)
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise RuntimeFault("INDEX_BUILD_DEADLINE_EXCEEDED", 409)
        task = asyncio.create_task(asyncio.to_thread(self._invoke, method, request, min(self.timeout, remaining)))
        try:
            return await asyncio.shield(task)
        except BaseException:
            # 等待有deadline的RPC结束，避免任务标失败后仍有旧线程继续写入；不重放。
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            raise

    @staticmethod
    def _guard(job: IndexJob) -> str:
        values = {"tenant": job.scope.tenant_id, "application": job.scope.application_id,
                  "kb": job.knowledge_base_id, "generation": job.generation_id}
        return " and ".join(f"{key} == {json.dumps(value, ensure_ascii=False)}" for key, value in values.items())

    async def create(self, job: IndexJob) -> None:
        await self._rpc(job, "CreateCollection", self.Prepare.create_collection_request(
            job.collection_name, self._schema(job), consistency_level="Strong"))
        for field, name, index_type, metric in (
            ("dense", "dense_idx", job.binding.profile.dense_index, job.binding.profile.dense_metric),
            ("sparse", "sparse_idx", job.binding.profile.sparse_index, job.binding.profile.sparse_metric),
        ):
            await self._rpc(job, "CreateIndex", self.Prepare.create_index_request(
                job.collection_name, field, {"index_type": index_type, "metric_type": metric}, index_name=name))

    async def insert(self, job: IndexJob, members: tuple[IndexMember, ...],
                     vectors: tuple[Vector, ...]) -> None:
        if not members or len(members) > 10 or len(members) != len(vectors):
            raise RuntimeFault("INDEX_BATCH_INVALID", 422)
        rows = []
        for member, vector in zip(members, vectors, strict=True):
            if len(vector) != job.binding.profile.dimensions or canonical_hash(member.body) != member.body_hash:
                raise RuntimeFault("INDEX_BATCH_INVALID", 422)
            rows.append({"id": member.chunk_id, "tenant": job.scope.tenant_id,
                         "application": job.scope.application_id, "kb": job.knowledge_base_id,
                         "generation": job.generation_id, "version_id": member.version_id,
                         "body_hash": member.body_hash, "text": member.body, "dense": list(vector)})
        schema = self._schema(job).to_dict()
        response = await self._rpc(job, "Insert", self.Prepare.row_insert_param(
            job.collection_name, rows, "", schema["fields"], enable_dynamic=False))
        if response.insert_cnt != len(rows) or list(response.IDs.str_id.data) != [m.chunk_id for m in members]:
            raise RuntimeFault("INDEX_INSERT_ACK_MISMATCH", 503)

    async def _search(self, job: IndexJob, field: str, data: Any, guard: str) -> tuple[str, ...]:
        request = self.Prepare.search_requests_with_expr(
            job.collection_name, field, {"metric_type": "COSINE" if field == "dense" else "BM25", "params": {}},
            1, data=[data], expr=guard, use_default_consistency=False, consistency_level=0,
        )
        response = await self._rpc(job, "Search", request)
        return tuple(response.results.ids.str_id.data)

    async def seal(self, job: IndexJob, members: tuple[IndexMember, ...],
                   samples: tuple[tuple[IndexMember, Vector], ...]) -> IndexReadiness:
        flushed = await self._rpc(job, "Flush", self.Prepare.flush_param([job.collection_name]))
        if job.collection_name not in flushed.coll_flush_ts or flushed.coll_flush_ts[job.collection_name] <= 0:
            raise RuntimeFault("INDEX_FLUSH_ACK_MISMATCH", 503)
        for _ in range(120):
            state = await self._rpc(job, "GetFlushState", self.messages.GetFlushStateRequest(
                collection_name=job.collection_name, flush_ts=flushed.coll_flush_ts[job.collection_name]))
            if state.flushed:
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeFault("INDEX_FLUSH_NOT_READY", 503)
        for field, name in (("dense", "dense_idx"), ("sparse", "sparse_idx")):
            for _ in range(120):
                response = await self._rpc(job, "DescribeIndex", self.Prepare.describe_index_request(job.collection_name, name))
                if len(response.index_descriptions) != 1:
                    raise RuntimeFault("INDEX_NOT_READY", 503)
                description = response.index_descriptions[0]
                parameters = {p.key: p.value for p in description.params}
                expected_metric = "COSINE" if field == "dense" else "BM25"
                expected_index = "FLAT" if field == "dense" else "SPARSE_INVERTED_INDEX"
                if (description.field_name != field or parameters.get("metric_type") != expected_metric
                    or parameters.get("index_type") != expected_index):
                    raise RuntimeFault("INDEX_SCHEMA_MISMATCH", 503)
                status = self.common.IndexState.Name(description.state)
                if status == "Finished" and description.pending_index_rows == 0:
                    break
                if status == "Failed":
                    raise RuntimeFault("INDEX_BUILD_FAILED", 503)
                await asyncio.sleep(0.5)
            else:
                raise RuntimeFault("INDEX_NOT_READY", 503)
        await self._rpc(job, "LoadCollection", self.Prepare.load_collection(job.collection_name))
        for _ in range(120):
            loaded = await self._rpc(job, "GetLoadState", self.messages.GetLoadStateRequest(collection_name=job.collection_name))
            if self.common.LoadState.Name(loaded.state) == "LoadStateLoaded":
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeFault("INDEX_NOT_LOADED", 503)
        # Strong全collection计数防重复/额外行；过滤回读全体ID/版本/hash，不读回正文。
        count = await self._rpc(job, "Query", self.Prepare.query_request(
            job.collection_name, "", ["count(*)"], [], use_default_consistency=False, consistency_level=0))
        if not count.fields_data or list(count.fields_data[0].scalars.long_data.data) != [len(members)]:
            raise RuntimeFault("INDEX_COUNT_MISMATCH", 503)
        actual = await self._rpc(job, "Query", self.Prepare.query_request(
            job.collection_name, self._guard(job), ["id", "version_id", "body_hash"], [],
            limit=len(members) + 1, use_default_consistency=False, consistency_level=0))
        columns = {f.field_name: list(f.scalars.string_data.data) for f in actual.fields_data}
        if set(columns) != {"id", "version_id", "body_hash"} or any(len(v) != len(members) for v in columns.values()):
            raise RuntimeFault("INDEX_MEMBERS_MISMATCH", 503)
        found = sorted(zip(columns["id"], columns["version_id"], columns["body_hash"], strict=True))
        if found != sorted((m.chunk_id, m.version_id, m.body_hash) for m in members):
            raise RuntimeFault("INDEX_MEMBERS_MISMATCH", 503)
        expected_samples = sample_members(members)
        if tuple(m.chunk_id for m, _ in samples) != tuple(m.chunk_id for m in expected_samples):
            raise RuntimeFault("INDEX_SAMPLES_MISMATCH", 503)
        for member, vector in samples:
            guard = self._guard(job) + " and id == " + json.dumps(member.chunk_id)
            stored = await self._rpc(job, "Query", self.Prepare.query_request(
                job.collection_name, guard, ["id", "text", "dense"], [], limit=1,
                use_default_consistency=False, consistency_level=0))
            sample_fields = {field.field_name: field for field in stored.fields_data}
            if set(sample_fields) != {"id", "text", "dense"}:
                raise RuntimeFault("INDEX_SAMPLE_CONTENT_MISMATCH", 503)
            actual_vector = list(sample_fields["dense"].vectors.float_vector.data)
            if (list(sample_fields["id"].scalars.string_data.data) != [member.chunk_id]
                or list(sample_fields["text"].scalars.string_data.data) != [member.body]
                or len(actual_vector) != len(vector)
                or any(not math.isfinite(actual) or abs(actual - expected) > 1e-5 * (1 + abs(expected))
                       for actual, expected in zip(actual_vector, vector, strict=True))):
                raise RuntimeFault("INDEX_SAMPLE_CONTENT_MISMATCH", 503)
            for field, value in (("dense", list(vector)), ("sparse", member.body)):
                if await self._search(job, field, value, guard) != (member.chunk_id,):
                    raise RuntimeFault("INDEX_SAMPLE_RECALL_FAILED", 503)
        return IndexReadiness(generation_id=job.generation_id, binding_hash=job.binding.content_hash(),
                              members_hash=member_hash(members), row_count=len(members), dense_ready=True,
                              sparse_ready=True, sample_ids=tuple(m.chunk_id for m in expected_samples))
