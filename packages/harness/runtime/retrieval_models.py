"""统一治理链的检索专用入口：鉴权、不可变绑定、持久预留、单次派发与结果落账。"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import anyio
from pydantic import TypeAdapter

from harness.contracts.failures import ProviderError
from harness.contracts.retrieval_models import (
    EmbeddingPort, EmbeddingRequest, EmbeddingResult, PreparedRetrievalCall, RerankPort,
    RerankRequest, RerankResult, RetrievalAuthority, RetrievalLedger, RetrievalLimits,
    RetrievalModelBinding, RetrievalPayload, RetrievalResult, RetrievalScope, validate_retrieval_call,
)
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot

_RESULT: TypeAdapter[RetrievalResult] = TypeAdapter(RetrievalResult)


class GovernedRetrievalGateway:
    def __init__(
        self, *, ledger: RetrievalLedger, authority: RetrievalAuthority,
        embedding: EmbeddingPort, reranking: RerankPort,
        embedding_binding: RetrievalModelBinding, rerank_binding: RetrievalModelBinding,
        limits: RetrievalLimits, configuration_guard: Callable[[], None],
    ) -> None:
        self.ledger, self.authority = ledger, authority
        self.embedding, self.reranking = embedding, reranking
        self.embedding_binding, self.rerank_binding = embedding_binding, rerank_binding
        self.limits, self.configuration_guard = limits, configuration_guard

    async def embed(
        self, scope: RetrievalScope, call_id: str, request: EmbeddingRequest
    ) -> EmbeddingResult:
        result = await self._call(scope, call_id, request, self.embedding_binding)
        if not isinstance(result, EmbeddingResult):
            raise RuntimeFault("RETRIEVAL_RESPONSE_TASK_MISMATCH")
        return result

    async def rerank(
        self, scope: RetrievalScope, call_id: str, request: RerankRequest
    ) -> RerankResult:
        result = await self._call(scope, call_id, request, self.rerank_binding)
        if not isinstance(result, RerankResult):
            raise RuntimeFault("RETRIEVAL_RESPONSE_TASK_MISMATCH")
        return result

    async def _call(
        self, scope: RetrievalScope, call_id: str, payload: RetrievalPayload,
        binding: RetrievalModelBinding,
    ) -> RetrievalResult:
        # 重验传入对象，不能通过model_copy绕过冻结边界的大小/身份限制。
        candidate = PreparedRetrievalCall.model_validate_json(PreparedRetrievalCall(
            scope=scope, call_id=call_id, attempt_id=uuid4().hex,
            binding_hash=binding.snapshot.content_hash, payload=payload,
        ).model_dump_json())
        if (scope.embedding_binding_hash != self.embedding_binding.snapshot.content_hash
            or scope.rerank_binding_hash != self.rerank_binding.snapshot.content_hash):
            raise RuntimeFault("RETRIEVAL_SCOPE_BINDING_MISMATCH", 409)
        await self.authority.authorize(candidate.scope, candidate.payload)
        self.configuration_guard()
        validate_retrieval_call(candidate, binding, datetime.now(UTC))
        reservation = await self.ledger.reserve(candidate, self.limits)
        request = reservation.request
        if (request.fingerprint() != candidate.fingerprint()
            or request.payload != candidate.payload or request.scope != candidate.scope):
            raise RuntimeFault("RETRIEVAL_RESERVATION_BINDING_MISMATCH")
        if reservation.cached_result is not None:
            # 缓存同样重新鉴权/检查配置，不因成功过而绕过撤权或证据失效。
            await self.authority.authorize(request.scope, request.payload)
            self.configuration_guard()
            validate_retrieval_call(request, binding, datetime.now(UTC))
            return self._validate_result(request, binding, reservation.cached_result)
        try:
            await self.authority.authorize(request.scope, request.payload)
            self.configuration_guard()
            validate_retrieval_call(request, binding, datetime.now(UTC))
        except BaseException:
            await self._fail(request, {"code": "RETRIEVAL_PRE_DISPATCH_BLOCKED",
                                       "dispatch_status": "not_sent", "billing_status": "not_billable"})
            raise
        try:
            remaining = (request.scope.deadline - datetime.now(UTC)).total_seconds()
            async with asyncio.timeout(min(binding.connection.timeout_seconds, max(0.001, remaining))):
                result = (await self.embedding.embed(request) if isinstance(request.payload, EmbeddingRequest)
                          else await self.reranking.rerank(request))
            result = self._validate_result(request, binding, result)
        except ProviderError as exc:
            await self._fail(request, exc.failure.model_dump(mode="json"))
            raise
        except BaseException:
            await self._fail(request, {"code": "RETRIEVAL_OUTCOME_UNKNOWN",
                                       "dispatch_status": "unknown", "billing_status": "unknown"})
            raise
        # 响应事实先保存；落账失败保留reserved，不能重发或伪造成无费用。
        await self.ledger.succeed(request, result)
        # 调用期间撤权不抹掉供应商事实，但阻断结果交付；缓存再次读取也须鉴权。
        await self.authority.authorize(request.scope, request.payload)
        validate_retrieval_call(request, binding, datetime.now(UTC))
        return result

    def _validate_result(
        self, request: PreparedRetrievalCall, binding: RetrievalModelBinding, result: RetrievalResult
    ) -> RetrievalResult:
        result = _RESULT.validate_json(result.model_dump_json())
        if (result.call_id != request.call_id or result.attempt_id != request.attempt_id
            or result.request_fingerprint != request.fingerprint()
            or result.task_type != request.payload.task_type):
            raise RuntimeFault("RETRIEVAL_RESPONSE_BINDING_MISMATCH")
        if isinstance(result, EmbeddingResult) and isinstance(request.payload, EmbeddingRequest):
            if (len(result.vectors) != len(request.payload.texts)
                or any(len(vector) != binding.profile.dimensions for vector in result.vectors)):
                raise RuntimeFault("RETRIEVAL_RESPONSE_DIMENSION_MISMATCH")
        if isinstance(result, RerankResult) and isinstance(request.payload, RerankRequest):
            indexes = [item.index for item in result.items]
            if (len(indexes) != request.payload.top_n or len(set(indexes)) != len(indexes)
                or any(index >= len(request.payload.documents) for index in indexes)):
                raise RuntimeFault("RETRIEVAL_RESPONSE_INDEX_MISMATCH")
        return result

    async def _fail(self, request: PreparedRetrievalCall, data: object) -> None:
        # 取消仍保留失败事实；清理失败时pending继续阻断，不清除额度或重复派发。
        with anyio.CancelScope(shield=True):
            async with asyncio.timeout(5):
                await self.ledger.fail(request, JsonSnapshot.capture(data))
