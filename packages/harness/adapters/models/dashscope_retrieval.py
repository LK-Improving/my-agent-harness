"""独立检索协议客户端；复用已测响应Schema和单次HTTP，不使用聊天配置或complete。"""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import JsonValue, SecretStr

from harness.adapters.models.http import failure, post_json, protocol_failure, request_id
from harness.adapters.models.qwen import EmbeddingResult as WireEmbeddingResult
from harness.adapters.models.qwen import RerankResult as WireRerankResult
from harness.contracts.failures import ProviderErrorCode
from harness.contracts.retrieval_models import (
    EmbeddingRequest, EmbeddingResult, PreparedRetrievalCall, RerankItem, RerankRequest,
    RerankResult, RetrievalModelBinding, RetrievalUsage, validate_retrieval_call,
)


class DashScopeRetrievalClient:
    def __init__(
        self, binding: RetrievalModelBinding, credential: Callable[[], SecretStr]
    ) -> None:
        self.binding = binding
        self.credential = credential

    async def _send(
        self, request: PreparedRetrievalCall, body: dict[str, JsonValue], suffix: str
    ) -> tuple[bytes, str | None]:
        try:
            validate_retrieval_call(request, self.binding, datetime.now(UTC))
            credential = self.credential()
        except (ValueError, OSError):
            raise failure(ProviderErrorCode.REQUEST_BINDING_MISMATCH) from None
        remaining = (request.scope.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise failure(ProviderErrorCode.REQUEST_NOT_DISPATCHABLE)
        return await post_json(
            self.binding.connection.base_url + suffix, body, api_key=credential,
            timeout_seconds=min(self.binding.connection.timeout_seconds, remaining),
        )

    async def embed(self, request: PreparedRetrievalCall) -> EmbeddingResult:
        if not isinstance(request.payload, EmbeddingRequest):
            raise failure(ProviderErrorCode.REQUEST_BINDING_MISMATCH)
        raw, reference = await self._send(request, {
            "model": self.binding.profile.model_id, "input": list(request.payload.texts),
            "dimensions": self.binding.profile.dimensions, "encoding_format": "float",
        }, "/embeddings")
        try:
            result = WireEmbeddingResult.model_validate_json(raw)
            indexes = sorted(item.index for item in result.data)
            if (result.model != self.binding.profile.model_id
                or indexes != list(range(len(request.payload.texts)))
                or any(len(item.embedding) != self.binding.profile.dimensions for item in result.data)):
                raise ValueError("向量结果身份/维度/数量不一致")
            return EmbeddingResult(
                call_id=request.call_id, attempt_id=request.attempt_id,
                request_fingerprint=request.fingerprint(), request_id=request_id(result.request_id) or reference,
                usage=RetrievalUsage.model_validate(result.usage.model_dump()) if result.usage else None,
                vectors=tuple(tuple(item.embedding) for item in sorted(result.data, key=lambda item: item.index)),
            )
        except ValueError:
            raise protocol_failure(reference) from None

    async def rerank(self, request: PreparedRetrievalCall) -> RerankResult:
        if not isinstance(request.payload, RerankRequest):
            raise failure(ProviderErrorCode.REQUEST_BINDING_MISMATCH)
        payload = request.payload
        raw, reference = await self._send(request, {
            "model": self.binding.profile.model_id,
            "input": {"query": payload.query, "documents": list(payload.documents)},
            "parameters": {"top_n": payload.top_n, "return_documents": False},
        }, "")
        try:
            result = WireRerankResult.model_validate_json(raw)
            indexes = [item.index for item in result.output.results]
            if (len(indexes) != payload.top_n or len(set(indexes)) != len(indexes)
                or any(index >= len(payload.documents) for index in indexes)):
                raise ValueError("重排结果身份或数量不一致")
            return RerankResult(
                call_id=request.call_id, attempt_id=request.attempt_id,
                request_fingerprint=request.fingerprint(), request_id=request_id(result.request_id) or reference,
                usage=RetrievalUsage.model_validate(result.usage.model_dump()) if result.usage else None,
                items=tuple(RerankItem(index=item.index, relevance_score=item.relevance_score)
                            for item in result.output.results),
            )
        except ValueError:
            raise protocol_failure(reference) from None
