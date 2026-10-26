"""通义千问 HTTP 适配器草案；不自动重试，不自动发送验证请求。"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from harness.adapters.models.config import QWEN_ADAPTER_REF, QwenSettings
from harness.adapters.models.http import (
    ProviderUnavailable as ProviderUnavailable,
)
from harness.adapters.models.http import (
    decode_completion,
    http_failure,
    post_json,
    protocol_failure,
    request_id,
)
from harness.contracts.failures import ProviderErrorCode, ProviderFailure
from harness.contracts.model_envelope import prepared_chat_envelope
from harness.contracts.models import ModelResult, PreparedModelRequest

# 兼容已有调用方和边界用例的历史入口，不复制分类逻辑。
_http_failure = http_failure
_protocol_failure = protocol_failure
_request_id = request_id


class EmbeddingItem(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    index: int = Field(ge=0)
    embedding: list[float]


class RetrievalUsage(BaseModel):
    """只保留供应商实际返回的用量；缺失字段保持未知。"""

    total_tokens: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class EmbeddingResult(BaseModel):
    data: list[EmbeddingItem]
    model: str
    usage: RetrievalUsage | None = None
    request_id: str | None = None


class RerankItem(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    index: int = Field(ge=0)
    relevance_score: float


class _RerankOutput(BaseModel):
    results: list[RerankItem]


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    output: _RerankOutput
    request_id: str | None = None
    usage: RetrievalUsage | None = None


class QwenClient:
    """只实现单次供应商传输；运行预算、权限与尝试身份由统一网关治理。"""

    def __init__(self, settings: QwenSettings) -> None:
        self.settings = settings

    async def _post(
        self,
        url: str,
        body: dict[str, JsonValue],
        *,
        settings: QwenSettings,
    ) -> tuple[bytes, str | None]:
        return await post_json(
            url, body, api_key=settings.api_key, timeout_seconds=settings.timeout_seconds
        )

    async def complete(
        self,
        request: PreparedModelRequest,
    ) -> ModelResult:
        # 固定同一个配置对象直至本次传输结束，不在 await 后重新读取可变装配。
        settings = self.settings
        try:
            request.require_dispatchable(datetime.now(UTC))
            body = prepared_chat_envelope(request).as_object()
        except ValueError:
            raise ProviderUnavailable(
                ProviderFailure(
                    code=ProviderErrorCode.REQUEST_NOT_DISPATCHABLE,
                    dispatch_status="not_sent",
                    billing_status="not_billable",
                )
            ) from None
        if (
            request.route.provider != "dashscope"
            or request.route.adapter_ref != QWEN_ADAPTER_REF
            or request.route.model != settings.chat_model
            or request.route.configuration_hash != settings.chat_configuration_hash()
            or request.max_output_tokens > settings.max_output_tokens
        ):
            raise ProviderUnavailable(
                ProviderFailure(
                    code=ProviderErrorCode.REQUEST_BINDING_MISMATCH,
                    dispatch_status="not_sent",
                    billing_status="not_billable",
                )
            )
        raw, request_id = await self._post(
            settings.base_url.rstrip("/") + "/chat/completions",
            body,
            settings=settings,
        )
        return decode_completion(raw, request_id, request)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        settings = self.settings
        if not 0 < len(texts) <= 10 or any(not text or len(text) > 8000 for text in texts):
            raise ValueError("Embedding 输入必须为1至10条有界文本")
        raw, request_id = await self._post(
            settings.base_url.rstrip("/") + "/embeddings",
            {
                "model": settings.embedding_model,
                "input": list(texts),
                "dimensions": settings.embedding_dimensions,
                "encoding_format": "float",
            },
            settings=settings,
        )
        try:
            result = EmbeddingResult.model_validate_json(raw)
        except ValidationError:
            raise _protocol_failure(request_id) from None
        if sorted(item.index for item in result.data) != list(range(len(texts))):
            raise _protocol_failure(request_id)
        if any(len(item.embedding) != settings.embedding_dimensions for item in result.data):
            raise _protocol_failure(request_id)
        return result.model_copy(
            update={"request_id": _request_id(result.request_id) or request_id}
        )

    async def rerank(self, query: str, documents: list[str], *, top_n: int = 5) -> RerankResult:
        settings = self.settings
        if not query or len(query) > 8000 or not 0 < len(documents) <= 40:
            raise ValueError("Rerank 输入数量或查询长度超限")
        if not 0 < top_n <= len(documents) or any(
            not text or len(text) > 8000 for text in documents
        ):
            raise ValueError("Rerank TopN或文档长度不合法")
        raw, request_id = await self._post(
            settings.rerank_url,
            {
                "model": settings.rerank_model,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {"top_n": top_n, "return_documents": False},
            },
            settings=settings,
        )
        try:
            result = RerankResult.model_validate_json(raw)
        except ValidationError:
            raise _protocol_failure(request_id) from None
        indexes = [item.index for item in result.output.results]
        if (
            len(indexes) != top_n
            or len(set(indexes)) != len(indexes)
            or any(index >= len(documents) for index in indexes)
        ):
            raise _protocol_failure(request_id)
        return result.model_copy(
            update={"request_id": _request_id(result.request_id) or request_id}
        )
