"""检索供应商边界：发送前批次限制、响应完整性与未知用量，不联网。"""

import json

import pytest
from harness.adapters.models.config import QwenSettings
from harness.adapters.models.qwen import QwenClient
from harness.contracts.failures import ProviderError


class Transport(QwenClient):
    def __init__(self, response):
        super().__init__(QwenSettings(embedding_dimensions=2))
        self.response = response
        self.calls = 0

    async def _post(self, url, body, *, settings):
        self.calls += 1
        return json.dumps(self.response).encode(), "test-request-id"


@pytest.mark.parametrize("texts", [[], [""], ["文本"] * 11, ["文" * 8001]])
async def test_embedding_rejects_invalid_batch_before_dispatch(texts):
    client = Transport({})
    with pytest.raises(ValueError):
        await client.embed(texts)
    assert client.calls == 0


async def test_embedding_ten_items_and_reported_usage():
    client = Transport(
        {
            "model": "text-embedding-v4",
            "usage": {"total_tokens": 17},
            "data": [{"index": index, "embedding": [0.1, 0.2]} for index in range(10)],
        }
    )
    result = await client.embed(["测试"] * 10)
    assert client.calls == 1 and len(result.data) == 10
    assert result.usage.total_tokens == 17
    assert result.usage.input_tokens is None
    assert result.request_id == "test-request-id"


@pytest.mark.parametrize(
    "items",
    [
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 1, "embedding": [1.0, 0.0]}],
        [{"index": 0, "embedding": [float("nan"), 0.0]}],
        [{"index": 0, "embedding": [float("inf"), 0.0]}],
        [],
    ],
)
async def test_embedding_invalid_response_is_protocol_failure(items):
    client = Transport({"model": "text-embedding-v4", "data": items})
    with pytest.raises(ProviderError) as failure:
        await client.embed(["测试"])
    assert str(failure.value) == "protocol_error"
    assert client.calls == 1


async def test_missing_usage_remains_unknown():
    client = Transport(
        {"model": "text-embedding-v4", "data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    )
    assert (await client.embed(["测试"])).usage is None


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 0, "relevance_score": float("inf")}],
        [{"index": 2, "relevance_score": 0.5}],
        [],
    ],
)
async def test_rerank_invalid_response_is_protocol_failure(results):
    client = Transport({"output": {"results": results}})
    with pytest.raises(ProviderError):
        await client.rerank("问题", ["资料"], top_n=1)


async def test_rerank_rejects_empty_document_before_dispatch():
    client = Transport({})
    with pytest.raises(ValueError):
        await client.rerank("问题", [""], top_n=1)
    assert client.calls == 0


async def test_rerank_preserves_actual_order_usage_and_safe_request_id():
    client = Transport(
        {
            "request_id": "unsafe\nvalue",
            "usage": {"total_tokens": 23},
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        }
    )
    result = await client.rerank("问题", ["甲", "乙"], top_n=2)
    assert [item.index for item in result.output.results] == [1, 0]
    assert result.usage.total_tokens == 23
    assert result.request_id == "test-request-id"
