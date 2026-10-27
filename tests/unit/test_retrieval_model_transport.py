"""M3-07单次协议请求与响应边界；受控HTTP替身，不运行真实供应商。"""

import json

import pytest
from pydantic import SecretStr

from harness.adapters.models.dashscope_retrieval import DashScopeRetrievalClient
from harness.contracts.failures import ProviderError
from harness.contracts.hashing import canonical_hash
from harness.contracts.retrieval_models import (
    EmbeddingRequest, PreparedRetrievalCall, RerankEvidence, RerankRequest,
)
from tests.retrieval_fixtures import retrieval_values


@pytest.mark.parametrize("bad", ["dimensions", "indexes", "nonfinite", "model"])
async def test_embedding_protocol_rejects_unbound_response(monkeypatch, bad):
    _, _, bindings, scope = retrieval_values()
    body = {"model": "text-embedding-v4", "data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    if bad == "dimensions":
        body["data"][0]["embedding"] = [0.1]
    elif bad == "indexes":
        body["data"][0]["index"] = 1
    elif bad == "nonfinite":
        body["data"][0]["embedding"] = [float("nan"), 0.2]
    else:
        body["model"] = "other-model"

    async def send(url, payload, **kwargs):
        return json.dumps(body).encode(), "controlled-request"

    monkeypatch.setattr("harness.adapters.models.dashscope_retrieval.post_json", send)
    client = DashScopeRetrievalClient(bindings[0], lambda: SecretStr("controlled"))
    call = PreparedRetrievalCall(
        scope=scope, call_id="embedding-1", attempt_id="attempt-1",
        binding_hash=bindings[0].snapshot.content_hash,
        payload=EmbeddingRequest(texts=("中文",)),
    )
    with pytest.raises(ProviderError, match="protocol_error"):
        await client.embed(call)


async def test_single_embedding_http_uses_fixed_profile_and_actual_usage(monkeypatch):
    _, _, bindings, scope = retrieval_values()
    calls = []

    async def send(url, payload, **kwargs):
        calls.append((url, payload))
        return json.dumps({
            "model": "text-embedding-v4", "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"total_tokens": 3},
        }).encode(), "controlled-request"

    monkeypatch.setattr("harness.adapters.models.dashscope_retrieval.post_json", send)
    call = PreparedRetrievalCall(
        scope=scope, call_id="embedding-1", attempt_id="attempt-1",
        binding_hash=bindings[0].snapshot.content_hash,
        payload=EmbeddingRequest(texts=("中文",)),
    )
    result = await DashScopeRetrievalClient(
        bindings[0], lambda: SecretStr("controlled")
    ).embed(call)
    assert len(calls) == 1
    assert calls[0][0].endswith("/embeddings") and calls[0][1]["dimensions"] == 2
    assert result.usage.total_tokens == 3 and result.usage.input_tokens is None
    assert result.request_fingerprint == call.fingerprint()


async def test_rerank_sends_authorized_bodies_without_internal_evidence_ids(monkeypatch):
    _, _, bindings, scope = retrieval_values()
    calls = []

    async def send(url, payload, **kwargs):
        calls.append((url, payload))
        return json.dumps({"output": {"results": [{"index": 0, "relevance_score": 0.9}]}}).encode(), None

    monkeypatch.setattr("harness.adapters.models.dashscope_retrieval.post_json", send)
    call = PreparedRetrievalCall(
        scope=scope, call_id="rerank-1", attempt_id="attempt-1",
        binding_hash=bindings[1].snapshot.content_hash,
        payload=RerankRequest(query="问题", documents=("正文",), top_n=1, evidence=(
            RerankEvidence(chunk_id="private-chunk-id", version_id="version-1",
                           body_hash=canonical_hash("正文")),
        )),
    )
    result = await DashScopeRetrievalClient(
        bindings[1], lambda: SecretStr("controlled")
    ).rerank(call)
    assert len(calls) == 1 and calls[0][0] == bindings[1].connection.base_url
    assert calls[0][1]["parameters"] == {"top_n": 1, "return_documents": False}
    assert "private-chunk-id" not in json.dumps(calls[0][1])
    assert result.usage is None
