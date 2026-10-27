"""M3-07治理/配置/响应边界用例；仅编写，当前未运行。"""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from harness.bootstrap.providers import ModelConfigurationError
from harness.bootstrap.retrieval_models import load_retrieval_runtime
from harness.contracts.hashing import canonical_hash
from harness.contracts.retrieval_models import (
    EmbeddingRequest, RerankEvidence, RerankRequest, RetrievalLimits,
)
from harness.contracts.runtime import RuntimeFault
from harness.providers.registry import ProviderResolutionError
from harness.providers.retrieval import resolve_retrieval_profile, retrieval_registry
from harness.runtime.retrieval_models import GovernedRetrievalGateway

from tests.retrieval_fixtures import (
    ENVIRONMENT, Authority, EmbeddingTransport, MemoryLedger, RerankTransport,
    retrieval_values, write_deployment,
)


def gateway():
    config, _, bindings, scope = retrieval_values()
    ledger, authority, transport = MemoryLedger(), Authority(), EmbeddingTransport()
    instance = GovernedRetrievalGateway(
        ledger=ledger, authority=authority, embedding=transport, reranking=RerankTransport(),
        embedding_binding=bindings[0], rerank_binding=bindings[1], limits=config.limits,
        configuration_guard=lambda: None,
    )
    return instance, ledger, authority, transport, scope


def test_disabled_rag_reads_no_configuration(tmp_path, monkeypatch):
    def forbidden(root):
        raise AssertionError("关闭RAG不得读取配置")
    monkeypatch.setattr("harness.bootstrap.retrieval_models.model_values", forbidden)
    assert load_retrieval_runtime(tmp_path, enabled=False) is None


def test_enabled_rag_requires_real_governance_ports(tmp_path):
    with pytest.raises(ModelConfigurationError, match="RETRIEVAL_GOVERNANCE_REQUIRED"):
        load_retrieval_runtime(tmp_path, enabled=True)


@pytest.mark.parametrize("change", ["endpoint", "dimensions", "expired", "revoked", "environment"])
def test_resolver_rejects_untrusted_binding(change):
    config, proofs, _, _ = retrieval_values()
    revoked = frozenset()
    environment = ENVIRONMENT
    if change == "endpoint":
        config = config.model_copy(update={"connections": (
            config.connections[0].model_copy(update={"base_url": config.connections[1].base_url}),
            config.connections[1],
        )})
    elif change == "dimensions":
        config = config.model_copy(update={"profiles": (
            config.profiles[0].model_copy(update={"dimensions": 3}), config.profiles[1],
        )})
    elif change == "expired":
        proofs = (proofs[0].model_copy(update={
            "verified_at": datetime.now(UTC) - timedelta(hours=2),
            "expires_at": datetime.now(UTC) - timedelta(hours=1),
        }), proofs[1])
    elif change == "revoked":
        revoked = frozenset({proofs[0].ref})
    else:
        environment = "0" * 64
    with pytest.raises(ProviderResolutionError):
        resolve_retrieval_profile(
            retrieval_registry(), config, config.embedding_profile_ref, task="embedding",
            evidence=proofs, revoked_refs=revoked,
            environment_profile_hash=environment, now=datetime.now(UTC),
        )


def test_rerank_requires_bound_evidence_and_batch_limits():
    with pytest.raises(ValidationError):
        RerankRequest(query="问题", documents=("正文",), top_n=1)
    with pytest.raises(ValidationError):
        RerankRequest(query="问题", documents=("正文",), top_n=1, evidence=(
            RerankEvidence(chunk_id="c", version_id="v", body_hash=canonical_hash("其他正文")),
        ))
    with pytest.raises(ValidationError):
        EmbeddingRequest(texts=("文本",) * 11)


async def test_success_replay_and_revocation_never_resend():
    instance, ledger, authority, transport, scope = gateway()
    result = await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert result.usage is None
    assert await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",))) == result
    assert transport.calls == ledger.calls == 1
    authority.denied = True
    with pytest.raises(RuntimeFault, match="FORBIDDEN"):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert transport.calls == 1


async def test_denied_rerank_never_reaches_ledger_or_transport():
    instance, ledger, authority, _, scope = gateway()
    authority.denied = True
    request = RerankRequest(query="问题", documents=("正文",), top_n=1, evidence=(
        RerankEvidence(chunk_id="c", version_id="v", body_hash=canonical_hash("正文")),
    ))
    with pytest.raises(RuntimeFault, match="FORBIDDEN"):
        await instance.rerank(scope, "rank-1", request)
    assert ledger.calls == 0


async def test_cancel_records_unknown_without_retry_or_budget_reset():
    instance, ledger, _, transport, scope = gateway()
    transport.cancel = True
    with pytest.raises(asyncio.CancelledError):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert ledger.failure["billing_status"] == "unknown"
    with pytest.raises(RuntimeFault, match="RETRIEVAL_ATTEMPT_NOT_REPLAYABLE"):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert transport.calls == ledger.calls == 1


async def test_changed_input_and_result_binding_fail_closed():
    instance, ledger, _, transport, scope = gateway()
    await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    with pytest.raises(RuntimeFault, match="RETRIEVAL_CALL_ID_CONFLICT"):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("不同输入",)))
    transport.bad_binding = True
    with pytest.raises(RuntimeFault, match="RETRIEVAL_RESPONSE_BINDING_MISMATCH"):
        await instance.embed(scope, "batch-2", EmbeddingRequest(texts=("中文",)))
    assert ledger.failure["billing_status"] == "unknown"


async def test_cumulative_limit_blocks_new_call_ids():
    instance, _, _, transport, scope = gateway()
    instance.limits = RetrievalLimits(max_calls=1)
    await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    with pytest.raises(RuntimeFault, match="RETRIEVAL_BUDGET_EXCEEDED"):
        await instance.embed(scope, "batch-2", EmbeddingRequest(texts=("中文",)))
    assert transport.calls == 1


async def test_post_dispatch_revocation_keeps_result_but_blocks_delivery():
    instance, ledger, authority, transport, scope = gateway()
    original = transport.embed

    async def revoke_after_response(request):
        result = await original(request)
        authority.denied = True
        return result

    transport.embed = revoke_after_response
    with pytest.raises(RuntimeFault, match="FORBIDDEN"):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert ledger.entries["batch-1"][1] is not None
    assert transport.calls == 1


async def test_result_save_failure_leaves_attempt_blocked():
    instance, ledger, _, transport, scope = gateway()

    async def fail_save(request, result):
        raise RuntimeError("controlled-database-failure")

    ledger.succeed = fail_save
    with pytest.raises(RuntimeError, match="controlled-database-failure"):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    with pytest.raises(RuntimeFault, match="RETRIEVAL_ATTEMPT_NOT_REPLAYABLE"):
        await instance.embed(scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert transport.calls == 1 and ledger.failure is None


async def test_deadline_and_second_binding_mismatch_block_before_reserve():
    instance, ledger, _, transport, scope = gateway()
    expired = scope.model_copy(update={"deadline": datetime.now(UTC) - timedelta(seconds=1)})
    with pytest.raises(ValueError):
        await instance.embed(expired, "batch-1", EmbeddingRequest(texts=("中文",)))
    bad_scope = scope.model_copy(update={"rerank_binding_hash": "0" * 64})
    with pytest.raises(RuntimeFault, match="RETRIEVAL_SCOPE_BINDING_MISMATCH"):
        await instance.embed(bad_scope, "batch-1", EmbeddingRequest(texts=("中文",)))
    assert transport.calls == ledger.calls == 0


def test_bootstrap_fixed_snapshot_key_rotation_and_evidence_revocation(tmp_path, monkeypatch):
    _, proofs, values = write_deployment(tmp_path, monkeypatch)
    runtime = load_retrieval_runtime(
        tmp_path, enabled=True, environment_profile_hash=ENVIRONMENT,
        ledger=MemoryLedger(), authority=Authority(),
    )
    assert runtime.ready()
    snapshot = runtime.snapshot
    values["DASHSCOPE_API_KEY"] = "controlled-rotated"
    assert runtime.ready() and runtime.snapshot == snapshot
    bundle = json.loads((tmp_path / "proofs.json").read_text("utf-8"))
    bundle["revoked_refs"] = [proofs[0].ref]
    (tmp_path / "proofs.json").write_text(json.dumps(bundle), "utf-8")
    assert not runtime.ready()
    assert "controlled-rotated" not in runtime.snapshot.canonical_text


def test_bootstrap_configuration_drift_and_no_legacy_fallback(tmp_path, monkeypatch):
    write_deployment(tmp_path, monkeypatch)
    runtime = load_retrieval_runtime(
        tmp_path, enabled=True, environment_profile_hash=ENVIRONMENT,
        ledger=MemoryLedger(), authority=Authority(),
    )
    config = json.loads((tmp_path / "models.json").read_text("utf-8"))
    config["limits"]["max_calls"] += 1
    (tmp_path / "models.json").write_text(json.dumps(config), "utf-8")
    assert not runtime.ready()
