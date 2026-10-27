"""M3-07真实PG事务用例定义；本轮未执行，不调用模型供应商。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from harness.adapters.postgres.connection import PostgresUnitOfWork
from harness.adapters.retrieval_models_postgres import PostgresRetrievalLedger
from harness.contracts.retrieval_models import (
    EmbeddingRequest, EmbeddingResult, PreparedRetrievalCall, RetrievalLimits,
)
from harness.contracts.hashing import canonical_json
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.runtime.retrieval_models import GovernedRetrievalGateway
from tests.retrieval_fixtures import Authority, EmbeddingTransport, RerankTransport, retrieval_values

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


class ScopedConnection:
    """仅替换固定schema名，SQL/事务仍由真实PG执行；不碰生产或其他用例表。"""

    def __init__(self, connection, namespace):
        self.connection, self.namespace = connection, namespace

    async def execute(self, statement, parameters=()):
        return await self.connection.execute(
            statement.replace("harness_rag.", self.namespace + "."), parameters
        )


class ScopedUow:
    def __init__(self, dsn, namespace):
        self.uow, self.namespace = PostgresUnitOfWork(dsn), namespace

    @asynccontextmanager
    async def transaction(self):
        async with self.uow.transaction() as connection:
            yield ScopedConnection(connection, self.namespace)


@pytest.fixture
def retrieval_ledger(pg_schema):
    dsn, namespace = pg_schema
    migration = (ROOT / "migrations/rag/0001_retrieval_models.sql").read_text("utf-8")
    with psycopg.connect(dsn) as conn:
        conn.execute(migration.replace("harness_rag", namespace), prepare=False)
    ledger = PostgresRetrievalLedger(dsn)
    ledger.uow = ScopedUow(dsn, namespace)
    return ledger


def request(scope, bindings, call_id="batch-1"):
    return PreparedRetrievalCall(
        scope=scope, call_id=call_id, attempt_id=uuid4().hex,
        binding_hash=bindings[0].snapshot.content_hash,
        payload=EmbeddingRequest(texts=("合成中文资料",)),
    )


async def test_concurrent_same_call_only_reserves_once_and_success_is_cached(retrieval_ledger):
    config, _, bindings, scope = retrieval_values()
    call = request(scope, bindings)
    results = await asyncio.gather(
        retrieval_ledger.reserve(call, config.limits),
        retrieval_ledger.reserve(call.model_copy(update={"attempt_id": uuid4().hex}), config.limits),
        return_exceptions=True,
    )
    assert sum(isinstance(item, RuntimeFault) for item in results) == 1
    result = EmbeddingResult(call_id=call.call_id, attempt_id=call.attempt_id,
                             request_fingerprint=call.fingerprint(), vectors=((0.1, 0.2),))
    # 并发赢家的attempt由实际持久预留决定，不假设调度顺序。
    winner = next(item for item in results if not isinstance(item, BaseException))
    result = result.model_copy(update={"attempt_id": winner.request.attempt_id})
    await retrieval_ledger.succeed(winner.request, result)
    cached = await retrieval_ledger.reserve(call, config.limits)
    assert cached.cached_result == result
    assert cached.request.attempt_id == winner.request.attempt_id


async def test_unknown_attempt_and_new_context_cannot_reset_owner_budget(retrieval_ledger):
    _, _, bindings, scope = retrieval_values()
    call = request(scope, bindings)
    limits = RetrievalLimits(max_calls=1)
    await retrieval_ledger.reserve(call, limits)
    await retrieval_ledger.fail(call, JsonSnapshot.capture({
        "code": "CANCELLED", "dispatch_status": "unknown", "billing_status": "unknown",
    }))
    with pytest.raises(RuntimeFault, match="RETRIEVAL_ATTEMPT_NOT_REPLAYABLE"):
        await retrieval_ledger.reserve(call, limits)
    with pytest.raises(RuntimeFault, match="RETRIEVAL_BUDGET_EXCEEDED"):
        await retrieval_ledger.reserve(request(scope, bindings, "batch-2"), limits)
    new_context = scope.model_copy(update={"context_id": uuid4().hex})
    with pytest.raises(RuntimeFault, match="RETRIEVAL_CONTEXT_BINDING_MISMATCH"):
        await retrieval_ledger.reserve(request(new_context, bindings), limits)


async def test_real_pg_gateway_cache_and_cross_identity_refusal(retrieval_ledger):
    config, _, bindings, scope = retrieval_values()
    transport, authority = EmbeddingTransport(), Authority()
    gateway = GovernedRetrievalGateway(
        ledger=retrieval_ledger, authority=authority, embedding=transport, reranking=RerankTransport(),
        embedding_binding=bindings[0], rerank_binding=bindings[1], limits=config.limits,
        configuration_guard=lambda: None,
    )
    payload = EmbeddingRequest(texts=("受控中文资料",))
    result = await gateway.embed(scope, "batch-1", payload)
    assert await gateway.embed(scope, "batch-1", payload) == result
    assert transport.calls == 1 and result.usage is None
    other_identity = scope.model_copy(update={"subject_id": "another-user"})
    with pytest.raises(RuntimeFault, match="RETRIEVAL_CONTEXT_BINDING_MISMATCH"):
        await gateway.embed(other_identity, "batch-1", payload)


async def test_same_call_changed_input_is_conflict_and_failed_cas_keeps_truth(retrieval_ledger):
    config, _, bindings, scope = retrieval_values()
    call = request(scope, bindings)
    await retrieval_ledger.reserve(call, config.limits)
    changed = call.model_copy(update={"payload": EmbeddingRequest(texts=("其他资料",))})
    with pytest.raises(RuntimeFault, match="RETRIEVAL_CALL_ID_CONFLICT"):
        await retrieval_ledger.reserve(changed, config.limits)
    result = EmbeddingResult(call_id=call.call_id, attempt_id=call.attempt_id,
                             request_fingerprint=call.fingerprint(), vectors=((0.1, 0.2),))
    await retrieval_ledger.succeed(call, result)
    with pytest.raises(RuntimeFault, match="RETRIEVAL_FAILURE_CAS_FAILED"):
        await retrieval_ledger.fail(call, JsonSnapshot.capture({"code": "LATE_FAILURE"}))
    assert (await retrieval_ledger.reserve(call, config.limits)).cached_result == result


async def test_input_bytes_budget_and_limits_change_do_not_reset_context(retrieval_ledger):
    _, _, bindings, scope = retrieval_values()
    call = request(scope, bindings)
    size = len(canonical_json(call.payload.model_dump(mode="json")).encode("utf-8"))
    limits = RetrievalLimits(max_calls=3, max_input_bytes=size)
    await retrieval_ledger.reserve(call, limits)
    with pytest.raises(RuntimeFault, match="RETRIEVAL_BUDGET_EXCEEDED"):
        await retrieval_ledger.reserve(request(scope, bindings, "batch-2"), limits)
    with pytest.raises(RuntimeFault, match="RETRIEVAL_CONTEXT_BINDING_MISMATCH"):
        await retrieval_ledger.reserve(
            request(scope, bindings, "batch-3"),
            limits.model_copy(update={"max_input_bytes": size * 2}),
        )
