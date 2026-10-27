"""可选检索模型持久账本；短事务预留、成功重读、未知/失败禁止自动重发。"""

from pydantic import TypeAdapter
from psycopg.types.json import Jsonb

from harness.adapters.postgres.connection import PostgresUnitOfWork
from harness.contracts.hashing import canonical_json
from harness.contracts.retrieval_models import (
    PreparedRetrievalCall, RetrievalLimits, RetrievalReservation, RetrievalResult,
)
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot

_RESULT: TypeAdapter[RetrievalResult] = TypeAdapter(RetrievalResult)


class PostgresRetrievalLedger:
    def __init__(self, dsn: str) -> None:
        self.uow = PostgresUnitOfWork(dsn)

    async def reserve(
        self, request: PreparedRetrievalCall, limits: RetrievalLimits
    ) -> RetrievalReservation:
        scope = request.scope
        scope_data, limits_data = scope.model_dump(mode="json"), limits.model_dump(mode="json")
        input_bytes = len(canonical_json(request.payload.model_dump(mode="json")).encode("utf-8"))
        async with self.uow.transaction() as conn:
            await conn.execute(
                "INSERT INTO harness_rag.model_contexts(context_id,tenant_id,application_id,"
                "owner_kind,owner_id,knowledge_base_id,scope,limits,deadline) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (scope.context_id, scope.tenant_id, scope.application_id, scope.owner_kind,
                 scope.owner_id, scope.knowledge_base_id, Jsonb(scope_data), Jsonb(limits_data), scope.deadline),
            )
            context = await (await conn.execute(
                "SELECT *,clock_timestamp() AS db_now FROM harness_rag.model_contexts "
                "WHERE context_id=%s FOR UPDATE", (scope.context_id,),
            )).fetchone()
            if context is None or context["scope"] != scope_data or context["limits"] != limits_data:
                raise RuntimeFault("RETRIEVAL_CONTEXT_BINDING_MISMATCH", 409)
            if context["db_now"] >= context["deadline"]:
                raise RuntimeFault("RETRIEVAL_DEADLINE_EXCEEDED", 409)
            existing = await (await conn.execute(
                "SELECT * FROM harness_rag.model_calls WHERE context_id=%s AND call_id=%s",
                (scope.context_id, request.call_id),
            )).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != request.fingerprint():
                    raise RuntimeFault("RETRIEVAL_CALL_ID_CONFLICT", 409)
                if existing["state"] != "succeeded":
                    raise RuntimeFault("RETRIEVAL_ATTEMPT_NOT_REPLAYABLE", 409)
                restored = request.model_copy(update={"attempt_id": existing["attempt_id"]})
                return RetrievalReservation(request=restored, cached_result=_RESULT.validate_python(existing["result"]))
            if (context["calls_used"] >= limits.max_calls
                or context["input_bytes_used"] + input_bytes > limits.max_input_bytes):
                raise RuntimeFault("RETRIEVAL_BUDGET_EXCEEDED", 429)
            await conn.execute(
                "UPDATE harness_rag.model_contexts SET calls_used=calls_used+1,"
                "input_bytes_used=input_bytes_used+%s WHERE context_id=%s", (input_bytes, scope.context_id),
            )
            await conn.execute(
                "INSERT INTO harness_rag.model_calls(context_id,call_id,attempt_id,request_fingerprint,"
                "binding_hash,task_type,input_bytes,state) VALUES(%s,%s,%s,%s,%s,%s,%s,'reserved')",
                (scope.context_id, request.call_id, request.attempt_id, request.fingerprint(),
                 request.binding_hash, request.payload.task_type, input_bytes),
            )
            return RetrievalReservation(request=request)

    async def succeed(self, request: PreparedRetrievalCall, result: RetrievalResult) -> None:
        if (result.call_id != request.call_id or result.attempt_id != request.attempt_id
            or result.request_fingerprint != request.fingerprint()
            or result.task_type != request.payload.task_type):
            raise RuntimeFault("RETRIEVAL_RESPONSE_BINDING_MISMATCH", 409)
        async with self.uow.transaction() as conn:
            updated = await conn.execute(
                "UPDATE harness_rag.model_calls SET state='succeeded',result=%s,finished_at=clock_timestamp() "
                "WHERE context_id=%s AND call_id=%s AND attempt_id=%s AND request_fingerprint=%s "
                "AND state='reserved'",
                (Jsonb(result.model_dump(mode="json")), request.scope.context_id, request.call_id,
                 request.attempt_id, request.fingerprint()),
            )
            if updated.rowcount != 1:
                raise RuntimeFault("RETRIEVAL_RESULT_CAS_FAILED", 409)

    async def fail(self, request: PreparedRetrievalCall, failure: JsonSnapshot) -> None:
        async with self.uow.transaction() as conn:
            updated = await conn.execute(
                "UPDATE harness_rag.model_calls SET state='failed',failure=%s,finished_at=clock_timestamp() "
                "WHERE context_id=%s AND call_id=%s AND attempt_id=%s AND request_fingerprint=%s "
                "AND state='reserved'",
                (Jsonb(failure.as_object()), request.scope.context_id, request.call_id,
                 request.attempt_id, request.fingerprint()),
            )
            if updated.rowcount != 1:
                raise RuntimeFault("RETRIEVAL_FAILURE_CAS_FAILED", 409)
