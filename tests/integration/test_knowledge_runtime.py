"""实际create_agent/PG/执行器/提交/API-SSE接线定义；模型/召回为受控替身，本轮未运行。"""

import json
import re
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row

from domain_tools.knowledge.completion import KnowledgeCompletionValidator
from domain_tools.knowledge.context import KnowledgeContext
from domain_tools.knowledge.retrieval import registered_knowledge, require_query
from harness.adapters.development_auth import DevelopmentAuthenticator
from harness.bootstrap.api import create_app
from harness.bootstrap.assembly import DevelopmentAssembly
from harness.contracts.execution import ExecutionProfile
from harness.contracts.hashing import RunRequest
from harness.contracts.models import ModelResult, ModelUsage
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import CancelRequest, ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ToolBinding
from harness.graphs.agent import AgentEngine
from harness.runtime.models import GovernedModelGateway
from harness.tools.reads import ReadCoordinator
from tests.integration.test_indexing_postgres import index_repository as index_repository
from tests.integration.test_ingestion_postgres import repository as repository
from tests.integration.test_retrieval_pipeline_postgres import Connection, Uow
from tests.integration.test_retrieval_pipeline_postgres import retrieval_case as retrieval_case
from tests.unit.test_ingestion import actor as ingestion_actor

pytestmark = pytest.mark.integration


def actor():
    original = ingestion_actor()
    return original.model_copy(update={"roles": original.roles | {"user"}})


class AnswerModel:
    def __init__(self):
        self.calls, self.envelopes = 0, []

    async def complete(self, request):
        self.calls += 1
        messages = request.messages.as_object_array()
        self.envelopes.append(messages)
        data = json.loads(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))["data"]
        evidence = data["results"][0]["evidence"]
        if evidence:
            e = evidence[0]
            ref = f"{e['snapshot_id']}/{e['version_id']}/{e['chunk_id']}"
            answer = {"answer": "资料说明[[source:" + ref + "]]", "answer_kind": "answer",
                      "task_outcome": "achieved", "evidence_refs": [ref]}
        else:
            answer = {"answer": "暂无授权证据，请补充问题。", "answer_kind": "clarification",
                      "task_outcome": "needs_input", "evidence_refs": []}
        return ModelResult(model=request.route.model, content=json.dumps(answer, ensure_ascii=False),
                             finish_reason="stop", request_id="controlled-answer", usage=ModelUsage(input_tokens=1, output_tokens=1),
                             model_call_id=request.model_call_id, model_attempt_id=request.model_attempt_id,
                             request_fingerprint=request.fingerprint())


class ApiConnection(Connection):
    async def execute(self, statement, parameters=()):
        # 隔离fixture改了core表名，ON CONFLICT中的裸表引用也须对应；生产SQL不改。
        statement = re.sub(r"(?<![\w.])(read_windows|request_windows)\.",
                           lambda match: "runtime_" + match.group(1) + ".", statement)
        return await super().execute(statement, parameters)


class ApiUow(Uow):
    @asynccontextmanager
    async def transaction(self):
        async with self.uow.transaction() as connection:
            yield ApiConnection(connection, self.namespace)


@pytest.fixture(params=("memory", "postgres"))
async def knowledge_case(retrieval_case, request):
    c = retrieval_case
    store = c.repo.execution
    store.identity_source = SimpleNamespace(current=lambda identity: actor())
    c.repo.current_identity = lambda identity: actor()
    store._uow = ApiUow(c.dsn, c.namespace)
    entry = registered_knowledge(c.service)
    store.delegated_read_hashes = frozenset({entry.spec.content_hash})
    profile = ExecutionProfile.model_validate({
        "route": {"provider": "controlled", "model": "controlled", "adapter_ref": "controlled@1", "configuration_hash": "f" * 64,
                  "capabilities": ["tool_calling"], "capabilities_verified": True, "context_window_tokens": 100000, "max_output_tokens": 4096},
        "budget": {"version": "test-budget@1", "max_input_tokens": 16000, "max_output_tokens": 4096,
                   "max_total_tokens": 32000, "execution_budget_ms": 120000},
    })
    with psycopg.connect(c.dsn) as conn:
        row = conn.execute(sql.SQL("SELECT r.manifest FROM {}.runtime_releases r").format(sql.Identifier(c.namespace))).fetchone()
        old = ReleaseManifest.model_validate(row[0])
        spec = old.spec.model_copy(update={"allowed_tools": (ToolBinding(name="retrieve_knowledge", version="1", max_operations_per_run=1),),
                                            "completion_validator_ref": "travel.evidence-completion@1"})
        profiles = old.profiles.as_object()
        profiles["execution"] = profile.model_dump(mode="json")
        profiles["resource_authority_ref"] = "knowledge-read@1"
        profiles["read_accounting"] = {entry.spec.content_hash: "child-model-ledger@1"}
        manifest = ReleaseManifest.model_validate({**old.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
                                                    "tool_schema_hashes": {"retrieve_knowledge@1": entry.spec.content_hash},
                                                    "model_profile": profile.route.model_dump(mode="json"), "profiles": profiles})
        conn.execute(sql.SQL("UPDATE {}.runtime_releases SET manifest=%s,manifest_hash=%s").format(sql.Identifier(c.namespace)),
                     (Jsonb(manifest.model_dump(mode="json")), manifest.content_hash()))
        run_request = RunRequest(request_id="run-request", message=c.query.query, expected_revision=0)
        conn.execute(sql.SQL("UPDATE {}.runtime_runs SET manifest_hash=%s,request=%s,request_hash=%s,principal=%s WHERE run_id=%s")
                     .format(sql.Identifier(c.namespace)),
                     (manifest.content_hash(), Jsonb(run_request.model_dump(mode="json")), run_request.business_hash(),
                      Jsonb(actor().model_dump(mode="json")), c.run_id))
        session_id = conn.execute(sql.SQL("SELECT session_id FROM {}.runtime_runs WHERE run_id=%s")
                                   .format(sql.Identifier(c.namespace)), (c.run_id,)).fetchone()[0]
    run = ClaimedRun(run_id=c.run_id, session_id=session_id, principal=actor(), request=run_request, manifest=manifest,
                     execution_deadline=datetime.now(UTC) + timedelta(minutes=2), history=JsonSnapshot.capture([]))
    validator = KnowledgeCompletionValidator(store, c.service)
    reads = ReadCoordinator(store, (entry,), task_guard=require_query)
    context = KnowledgeContext(reads, validator)
    model = AnswerModel()
    store.answer_authority = partial(DevelopmentAssembly.authorize_answer,
                                      SimpleNamespace(retrieval=SimpleNamespace(repository=c.repo)))
    store.resource_authority = c.repo.authorize_public_read
    store.resource_authority_ref = "knowledge-read@1"
    async with AsyncExitStack() as stack:
        checkpointer = InMemorySaver()
        if request.param == "postgres":
            connection = await psycopg.AsyncConnection.connect(c.dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row)
            await stack.enter_async_context(connection)
            checkpointer = AsyncPostgresSaver(connection)
            await checkpointer.setup()
        engine = AgentEngine(store, GovernedModelGateway(store, model, profile, context_authority=context.authorize), reads, None,
                              checkpointer, "worker-1", {manifest.spec.release_id: manifest},
                              {manifest.spec.release_id: "知识回答必须引用本轮证据，并返回约定JSON。"},
                              lambda run, candidate: validator.validate(run, "worker-1", candidate), require_query,
                              prepare_context=context.prepare, authorize_context=context.authorize)
        c.store, c.run, c.engine, c.model, c.reads, c.context = store, run, engine, model, reads, context
        yield c


async def test_real_agent_executor_pg_commit_api_and_sse(knowledge_case):
    c = knowledge_case
    candidate = await c.engine.execute(c.run)
    assert candidate.validation.verdict == "pass" and candidate.task.evidence_refs
    assert c.model.calls == 1 and c.embed.calls == c.rerank.calls == 1
    assert [m["role"] for m in c.model.envelopes[0]][-3:] == ["user", "assistant", "tool"]
    await c.store.complete("worker-1", c.run, candidate)
    await c.store.complete("worker-1", c.run, candidate)
    auth = DevelopmentAuthenticator({"k" * 40: actor()}, environment="development")
    app = create_app(c.store, auth.authenticate)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        headers = {"Authorization": "Bearer " + "k" * 40}
        response = await client.get(f"/api/v1/runs/{c.run.run_id}", headers=headers)
        assert response.status_code == 200 and response.json()["result"]["task_outcome"] == "achieved"
        stream = await client.get(f"/api/v1/runs/{c.run.run_id}/events", headers=headers)
        assert stream.status_code == 200 and "answer.delta" in stream.text and "run.completed" in stream.text
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT count(*) FROM {}.runtime_commits").format(sql.Identifier(c.namespace))).fetchone()[0] == 1


async def test_no_evidence_is_successful_run_but_not_achieved_task(knowledge_case):
    c = knowledge_case
    c.recall.hits = ()
    candidate = await c.engine.execute(c.run)
    assert candidate.answer_kind == "clarification" and candidate.task_outcome == "needs_input"
    assert candidate.validation.verdict == "needs_clarification" and c.rerank.calls == 0
    await c.store.complete("worker-1", c.run, candidate)
    assert (await c.store.get_run(actor(), c.run.run_id)).result["task_outcome"] == "needs_input"


async def test_revocation_between_validation_and_publish_blocks_answer(knowledge_case):
    c = knowledge_case
    candidate = await c.engine.execute(c.run)
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("UPDATE {}.documents SET revoked_at=clock_timestamp()").format(sql.Identifier(c.namespace)))
    with pytest.raises(RuntimeFault):
        await c.store.complete("worker-1", c.run, candidate)
    with psycopg.connect(c.dsn) as conn:
        assert conn.execute(sql.SQL("SELECT count(*) FROM {}.runtime_events WHERE type='answer.delta'")
                            .format(sql.Identifier(c.namespace))).fetchone()[0] == 0


async def test_terminal_sse_replay_rechecks_knowledge_grant(knowledge_case):
    c = knowledge_case
    candidate = await c.engine.execute(c.run)
    await c.store.complete("worker-1", c.run, candidate)
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("DELETE FROM {}.knowledge_grants").format(sql.Identifier(c.namespace)))
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await c.store.get_run(actor(), c.run.run_id)
    with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
        await c.store.events(actor(), c.run.run_id, 0)


async def test_judge_reference_material_rechecks_current_knowledge_grant(knowledge_case):
    c = knowledge_case
    candidate = await c.engine.execute(c.run)
    await c.store.complete("worker-1", c.run, candidate)
    async with c.store._transaction() as conn:
        run, session = await c.store._authorized_run(conn, actor(), c.run.run_id, lock=True)
        material = await c.repo.judge_reference_material(conn, actor(), run, session)
    assert material["version"] == "knowledge-judge-material@1"
    assert material["results"] and material["results"][0]["evidence"]
    with psycopg.connect(c.dsn) as conn:
        conn.execute(sql.SQL("DELETE FROM {}.knowledge_grants").format(sql.Identifier(c.namespace)))
    async with c.store._transaction() as conn:
        run, session = await c.store._authorized_run(conn, actor(), c.run.run_id, lock=True)
        with pytest.raises(RuntimeFault, match="ACCESS_DENIED"):
            await c.repo.judge_reference_material(conn, actor(), run, session)


async def test_successful_receipt_can_be_reauthorized_by_new_run_owner_without_dispatch(knowledge_case):
    c = knowledge_case
    await c.context.prepare(c.run, "worker-1")
    with psycopg.connect(c.dsn) as conn:
        # 仅模拟上游受控恢复已完成后的权威owner状态；不冒充旧进程终止证明演练。
        conn.execute(sql.SQL("UPDATE {}.runtime_runs SET owner_id='worker-2' WHERE run_id=%s")
                     .format(sql.Identifier(c.namespace)), (c.run.run_id,))
    await c.context.prepare(c.run, "worker-2")
    assert c.embed.calls == c.rerank.calls == 1


async def test_disabled_optional_authority_cannot_read_old_knowledge_answer_or_cancel_result(knowledge_case):
    c = knowledge_case
    candidate = await c.engine.execute(c.run)
    await c.store.complete("worker-1", c.run, candidate)
    revision = (await c.store.get_run(actor(), c.run.run_id)).revision
    c.store.resource_authority, c.store.resource_authority_ref = None, None
    with pytest.raises(RuntimeFault, match="RESOURCE_AUTHORITY_UNAVAILABLE"):
        await c.store.get_run(actor(), c.run.run_id)
    with pytest.raises(RuntimeFault, match="RESOURCE_AUTHORITY_UNAVAILABLE"):
        await c.store.events(actor(), c.run.run_id, 0)
    with pytest.raises(RuntimeFault, match="RESOURCE_AUTHORITY_UNAVAILABLE"):
        await c.store.create_run(actor(), c.run.session_id, c.run.request)
    view = await c.store.cancel_run(actor(), c.run.run_id, CancelRequest(request_id="cancel-read",
                                  expected_run_revision=revision))
    assert view.result is None
