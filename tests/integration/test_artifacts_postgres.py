"""真实PG产物不可变、并发容量、同源分页与故障回滚定义；本批未运行。"""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from psycopg.errors import RaiseException

from harness.adapters.artifacts_postgres import PostgresArtifacts
from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.contracts.artifacts import ArtifactProfile, ArtifactRead
from harness.contracts.execution import ExecutionProfile
from harness.contracts.hashing import RunRequest
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AdmissionProfile, EvaluationRequest, RuntimeFault, SessionRequest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ToolBinding, ToolSpec
from harness.tools.artifacts import registered_artifact_reader
from harness.tools.reads import ReadCoordinator, RegisteredRead

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


class Empty(Contract):
    pass


class LargeResult(Contract):
    body: str


@pytest_asyncio.fixture
async def artifact_case(prepared, request):
    _, actor, original, _ = prepared
    store = PostgresExecutionStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    profile = ArtifactProfile(projection_chars=64, max_artifact_bytes=40000, max_run_bytes=40000,
                              ttl_seconds=1 if getattr(request, "param", None) == "expire" else 604800)
    artifacts = PostgresArtifacts(store, profile)
    calls = []

    async def source(actor, args):
        calls.append(args)
        return {"body": "开头" + "😀中文" * 700 + "中段唯一凭据" + "😀中文" * 800 + "尾巴"}

    tool = ToolSpec(name="large_result", version="1", handler_ref="test.large-result@1", effect="read",
                    input_schema=Empty.model_json_schema(), output_schema=LargeResult.model_json_schema(),
                    authorization_policy_ref="test-session@1", timeout_ms=10000,
                    cost={"billable": False, "upper_bound": "0"})
    entry = RegisteredRead(spec=JsonSnapshot.capture(tool.model_dump(mode="json")), input_type=Empty,
                           output_type=LargeResult, handler=source, authorize=lambda actor, args: True)
    reader = registered_artifact_reader(artifacts)
    entries = (entry, reader)
    spec = original.spec.model_copy(update={"release_id": "artifact-test", "allowed_tools": (
        ToolBinding(name="large_result", version="1", max_operations_per_run=8),
        ToolBinding(name="read_artifact", version="1", max_operations_per_run=4))})
    execution = ExecutionProfile.model_validate_json((ROOT / "examples/execution-profile.example.json").read_text("utf-8"))
    # 控制场景显式给出窗口，不把示例中的未确认能力当作供应商实测。
    execution = execution.model_copy(update={"route": execution.route.model_copy(update={"context_window_tokens": 100000})})
    manifest = ReleaseManifest.model_validate({**original.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
        "tool_schema_hashes": {f"{ToolSpec.model_validate(item.spec.as_object()).name}@1": item.spec.content_hash for item in entries},
        "model_profile": execution.route.model_dump(mode="json"),
        "profiles": {"artifacts": profile.model_dump(mode="json"), "execution": execution.model_dump(mode="json")}})
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(actor, EvaluationRequest(candidate_release_id=spec.release_id,
        dataset_hash="e" * 64, gate_policy_version="artifact-test@1", resource_scopes=actor.resource_scopes,
        expires_at=datetime.now(UTC) + timedelta(hours=1)))
    session = await store.create_session(actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id))
    await store.create_run(actor, session, RunRequest(request_id="artifact-run", message="读取中间内容", expected_revision=0))
    owner = uuid4().hex
    await store.register_worker(owner)
    run = await store.claim(owner)
    assert run is not None and run.session_id == session
    reads = ReadCoordinator(store, entries)
    try:
        yield store, artifacts, reads, actor, owner, run, calls, execution
    finally:
        await store.fail(owner, run.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)


async def test_durable_projection_paging_replay_and_counts(artifact_case):
    store, artifacts, reads, actor, owner, run, calls, _ = artifact_case
    receipt = await reads.execute(run, owner, "large_result", "source", {})
    first, duplicate = await asyncio.gather(artifacts.project(run, owner, receipt), artifacts.project(run, owner, receipt))
    assert first == duplicate and len(calls) == 1
    info = first.as_object()["artifact"]
    saved = [event for event in await store.events(actor, run.run_id, 0) if event["type"] == "artifact.saved"]
    assert len(saved) == 1 and saved[0]["payload"]["artifact_ref"] == info["artifact_ref"]
    offset, pages = 0, []
    while True:
        page = await artifacts.read(actor, run.session_id, ArtifactRead(artifact_ref=info["artifact_ref"], offset=offset, limit=4000))
        assert page.content_hash == info["content_hash"] and page.source_operation_id == receipt.operation_id
        pages.append(page.text)
        if page.next_offset is None:
            break
        assert page.next_offset == offset + len(page.text)
        offset = page.next_offset
    assert JsonSnapshot.capture(json.loads("".join(pages))) == JsonSnapshot.capture(receipt.data)
    args = {"artifact_ref": info["artifact_ref"], "offset": 2000, "limit": 100}
    read = await reads.execute(run, owner, "read_artifact", "page", args)
    assert await reads.execute(run, owner, "read_artifact", "page", args) == read
    assert (await artifacts.project(run, owner, read)).materialize() == read.data
    assert len(calls) == 1
    async with store._transaction() as conn:
        count = await store._one(conn, "SELECT count(*) AS n FROM harness.artifacts WHERE run_id=%s", (run.run_id,))
        assert count["n"] == 1
        row = await store._one(conn, "SELECT tool_dispatches FROM harness.runs WHERE run_id=%s", (run.run_id,))
        assert row["tool_dispatches"] == 2
    # 原始回执没有变成投影；验证器仍可读取完整源输出。
    assert any(item.operation_id == receipt.operation_id and item.data == receipt.data
               for item in await store.read_receipts(actor, run.run_id))


async def test_immutable_body_scope_and_revoke(artifact_case):
    store, artifacts, reads, actor, owner, run, _, _ = artifact_case
    receipt = await reads.execute(run, owner, "large_result", "source", {})
    info = (await artifacts.project(run, owner, receipt)).as_object()["artifact"]
    request = ArtifactRead(artifact_ref=info["artifact_ref"])
    for change in ({"tenant_id": "other"}, {"application_id": "other"}, {"subject_id": "other"}, {"resource_scopes": frozenset()}):
        with pytest.raises(RuntimeFault):
            await artifacts.read(actor.model_copy(update=change), run.session_id, request)
    with pytest.raises(RuntimeFault):
        await artifacts.read(actor, "other-session", request)
    with pytest.raises(RaiseException):
        async with store._transaction() as conn:
            await conn.execute("UPDATE harness.artifacts SET body='changed' WHERE artifact_ref=%s", (request.artifact_ref,))
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.artifacts SET revoked_at=clock_timestamp() WHERE artifact_ref=%s", (request.artifact_ref,))
    with pytest.raises(RuntimeFault, match="ARTIFACT_REVOKED"):
        await artifacts.read(actor, run.session_id, request)
    with pytest.raises(RuntimeFault, match="ARTIFACT_REVOKED"):
        await artifacts.authorize_context(run, owner)


async def test_capacity_serializes_concurrent_sources(artifact_case):
    store, artifacts, reads, _, owner, run, _, _ = artifact_case
    receipts = [await reads.execute(run, owner, "large_result", f"source-{i}", {}) for i in range(3)]
    results = await asyncio.gather(*(artifacts.project(run, owner, receipt) for receipt in receipts), return_exceptions=True)
    assert sum(isinstance(result, JsonSnapshot) for result in results) == 2
    assert sum(isinstance(result, RuntimeFault) and result.code == "ARTIFACT_RUN_CAPACITY" for result in results) == 1
    async with store._transaction() as conn:
        row = await store._one(conn, "SELECT sum(total_bytes) AS used,count(*) AS n FROM harness.artifacts WHERE run_id=%s", (run.run_id,))
        assert row["used"] <= artifacts.profile.max_run_bytes and row["n"] == 2


@pytest.mark.parametrize("failure", ["audit", "event"])
async def test_audit_failure_rolls_back_artifact_without_fake_ref(artifact_case, monkeypatch, failure):
    store, artifacts, reads, _, owner, run, calls, _ = artifact_case
    receipt = await reads.execute(run, owner, "large_result", "source", {})
    target = "_audit" if failure == "audit" else "_event"
    original = getattr(store, target)

    async def broken(conn, actor_or_run, action, resource):
        if action == "artifact.saved":
            raise RuntimeFault("AUDIT_UNAVAILABLE", 503)
        await original(conn, actor_or_run, action, resource)

    monkeypatch.setattr(store, target, broken)
    with pytest.raises(RuntimeFault, match="AUDIT_UNAVAILABLE"):
        await artifacts.project(run, owner, receipt)
    async with store._transaction() as conn:
        assert (await store._one(conn, "SELECT count(*) AS n FROM harness.artifacts WHERE run_id=%s", (run.run_id,)))["n"] == 0
    monkeypatch.setattr(store, target, original)
    assert (await artifacts.project(run, owner, receipt)).as_object()["artifact"]["artifact_ref"]
    assert len(calls) == 1


async def test_page_dispatch_and_full_context_budget_are_not_free(artifact_case):
    store, artifacts, reads, _, owner, run, calls, profile = artifact_case
    receipt = await reads.execute(run, owner, "large_result", "source", {})
    info = (await artifacts.project(run, owner, receipt)).as_object()["artifact"]
    pages = []
    for i in range(4):
        receipt = await reads.execute(run, owner, "read_artifact", f"page-{i}",
                                      {"artifact_ref": info["artifact_ref"], "offset": 1000 * i, "limit": 4000})
        pages.append(receipt.data)
    with pytest.raises(RuntimeFault, match="READ_TOOL_LIMIT"):
        await reads.execute(run, owner, "read_artifact", "page-5", {"artifact_ref": info["artifact_ref"]})
    # 将完整页放入完整工具事务组；输入硬上限优先于累计额度拒绝。
    messages = [{"role": "user", "content": "读取"}]
    for i, page in enumerate(pages):
        messages.extend([{"role": "assistant", "tool_calls": [{"id": f"page-{i}", "type": "function",
            "function": {"name": "read_artifact", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"page-{i}", "content": JsonSnapshot.capture(page).canonical_text}])
    async with store._transaction() as conn:
        await conn.execute("UPDATE harness.runs SET model_reserved_tokens=%s WHERE run_id=%s",
                           (profile.budget.max_total_tokens - 1, run.run_id))
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED"):
        await store.reserve_model(run, owner, JsonSnapshot.capture(messages), JsonSnapshot.capture([]), profile)
    with pytest.raises(RuntimeFault, match="MODEL_BUDGET_EXHAUSTED"):
        await store.reserve_model(run, owner, JsonSnapshot.capture([{"role": "user", "content": "继续"}]),
                                  JsonSnapshot.capture([]), profile)
    assert len(calls) == 1


async def test_official_graph_projects_then_reads_omitted_middle(artifact_case):
    """受控网关只测官方图与真实PG端口连接，不冒充真实模型/API-SSE验收。"""
    from langchain_core.messages import HumanMessage

    from harness.contracts.models import ModelResult, ModelUsage
    from harness.graphs.agent import AgentEngine

    store, artifacts, reads, _, owner, run, calls, _ = artifact_case

    class ControlledGateway:
        count = 0

        async def complete(self, current, worker, messages, tools):
            self.count += 1
            content, tool_calls = "", []
            if self.count == 1:
                tool_calls = [{"id": "source", "type": "function", "function": {"name": "large_result", "arguments": "{}"}}]
            elif self.count == 2:
                data = json.loads(messages.as_object_array()[-1]["content"])["data"]
                assert data["truncated"] and "中段唯一凭据" not in data["artifact"]["text"]
                tool_calls = [{"id": "page", "type": "function", "function": {"name": "read_artifact",
                    "arguments": json.dumps({"artifact_ref": data["artifact"]["artifact_ref"], "offset": 2000, "limit": 2000})}}]
            else:
                assert self.count == 3
                data = json.loads(messages.as_object_array()[-1]["content"])["data"]
                assert "中段唯一凭据" in data["text"] and "artifact" not in data
                content = "已读取中段唯一凭据"
            return ModelResult(model="controlled", content=content, tool_calls=JsonSnapshot.capture(tool_calls),
                finish_reason="stop", request_id="test", usage=ModelUsage(input_tokens=10, output_tokens=10),
                model_call_id=f"model-{self.count}", model_attempt_id=f"attempt-{self.count}", request_fingerprint="a" * 64)

    async def unused_validator(current, candidate):
        raise AssertionError("这里只调用图连接测试，不提交成功turn")

    gateway = ControlledGateway()
    engine = AgentEngine(store, gateway, reads, None, None, owner,
        {run.manifest.spec.release_id: run.manifest}, {run.manifest.spec.release_id: "受控图连接用例"},
        unused_validator, artifact_projector=artifacts, authorize_context=artifacts.authorize_context)
    output = await engine.graph(run).ainvoke({"messages": [HumanMessage(content="读取中段")],
        "harness_run_id": run.run_id, "harness_manifest_hash": run.manifest.content_hash()}, engine.config(run))
    assert output["messages"][-1].content == "已读取中段唯一凭据"
    assert len(calls) == 1 and gateway.count == 3
