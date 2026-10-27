"""知识工具/引用/装配边界用例定义；只用合成端口，不等于实际API/SSE验收。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from domain_tools.knowledge.completion import reference, validate_knowledge
from domain_tools.knowledge.context import KnowledgeContext
from domain_tools.knowledge.retrieval import KnowledgeOutput, registered_knowledge, require_query
from harness.contracts.execution import ReadReservation
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.retrieval import RetrievalResult
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec, ToolSpec
from harness.contracts.state import TaskState, ToolReceipt, ValidationReport
from harness.tools.reads import ReadCoordinator
from tests.retrieval_pipeline_fixtures import retrieval_ticket

ROOT = Path(__file__).resolve().parents[2]


def values():
    ticket, _, evidence = retrieval_ticket()
    spec = AgentSpec.model_validate_json((ROOT / "agents/travel.json").read_text("utf-8"))
    spec = spec.model_copy(update={"knowledge_bindings": (spec.knowledge_bindings[0].model_copy(update={
        "knowledge_base_id": ticket.pin.knowledge_base_id, "snapshot_id": ticket.pin.snapshot_id}),)})
    manifest = ReleaseManifest(spec=spec, source_revision=None, source_tree_hash="1" * 64,
                                dependency_lock_hash="2" * 64, prompt_hash="3" * 64,
                                tool_schema_hashes={f"{b.name}@{b.version}": "4" * 64 for b in spec.allowed_tools}, model_profile={}, profiles={},
                                environment_profile_hash="5" * 64)
    run = ClaimedRun(run_id="run-1", session_id="session-1", principal=ticket.actor,
                     request=RunRequest(request_id="r1", message=ticket.query.query, expected_revision=0), manifest=manifest,
                     execution_deadline=datetime.now(UTC) + timedelta(minutes=2), history=JsonSnapshot.capture([]))
    result = RetrievalResult(retrieval_id=ticket.retrieval_id, outcome="evidence", evidence=evidence)
    output = KnowledgeOutput(results=(result,))
    receipt = ToolReceipt(operation_id="read-1", attempt_id="attempt-1", status="succeeded",
                            args_hash=canonical_hash({"query": run.request.message}), data=output.model_dump(mode="json"),
                            cost_status="unknown")
    ref = reference(evidence[0])
    candidate = AnswerCandidate(draft_id="draft-1", model_call_id="call-1", model_attempt_id="model-attempt-1",
                                  answer=f"资料说明[[source:{ref}]]", answer_kind="answer", task_outcome="achieved",
                                  task=TaskState(objective=run.request.message, source_turn_id=run.run_id, evidence_refs=(ref,)),
                                  validation=ValidationReport(validator_ref=spec.completion_validator_ref, input_hash="0" * 64,
                                                                verdict="fail", checks={}, checked_at=datetime.now(UTC)))
    return run, output, receipt, candidate


def test_exact_citations_and_candidate_hash_are_required():
    run, output, receipt, candidate = values()
    report = validate_knowledge(run, candidate, (output,), (receipt,))
    assert report.verdict == "pass" and report.input_hash == canonical_hash(candidate.validation_input())
    forged = candidate.model_copy(update={"answer": "没有引用的回答"})
    assert validate_knowledge(run, forged, (output,), (receipt,)).verdict == "fail"
    forged = candidate.model_copy(update={"task": candidate.task.model_copy(update={"evidence_refs": ("other/version/chunk",)})})
    assert validate_knowledge(run, forged, (output,), (receipt,)).verdict == "fail"
    assert validate_knowledge(run, candidate, (), ()).verdict == "fail"


def test_no_evidence_only_allows_clarification_or_abstention():
    run, _, receipt, candidate = values()
    output = KnowledgeOutput(results=(RetrievalResult(retrieval_id="empty-1", outcome="no_evidence", evidence=()),))
    assert validate_knowledge(run, candidate, (output,), (receipt,)).verdict == "fail"
    candidate = candidate.model_copy(update={"answer": "暂无证据，请补充地点。", "answer_kind": "clarification",
                                             "task_outcome": "needs_input", "task": candidate.task.model_copy(update={"evidence_refs": ()})})
    assert validate_knowledge(run, candidate, (output,), (receipt,)).verdict == "needs_clarification"


def test_query_cannot_change_after_extension_or_include_scope():
    run, _, _, _ = values()
    entry = registered_knowledge(None)
    spec = ToolSpec.model_validate(entry.spec.as_object())
    require_query(run, spec, {"query": run.request.message})
    with pytest.raises(RuntimeFault, match="QUERY_CHANGED"):
        require_query(run, spec, {"query": "其他问题"})
    with pytest.raises(ValueError):
        entry.input_type.model_validate({"query": run.request.message, "knowledge_base_id": "other"})


class Ledger:
    def __init__(self):
        self.receipt, self.calls = None, 0

    async def reserve_read(self, run, owner, logical, spec, arguments):
        self.calls += 1
        self.arguments = arguments
        return ReadReservation(operation_id="read-1", attempt_id="attempt-1", receipt=self.receipt)

    async def save_read(self, reservation, result, error):
        self.receipt = ToolReceipt(operation_id=reservation.operation_id, attempt_id=reservation.attempt_id,
                                   args_hash=self.arguments.content_hash, status="failed" if error else "succeeded",
                                   data=result.materialize() if result else None, cost_status="unknown")
        return self.receipt


class Service:
    def __init__(self, output):
        self.output, self.calls, self.revoked = output, 0, False

    async def retrieve(self, actor, owner, run_id, kb_id, query):
        self.calls += 1
        return self.output.results[0]

    async def authorize_result(self, *args):
        if self.revoked:
            raise RuntimeFault("KNOWLEDGE_REVOKED", 403)
        return args[-1]


async def test_context_uses_executor_once_complete_tool_group_and_rechecks_cache():
    run, output, _, _ = values()
    service, ledger = Service(output), Ledger()
    entry = registered_knowledge(service)
    reads = ReadCoordinator(ledger, (entry,), task_guard=require_query)
    context = KnowledgeContext(reads, None)
    messages = await context.prepare(run, "worker-1")
    assert [m.type for m in messages] == ["human", "ai", "tool"]
    assert messages[1].tool_calls[0]["id"] == messages[2].tool_call_id
    await context.prepare(run, "worker-2")
    assert service.calls == 1 and ledger.receipt.cost_status == "unknown"
    service.revoked = True
    with pytest.raises(RuntimeFault, match="REVOKED"):
        await context.prepare(run, "worker-2")


def test_unmanaged_billable_read_registration_is_rejected():
    from dataclasses import replace

    _, output, _, _ = values()
    entry = registered_knowledge(Service(output))
    with pytest.raises(ValueError):
        ReadCoordinator(Ledger(), (replace(entry, accounting_ref=None),))


@pytest.mark.parametrize("selected", ["operations", "travel"])
def test_actual_assembly_selects_optional_knowledge_without_loading_other_capabilities(tmp_path, monkeypatch, selected):
    import shutil
    import tomllib
    from types import SimpleNamespace

    from harness.bootstrap.assembly import DevelopmentAssembly
    from harness.bootstrap.retrieval import RetrievalRuntime
    from harness.bootstrap.server import ServerSettings
    from harness.contracts.execution import ExecutionProfile

    # 只在临时目录构建配置替身；不读取宿主机凭据、不创建服务连接。
    for folder in ("agents", "prompts"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    shutil.copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    locked = {p["name"]: p.get("version") for p in tomllib.loads((tmp_path / "uv.lock").read_text("utf-8"))["package"]}
    profile = ExecutionProfile.model_validate_json((ROOT / "examples/execution-profile.example.json").read_text("utf-8"))
    settings = ServerSettings(agent_profile=selected, rag_enabled=selected == "travel",
                              database_dsn="postgresql://unused", identity_file=tmp_path / "identity.json")
    monkeypatch.setattr("harness.bootstrap.assembly.load_server_settings", lambda root: settings)
    monkeypatch.setattr("harness.bootstrap.assembly.source_hash", lambda root: "1" * 64)
    monkeypatch.setattr("harness.bootstrap.assembly.version", lambda name: locked[name])
    monkeypatch.setattr("harness.bootstrap.assembly.load_model_runtime", lambda *args, **kwargs: SimpleNamespace(
        profile=profile, transport=object(), binding_snapshot=None, ready=lambda: True))
    _, configuration, _ = retrieval_ticket()
    configuration = configuration.model_copy(update={"snapshots": (configuration.snapshots[0].model_copy(
        update={"knowledge_base_id": "example-kb"}),)})
    calls = []

    def optional(*args, **kwargs):
        calls.append(kwargs)
        return RetrievalRuntime(Service(None), configuration,
                                SimpleNamespace(authorize_public_read=None), lambda: None)

    monkeypatch.setattr("harness.bootstrap.retrieval.load_retrieval", optional)
    assembly = DevelopmentAssembly(tmp_path, owner="fixture-worker", checkpointer=None)
    assert assembly.manifest.profiles.as_object()["context_boundary"] == {
        "version": "session-memory-boundary@1", "memory_profile_ref": "session-only@1",
        "automatic_long_term_profile": False, "semantic_history_recall": False, "complex_fact_extraction": False}
    if selected == "operations":
        assert not calls and assembly.retrieval is None
        assert assembly.engine.prepare_context is None and assembly.store.answer_authority is None
        assert "retrieval" not in assembly.manifest.profiles.as_object()
    else:
        assert len(calls) == 1 and calls[0]["execution"] is assembly.store
        assert assembly.engine.prepare_context is not None and assembly.store.answer_authority is not None
        assert tuple(assembly.reads.entries) == ("retrieve_knowledge", "read_artifact")
        assert assembly.manifest.spec.knowledge_bindings[0].snapshot_id == configuration.snapshots[0].snapshot_id
        assert assembly.manifest.profiles.as_object()["read_accounting"]
        assert assembly.manifest.spec.release_id.startswith("knowledge-development-")
        protocol = tmp_path / "prompts/knowledge/protocol.txt"
        protocol.write_text(protocol.read_text("utf-8") + "\n部署后变化", "utf-8")
        with pytest.raises(RuntimeFault, match="DRIFT"):
            assembly.assert_compatible(assembly.manifest)


def test_knowledge_profile_requires_explicit_rag_enablement(tmp_path, monkeypatch):
    from harness.bootstrap.assembly import DevelopmentAssembly
    from harness.bootstrap.server import ServerSettings

    monkeypatch.setattr("harness.bootstrap.assembly.load_server_settings", lambda root: ServerSettings(
        agent_profile="travel", rag_enabled=False, database_dsn="postgresql://unused", identity_file=tmp_path / "identity.json"))
    with pytest.raises(ValueError, match="显式启用RAG"):
        DevelopmentAssembly(tmp_path, owner="fixture-worker", checkpointer=None)


async def test_sse_rechecks_permission_between_events_in_same_loaded_batch():
    from types import SimpleNamespace

    from httpx import ASGITransport, AsyncClient

    from harness.bootstrap.api import create_app

    run, _, _, _ = values()

    class Store:
        reads, released = 0, False

        async def admit_read(self, actor): pass
        async def acquire_sse(self, actor): return "lease"
        async def renew_sse(self, actor, lease): pass
        async def release_sse(self, lease): self.released = True

        async def events(self, actor, run_id, cursor):
            return [{"seq": 1, "type": "answer.delta", "delta": "first"},
                    {"seq": 2, "type": "answer.delta", "delta": "revoked-body"},
                    {"seq": 3, "type": "run.completed"}]

        async def get_run(self, actor, run_id):
            self.reads += 1
            if self.reads >= 3:
                raise RuntimeFault("KNOWLEDGE_REVOKED", 403)
            return SimpleNamespace(status="succeeded")

    store = Store()
    async with AsyncClient(transport=ASGITransport(app=create_app(store, lambda value: run.principal)),
                           base_url="http://test") as client:
        response = await client.get("/api/v1/runs/run-1/events")
    assert "first" in response.text and "revoked-body" not in response.text and store.released
