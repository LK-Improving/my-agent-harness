"""正式 M1 存储集成用例；仅显式独立 *_m1_test 库允许运行。"""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from harness.adapters.runtime_postgres import PostgresRuntimeStore
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import (
    AdmissionProfile,
    AnswerCandidate,
    CancelRequest,
    EvaluationRequest,
    Principal,
    RuntimeFault,
    SessionRequest,
)
from harness.contracts.specs import AgentSpec
from harness.contracts.state import TaskState, ValidationReport
from psycopg import AsyncConnection

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def prepared():
    dsn = os.environ.get("HARNESS_M1_TEST_DSN")
    if not dsn:
        pytest.fail("需显式设置独立 HARNESS_M1_TEST_DSN；不回退 M0 或业务库")
    parsed = urlparse(dsn)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.path.removeprefix(
        "/"
    ).endswith("_m1_test"):
        pytest.fail("M1 集成测试仅允许本机以 _m1_test 结尾的数据库")
    async with await AsyncConnection.connect(dsn) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(184031,1)")
            exists = await (
                await conn.execute("SELECT to_regclass('harness.schema_versions')")
            ).fetchone()
            if exists == (None,):
                await conn.execute(
                    (ROOT / "migrations/0001_runtime.sql").read_text("utf-8"), prepare=False
                )
            installed = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=2")
            ).fetchone()
            if installed is None:
                await conn.execute(
                    (ROOT / "migrations/0002_writes.sql").read_text("utf-8"), prepare=False
                )
            execution_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=3")
            ).fetchone()
            if execution_version is None:
                await conn.execute(
                    (ROOT / "migrations/0003_execution.sql").read_text("utf-8"), prepare=False
                )
            extension_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=4")
            ).fetchone()
            if extension_version is None:
                await conn.execute(
                    (ROOT / "migrations/0004_extensions.sql").read_text("utf-8"), prepare=False
                )
            artifact_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=5")
            ).fetchone()
            if artifact_version is None:
                await conn.execute(
                    (ROOT / "migrations/0005_artifacts.sql").read_text("utf-8"), prepare=False
                )
            retention_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=6")
            ).fetchone()
            if retention_version is None:
                await conn.execute(
                    (ROOT / "migrations/0006_artifact_retention.sql").read_text("utf-8"), prepare=False
                )
            context_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=7")
            ).fetchone()
            if context_version is None:
                await conn.execute(
                    (ROOT / "migrations/0007_context_purpose.sql").read_text("utf-8"), prepare=False
                )
            summary_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=8")
            ).fetchone()
            if summary_version is None:
                await conn.execute(
                    (ROOT / "migrations/0008_context_summaries.sql").read_text("utf-8"), prepare=False
                )
            retention_version = await (
                await conn.execute("SELECT version FROM harness.schema_versions WHERE version=9")
            ).fetchone()
            if retention_version is None:
                await conn.execute((ROOT / "migrations/0009_retention.sql").read_text("utf-8"), prepare=False)
    store = PostgresRuntimeStore(dsn, AdmissionProfile())
    actor = Principal(
        tenant_id="test-" + uuid4().hex,
        subject_id="developer",
        application_id="app",
        roles=frozenset({"release_admin", "evaluation_admin", "evaluator", "user"}),
        resource_scopes=frozenset({"synthetic"}),
    )
    spec = AgentSpec.model_validate_json((ROOT / "agents/operations.json").read_text("utf-8"))
    manifest = ReleaseManifest(
        spec=spec,
        source_revision=None,
        source_tree_hash="1" * 64,
        dependency_lock_hash="2" * 64,
        prompt_hash="3" * 64,
        tool_schema_hashes={f"{tool.name}@{tool.version}": "4" * 64 for tool in spec.allowed_tools},
        model_profile={},
        profiles={},
        environment_profile_hash="5" * 64,
    )
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(
        actor,
        EvaluationRequest(
            candidate_release_id=spec.release_id,
            dataset_hash="6" * 64,
            gate_policy_version="test@1",
            resource_scopes=actor.resource_scopes,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
    )
    session_id = await store.create_session(
        actor, SessionRequest(agent_id=spec.agent_id, evaluation_context_id=context.context_id)
    )
    yield store, actor, manifest, session_id
    # 不自动删除证据表或其他测试记录；独立库由操作者管理。


async def test_parallel_idempotency_and_conflicting_payload(prepared):
    store, actor, _, session_id = prepared
    request = RunRequest(request_id="same-key", message="查询合成工单", expected_revision=0)
    results = await asyncio.gather(
        *(store.create_run(actor, session_id, request) for _ in range(8))
    )
    assert len({row.run_id for row in results}) == 1
    with pytest.raises(RuntimeFault, match="REQUEST_ID_CONFLICT"):
        await store.create_run(
            actor, session_id, request.model_copy(update={"message": "另一输入"})
        )
    other = actor.model_copy(update={"tenant_id": "other-tenant"})
    with pytest.raises(RuntimeFault) as error:
        await store.get_run(other, results[0].run_id)
    assert error.value.status == 404
    await store.cancel_run(
        actor, results[0].run_id, CancelRequest(request_id="cleanup", expected_run_revision=0)
    )


async def test_unique_commit_old_revision_replay_and_persistent_events(prepared):
    store, actor, manifest, session_id = prepared
    request = RunRequest(request_id="commit-key", message="查询合成工单", expected_revision=0)
    created = await store.create_run(actor, session_id, request)
    identity = uuid4().hex
    await store.register_worker(identity)
    try:
        with pytest.raises(RuntimeFault, match="WORKER_ALREADY_REGISTERED"):
            await store.register_worker(uuid4().hex)
        run = await store.claim(identity)
        assert run is not None and run.run_id == created.run_id
        await store.started(identity, run, manifest.content_hash())
        task = TaskState(objective=request.message, source_turn_id=run.run_id)
        report = ValidationReport(
            validator_ref=manifest.spec.completion_validator_ref,
            input_hash="0" * 64,
            verdict="needs_clarification",
            checks={"candidate_valid": True},
            checked_at=datetime.now(UTC),
        )
        candidate = AnswerCandidate(
            draft_id="draft",
            model_call_id="call",
            model_attempt_id="attempt",
            answer="请提供工单编号。",
            answer_kind="clarification",
            task_outcome="needs_input",
            task=task,
            validation=report,
        )
        candidate = candidate.model_copy(
            update={
                "validation": report.model_copy(
                    update={"input_hash": canonical_hash(candidate.validation_input())}
                )
            }
        )
        await store.complete(identity, run, candidate)
        await store.complete(identity, run, candidate)
        replay = await store.create_run(actor, session_id, request)
        assert replay.run_id == run.run_id and replay.status == "succeeded"
        events = await store.events(actor, run.run_id, 0)
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        assert sum(event["type"] == "run.completed" for event in events) == 1
        await store.finish_slot(identity)
    finally:
        await store.fail(identity, created.run_id, "TEST_ABORTED")
        await store.finish_slot(identity)
        await store.release_worker(identity)
