"""采集中止、必需缺口与取消传播定义；本轮未执行。"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.adapters.evaluation_files import read_freeze
from harness.contracts.evaluation import CollectionFault
from harness.contracts.events import RuntimeEvent
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import RunView
from harness.contracts.specs import AgentSpec
from harness.evaluation.assets import cases
from harness.evaluation.collector import collect

ROOT = Path(__file__).resolve().parents[2]


def candidate():
    spec = AgentSpec.model_validate_json((ROOT / "agents/operations.json").read_bytes())
    return ReleaseManifest(spec=spec, source_revision=None, source_tree_hash="1" * 64,
                           dependency_lock_hash="2" * 64, prompt_hash="3" * 64,
                           tool_schema_hashes={f"{tool.name}@{tool.version}": "4" * 64
                                               for tool in spec.allowed_tools},
                           model_profile={}, profiles={}, environment_profile_hash="5" * 64)


class API:
    def __init__(self, manifest, failure=None):
        self.manifest, self.failure = manifest, failure
        self.sessions = 0

    async def create_context(self, request):
        return EvaluationContext(context_id="context", tenant_id="test", subject_id="test",
                                 agent_id=self.manifest.spec.agent_id,
                                 manifest_hash=self.manifest.content_hash(),
                                 environment_profile_hash=self.manifest.environment_profile_hash,
                                 **request.model_dump())

    async def create_session(self, request):
        self.sessions += 1
        return f"session-{self.sessions}"

    async def create_run(self, session_id, request):
        if self.failure:
            raise self.failure
        return self.view(session_id)

    def view(self, session_id):
        return RunView(run_id=f"run-{self.sessions}", session_id=session_id, status="failed",
                       revision=1, release_id=self.manifest.spec.release_id,
                       manifest_hash=self.manifest.content_hash(),
                       environment_profile_hash=self.manifest.environment_profile_hash,
                       cancel_requested=False, recovery_required=False, created_at=datetime.now(UTC))

    async def events(self, run_id):
        yield RuntimeEvent(run_id=run_id, seq=1, type="run.failed", timestamp=datetime.now(UTC),
                           payload={"code": "SYNTHETIC"})

    async def get_run(self, run_id):
        return self.view(f"session-{self.sessions}")


async def test_unknown_creation_preserves_intent_and_stops_remaining_cases():
    frozen, blobs = read_freeze(ROOT / "evals/m5")
    manifest = candidate()
    api = API(manifest, CollectionFault("CREATE_OUTCOME_UNKNOWN"))
    journal = []
    report = await collect(api, frozen, cases(blobs["holdout.jsonl"]), manifest,
                           "m5-gates@1", "test-batch", frozenset(), journal.append)
    assert api.sessions == 1
    assert report["outcome"] == "incomplete" and not report["collection_complete"]
    assert report["collection_error_code"] == "CREATE_OUTCOME_UNKNOWN"
    assert len(report["cases"]) == 1 and len(report["expected_case_ids"]) == 4
    assert [record["phase"] for record in journal][:4] == [
        "batch_intent", "context_created", "case_intent", "session_created"]


async def test_complete_collection_does_not_become_evaluation_pass():
    frozen, blobs = read_freeze(ROOT / "evals/m5")
    manifest = candidate()
    report = await collect(API(manifest), frozen, cases(blobs["holdout.jsonl"]), manifest,
                           "m5-gates@1", "test-batch", frozenset(), lambda record: None)
    assert report["collection_complete"] and report["outcome"] == "incomplete"
    assert "required_judge" in report["unverified"] and report["baseline"] is None
    assert len(report["cases"]) == 4


async def test_cancellation_is_not_converted_to_success_or_swallowed():
    frozen, blobs = read_freeze(ROOT / "evals/m5")
    manifest = candidate()
    with pytest.raises(asyncio.CancelledError):
        await collect(API(manifest, asyncio.CancelledError()), frozen, cases(blobs["holdout.jsonl"]),
                      manifest, "m5-gates@1", "test-batch", frozenset(), lambda record: None)
