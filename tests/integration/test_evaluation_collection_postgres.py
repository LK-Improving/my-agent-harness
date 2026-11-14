"""真实 PG 与 ASGI API/持久 SSE 采集定义；受控失败不充当模型端到端证明。"""

from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from harness.adapters.evaluation_files import read_freeze
from harness.adapters.evaluation_http import HTTPEvaluationAPI
from harness.bootstrap.api import create_app
from harness.evaluation.assets import cases
from harness.evaluation.collector import collect

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


async def test_candidate_without_active_release_collects_persisted_failure(prepared):
    store, actor, manifest, _ = prepared
    owner = "m5-test-" + uuid4().hex
    frozen, assets = read_freeze(ROOT / "evals/m5")
    records = []
    await store.register_worker(owner)

    class ControlledFailureAPI(HTTPEvaluationAPI):
        async def create_run(self, session_id, request):
            accepted = await super().create_run(session_id, request)
            claimed = await store.claim(owner)
            assert claimed is not None and claimed.run_id == accepted.run_id
            await store.started(owner, claimed, manifest.content_hash())
            await store.fail(owner, claimed.run_id, "SYNTHETIC_EVALUATION_FAILURE")
            await store.finish_slot(owner)
            return accepted

    try:
        app = create_app(store, lambda authorization: actor)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            report = await collect(ControlledFailureAPI(client), frozen,
                                   cases(assets["holdout.jsonl"]), manifest, "m5-gates@1",
                                   "batch-" + uuid4().hex, actor.resource_scopes, records.append)
        assert report["outcome"] == "incomplete" and report["collection_complete"]
        assert len(report["cases"]) == 4
        for entry in report["cases"]:
            assert entry["accepted"]["manifest_hash"] == manifest.content_hash()
            assert entry["started"]["payload"]["observed_manifest_hash"] == manifest.content_hash()
            assert entry["terminal"]["type"] == "run.failed"
            assert entry["final"]["status"] == "failed"
            stored = await store.events(actor, entry["accepted"]["run_id"], 0)
            assert [row["seq"] for row in stored] == [row["seq"] for row in entry["events"]]
        assert sum(record["phase"] == "case_intent" for record in records) == 4
    finally:
        await store.release_worker(owner)


async def test_ordinary_identity_cannot_create_collection_context(prepared):
    store, actor, manifest, _ = prepared
    ordinary = actor.model_copy(update={"roles": frozenset({"user"})})
    frozen, assets = read_freeze(ROOT / "evals/m5")
    records = []
    app = create_app(store, lambda authorization: ordinary)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        report = await collect(HTTPEvaluationAPI(client), frozen, cases(assets["development.jsonl"]),
                               manifest, "m5-gates@1", "denied-" + uuid4().hex,
                               frozenset(), records.append)
    assert not report["collection_complete"] and report["cases"] == []
    assert report["outcome"] == "not_run"
    assert report["collection_error_code"] == "API_REJECTED_403"
    assert not any(record["phase"] == "case_intent" for record in records)
