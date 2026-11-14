"""M5-04 PG版本观察/引用/审计定义；合成观察不能代替真实Worker取证。"""

from uuid import uuid4

import pytest

from harness.contracts.hashing import RunRequest
from harness.contracts.runtime import RuntimeFault
from tests.unit.test_evaluation_versions import synthetic_version

pytestmark = pytest.mark.integration


async def test_owned_start_and_terminal_references_are_persisted_and_scoped(prepared):
    store, actor, manifest, sid = prepared
    accepted = await store.create_run(actor, sid, RunRequest(request_id="version", message="查询", expected_revision=0))
    owner = "version-" + uuid4().hex
    await store.register_worker(owner)
    try:
        run = await store.claim(owner)
        assert run is not None and run.run_id == accepted.run_id
        start = await store.record_execution_version(owner, run, "start", synthetic_version(manifest))
        await store.started(owner, run, manifest.content_hash(), version_ref=start)
        terminal = await store.record_execution_version(owner, run, "terminal", synthetic_version(manifest))
        with pytest.raises(RuntimeFault, match="EXECUTION_VERSION_REFERENCE_MISMATCH"):
            await store.fail(owner, run.run_id, "TEST", version_ref=start)
        await store.fail(owner, run.run_id, "TEST", version_ref=terminal)
        events = await store.events(actor, run.run_id, 0)
        started = next(event for event in events if event["type"] == "run.started")
        assert started["payload"]["version_evidence_seq"] == start
        assert events[-1]["payload"]["version_evidence_seq"] == terminal
        assert events[-1]["payload"]["observed_manifest_hash"] == manifest.content_hash()
        evidence = await store.evaluation_evidence(actor, run.run_id)
        assert len(evidence.event_hashes) == len(events)
    finally:
        await store.fail(owner, accepted.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)


async def test_version_audit_outage_rolls_back_event_and_sequence(prepared, monkeypatch):
    store, actor, manifest, sid = prepared
    accepted = await store.create_run(actor, sid, RunRequest(request_id="audit-version", message="查询", expected_revision=0))
    owner = "version-" + uuid4().hex
    await store.register_worker(owner)
    original = store._audit

    async def fail_audit(conn, principal, action, resource):
        if action == "execution.version_observed":
            raise RuntimeError("synthetic audit failure")
        await original(conn, principal, action, resource)

    try:
        run = await store.claim(owner)
        assert run is not None
        monkeypatch.setattr(store, "_audit", fail_audit)
        with pytest.raises(RuntimeError):
            await store.record_execution_version(owner, run, "start", synthetic_version(manifest))
        events = await store.events(actor, run.run_id, 0)
        assert [event["type"] for event in events] == ["run.accepted"]
        monkeypatch.setattr(store, "_audit", original)
    finally:
        await store.fail(owner, accepted.run_id, "TEST_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)
