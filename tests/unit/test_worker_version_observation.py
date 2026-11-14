"""Worker版本采集的线程/阻断/暂停定义；无数据库、图或供应商执行。"""

import threading
from types import SimpleNamespace

import pytest

from harness.contracts.runtime import RuntimeFault
from harness.contracts.writes import ExecutionSuspended
from harness.runtime.worker import Worker
from tests.unit.test_evaluation_collector import candidate
from tests.unit.test_evaluation_versions import synthetic_version


class Store:
    def __init__(self):
        self.records = []

    async def record_execution_version(self, owner, run, phase, observation, error_code):
        self.records.append((phase, observation, error_code))
        return len(self.records)

    async def started(self, owner, run, observed, *, version_ref=None):
        self.started_reference = version_ref

    async def heartbeat(self, owner):
        pass

    async def continue_run(self, owner, run_id):
        return True

    async def fail(self, owner, run_id, code, *, version_ref=None):
        self.failed_reference = version_ref


class Engine:
    def __init__(self, suspend=False):
        self.calls, self.suspend = 0, suspend

    def observed_manifest_hash(self, manifest):
        return manifest.content_hash()

    async def execute(self, run):
        self.calls += 1
        if self.suspend:
            raise ExecutionSuspended("operation")
        raise RuntimeFault("SYNTHETIC_ENGINE_FAILURE")


async def test_version_collector_runs_off_loop_and_failed_execution_keeps_terminal_reference():
    manifest, store, engine = candidate(), Store(), Engine()
    loop_thread = threading.get_ident()
    threads = []

    def observe(release):
        threads.append(threading.get_ident())
        return synthetic_version(release)

    worker = Worker(store, engine, version_observer=observe)
    run = SimpleNamespace(run_id="run-test", manifest=manifest)
    with pytest.raises(RuntimeFault, match="SYNTHETIC_ENGINE_FAILURE"):
        await worker._execute(run)
    await worker._fail(run, "SYNTHETIC_ENGINE_FAILURE")
    assert all(thread != loop_thread for thread in threads)
    assert [phase for phase, *_ in store.records] == ["start", "terminal"]
    assert store.failed_reference == 2 and engine.calls == 1


async def test_unverified_start_never_executes_graph_and_pause_does_not_fake_terminal():
    manifest, store, engine = candidate(), Store(), Engine()
    worker = Worker(store, engine, version_observer=lambda release: synthetic_version(release).model_copy(
        update={"configuration_consistent": False}))
    with pytest.raises(RuntimeFault, match="EXECUTION_VERSION_UNVERIFIED"):
        await worker._execute(SimpleNamespace(run_id="run-test", manifest=manifest))
    assert engine.calls == 0
    store, engine = Store(), Engine(suspend=True)
    worker = Worker(store, engine, version_observer=synthetic_version)
    with pytest.raises(ExecutionSuspended):
        await worker._execute(SimpleNamespace(run_id="run-test", manifest=manifest))
    assert [phase for phase, *_ in store.records] == ["start"]
