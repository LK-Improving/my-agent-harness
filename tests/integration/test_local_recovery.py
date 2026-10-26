"""本机受控恢复定义：专用PG+受控子进程；不执行真实模型/业务工具。"""

import asyncio
import os
import subprocess
import sys
from uuid import uuid4

import psutil
import pytest
from harness.adapters.local_process import current_process, process_marker, prove_terminated
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.contracts.runtime import AdmissionProfile, RuntimeFault

pytestmark = pytest.mark.integration


async def test_live_child_blocks_retirement_then_dead_child_allows_new_worker(
    prepared, tmp_path, monkeypatch
):
    _, original, _, _ = prepared
    monkeypatch.chdir(tmp_path)
    actor = original.model_copy(
        update={
            "roles": frozenset({"recovery_operator"}),
            "resource_scopes": frozenset({"harness:recovery:local"}),
        }
    )
    store = PostgresRecoveryStore(os.environ["HARNESS_M1_TEST_DSN"], AdmissionProfile())
    child = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    identity = uuid4().hex
    try:
        captured = current_process().model_copy(
            update={
                "pid": child.pid,
                "started_at": psutil.Process(child.pid).create_time(),
                "start_marker": process_marker(child.pid),
            }
        )
        store.process_identity = captured
        await store.register_worker(identity)
        with pytest.raises(RuntimeFault, match="OLD_WORKER_STILL_PRESENT"):
            prove_terminated(await store.registered_process(actor, identity))
        with pytest.raises(RuntimeFault, match="WORKER_ALREADY_REGISTERED"):
            await store.register_worker(uuid4().hex)
        child.terminate()
        await asyncio.to_thread(child.wait, timeout=5)
        proof = prove_terminated(await store.registered_process(actor, identity))
        await store.retire_worker(actor, identity, proof)
        fresh = uuid4().hex
        store.process_identity = current_process()
        await store.register_worker(fresh)
        await store.release_worker(fresh)
    finally:
        if child.poll() is None:
            child.terminate()
            await asyncio.to_thread(child.wait, timeout=5)
