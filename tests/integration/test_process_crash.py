"""真实子进程退出后恢复，验证 sync 和单次业务提交窗口。"""

import os
import subprocess
import sys

import pytest

from tests.poc.support import ProbeStore

pytestmark = [pytest.mark.integration, pytest.mark.poc]


@pytest.mark.parametrize("phase,exit_code", [("before_commit", 31), ("after_commit", 32)])
def test_killed_process_recovers_without_second_effect_or_commit(pg_schema, phase, exit_code):
    dsn, _ = pg_schema
    env = {**os.environ, "HARNESS_POC_SCOPED_DSN": dsn, "PYTHONUTF8": "1"}
    command = [sys.executable, "-m", "tests.poc.crash_worker"]
    first = subprocess.run([*command, phase], env=env, capture_output=True, timeout=30)
    assert first.returncode == exit_code, first.stderr.decode("utf-8", errors="replace")
    # subprocess 已回收，退出码为旧进程终止证据；不靠心跳或租约到期猜测。
    store = ProbeStore(dsn)
    assert store.counts()["sandbox_writes"] == 1
    resumed = subprocess.run([*command, "resume"], env=env, capture_output=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr.decode("utf-8", errors="replace")
    assert store.counts() == {
        "sandbox_writes": 1,
        "probe_commits": 1,
        "probe_turns": 1,
        "probe_events": 1,
    }
    assert store.operation("crash-operation")["dispatches"] == 1
