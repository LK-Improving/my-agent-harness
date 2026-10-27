"""锁定版本官方saver全namespace/孤立副本清理定义；本批不运行。"""

import os
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from psycopg import AsyncConnection

from harness.adapters.cleanup_copies import ManagedTraceFiles, OfficialCheckpointCopies
from harness.bootstrap.checkpoints import saver
from harness.contracts.hashing import checkpoint_thread_id
from harness.contracts.snapshots import JsonSnapshot

pytestmark = pytest.mark.integration


async def test_official_thread_cleanup_removes_all_namespaces_blobs_writes_and_only_fixed_trace(prepared, tmp_path):
    # prepared已验证独立本机_m1_test；不回退业务库或其他项目配置。
    _, actor, _, _ = prepared
    dsn = os.environ["HARNESS_M1_TEST_DSN"]
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS harness_checkpoint")
    thread = checkpoint_thread_id(actor.tenant_id, actor.application_id, uuid4().hex)
    other = checkpoint_thread_id(actor.tenant_id, actor.application_id, uuid4().hex)
    async with saver(dsn) as checkpoint:
        await checkpoint.setup()
        for identity, namespace in ((thread, ""), (thread, "child"), (other, "")):
            state = empty_checkpoint()
            state["channel_values"] = {"private": "合成正文"}
            state["channel_versions"] = {"private": "1"}
            config = await checkpoint.aput({"configurable": {"thread_id": identity, "checkpoint_ns": namespace}},
                state, {"source": "input", "step": 0, "parents": {}}, {"private": "1"})
            await checkpoint.aput_writes(config, [("private", "合成待写正文")], "task")
        directory = tmp_path / ".runtime" / "traces"
        directory.mkdir(parents=True)
        (directory / (thread + ".jsonl")).write_text("合成trace", encoding="utf-8")
        (directory / (other + ".jsonl")).write_text("保留", encoding="utf-8")
        copies = OfficialCheckpointCopies(checkpoint, dsn, ManagedTraceFiles(tmp_path))
        target = JsonSnapshot.capture({"kind": "session", "threads": [thread]})
        assert not await copies.absent(target)
        await copies.erase(target)
        assert await copies.absent(target)
        await copies.erase(target)
        assert await checkpoint.aget_tuple({"configurable": {"thread_id": other}}) is not None
        assert (directory / (other + ".jsonl")).read_text("utf-8") == "保留"
        # 孤立blob也必须阻断“已清完”；alist为空不足以证明不存在副本。
        async with copies.uow.transaction() as conn:
            await conn.execute("""INSERT INTO harness_checkpoint.checkpoint_blobs(thread_id,checkpoint_ns,channel,version,type,blob)
                VALUES(%s,'orphan','private','orphan','bytes',%s)""", (thread, b"synthetic"))
        assert not await copies.absent(target)
        await copies.erase(target)
        assert await copies.absent(target)
