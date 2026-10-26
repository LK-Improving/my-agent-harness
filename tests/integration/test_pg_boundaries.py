"""验证 PG 约束、官方 pending writes 和执行者准入探针。"""

from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import pytest
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tests.poc.support import ProbeStore, saver

pytestmark = [pytest.mark.integration, pytest.mark.poc]


def test_worker_registration_survives_client_disconnect(pg_schema):
    dsn, _ = pg_schema
    store = ProbeStore(dsn)
    # 同时准入只能成功一个，连接断开不自动把登记转移给另一个执行者。
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(store.claim_worker, ["one", "two", "three", "four"]))
    assert sum(claims) == 1
    assert not ProbeStore(dsn).claim_worker("replacement")


def test_artifact_readback_permission_hash_and_unicode(pg_schema):
    dsn, _ = pg_schema
    store = ProbeStore(dsn)
    text = "预览" + "中😀" * 3000 + "必须读回的证据"
    digest = store.save_artifact("ref1", "tenant-a", text)
    page = store.read_artifact("ref1", "tenant-a", 6002, 100)
    assert page["content"] == "必须读回的证据"
    assert page["content_hash"] == digest
    assert page["next_offset"] == len(text)
    assert not page["has_more"]
    with pytest.raises(PermissionError):
        store.read_artifact("ref1", "tenant-b", 0, 10)
    with pytest.raises(ValueError):
        store.read_artifact("ref1", "tenant-a", 0, 4001)
    with store.connect() as conn:
        conn.execute("UPDATE probe_artifacts SET revoked=true WHERE ref='ref1'")
    with pytest.raises(PermissionError):
        store.read_artifact("ref1", "tenant-a", 0, 10)


def append(left, right):
    return left + right


class ParallelState(TypedDict):
    items: Annotated[list[str], append]


def test_pending_writes_do_not_reexecute_successful_parallel_node(pg_schema):
    dsn, _ = pg_schema
    invocations = {"good": 0, "bad": 0}

    def good(state):
        invocations["good"] += 1
        return {"items": ["good"]}

    def bad(state):
        invocations["bad"] += 1
        if invocations["bad"] == 1:
            raise RuntimeError("并行节点故障")
        return {"items": ["bad"]}

    def build(checkpointer):
        return (
            StateGraph(ParallelState)
            .add_node("good", good)
            .add_node("bad", bad)
            .add_edge(START, "good")
            .add_edge(START, "bad")
            .add_edge("good", END)
            .add_edge("bad", END)
            .compile(checkpointer=checkpointer)
        )

    config = {"configurable": {"thread_id": "pending-test"}}
    with saver(dsn) as checkpointer:
        with pytest.raises(RuntimeError, match="并行节点"):
            build(checkpointer).invoke({"items": []}, config, durability="sync")
        assert checkpointer.get_tuple(config).pending_writes
    with saver(dsn) as checkpointer:
        result = build(checkpointer).invoke(None, config, durability="sync")
        assert sorted(result["items"]) == ["bad", "good"]
    assert invocations == {"good": 1, "bad": 2}
