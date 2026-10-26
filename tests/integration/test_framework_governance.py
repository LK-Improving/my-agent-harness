"""真实 PG 上比较两种框架装配的审批、unknown 与回执恢复。"""

import pytest
from harness.contracts.hashing import canonical_hash
from langgraph.types import Command

from tests.poc.agent_comparison import agent_graph, custom_graph
from tests.poc.support import ProbeStore, UnknownOutcome, saver

pytestmark = [pytest.mark.integration, pytest.mark.poc]


@pytest.mark.parametrize("mode", ["create_agent", "custom_graph"])
def test_approval_unknown_resume_without_duplicate_write(pg_schema, mode, record_property):
    dsn, _ = pg_schema
    store = ProbeStore(dsn)
    args = {"order_id": "sandbox-order-001"}
    operation_id = "stable-operation-1"
    config = {"configurable": {"thread_id": "thread-1"}}

    def build(checkpointer):
        return (
            agent_graph(store, operation_id, checkpointer, args)
            if mode == "create_agent"
            else custom_graph(store, operation_id, checkpointer)
        )

    # 每一阶段重新建立 saver 和图，验证不依赖旧对象内存恢复。
    with saver(dsn) as checkpointer:
        graph = build(checkpointer)
        initial = (
            {"messages": [{"role": "user", "content": "为订单创建工单"}]}
            if mode == "create_agent"
            else {"args": args}
        )
        first = graph.invoke(initial, config, durability="sync")
        assert first["__interrupt__"][0].value["kind"] == "approval"
        assert store.counts()["sandbox_writes"] == 0
        assert graph.get_state(config).tasks[0].interrupts

    with saver(dsn) as checkpointer:
        graph = build(checkpointer)
        second = graph.invoke(
            Command(resume={"approved_args_hash": canonical_hash(args)}), config, durability="sync"
        )
        assert second["__interrupt__"][0].value["kind"] == "reconciliation"
        assert store.counts()["sandbox_writes"] == 1
        assert store.reconcile(operation_id)["conclusion"] == "still_unknown"
        with pytest.raises(UnknownOutcome):
            store.dispatch(operation_id, args)
        assert store.operation(operation_id)["dispatches"] == 1

    store.make_visible(operation_id)
    with saver(dsn) as checkpointer:
        graph = build(checkpointer)
        result = graph.invoke(Command(resume={"check_evidence": True}), config, durability="sync")
        assert "__interrupt__" not in result
        receipt = store.operation(operation_id)
        assert receipt["status"] == "succeeded"
        assert receipt["dispatches"] == 1
        assert store.counts()["sandbox_writes"] == 1
        store.commit("run-1", receipt["result"])
        store.commit("run-1", receipt["result"])
        assert store.counts() == {
            "sandbox_writes": 1,
            "probe_commits": 1,
            "probe_turns": 1,
            "probe_events": 1,
        }
        if mode == "create_agent":
            messages = result["messages"]
            tool_messages = [m for m in messages if m.type == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0].tool_call_id == "call-1"
    record_property("assembly", mode)
    record_property("external_writes", 1)
    record_property("dispatches", 1)


@pytest.mark.parametrize("mode", ["create_agent", "custom_graph"])
def test_changed_approval_cannot_dispatch(pg_schema, mode):
    dsn, _ = pg_schema
    store = ProbeStore(dsn)
    args = {"order_id": "sandbox-order-001"}
    config = {"configurable": {"thread_id": "approval-test"}}
    with saver(dsn) as checkpointer:
        graph = (
            agent_graph(store, "op", checkpointer, args)
            if mode == "create_agent"
            else custom_graph(store, "op", checkpointer)
        )
        initial = (
            {"messages": [{"role": "user", "content": "创建"}]}
            if mode == "create_agent"
            else {"args": args}
        )
        graph.invoke(initial, config, durability="sync")
        with pytest.raises(PermissionError):
            graph.invoke(
                Command(resume={"approved_args_hash": canonical_hash({"order_id": "other"})}),
                config,
                durability="sync",
            )
    assert store.counts()["sandbox_writes"] == 0
