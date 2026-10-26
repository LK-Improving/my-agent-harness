"""PG 持久首段标记和框架 middleware 入口的最小探针。"""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage

from tests.poc.agent_comparison import ScriptedToolModel
from tests.poc.support import ProbeStore

pytestmark = [pytest.mark.integration, pytest.mark.poc]


def generate_attempt(store, run_id, *, publish_before_failure):
    with store.connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS probe_streams(
                run_id text PRIMARY KEY, attempts integer NOT NULL DEFAULT 0,
                published boolean NOT NULL DEFAULT false, invalidated boolean NOT NULL DEFAULT false
            )
        """)
        conn.execute(
            "INSERT INTO probe_streams(run_id) VALUES(%s) ON CONFLICT DO NOTHING", (run_id,)
        )
        row = conn.execute(
            "SELECT * FROM probe_streams WHERE run_id=%s FOR UPDATE", (run_id,)
        ).fetchone()
        if row["published"]:
            return "rejected_after_publish"
        conn.execute("UPDATE probe_streams SET attempts=attempts+1 WHERE run_id=%s", (run_id,))
    if publish_before_failure:
        # 此提交代表已向公开事件存储发布，不能根据浏览器是否收到来决定重试。
        with store.connect() as conn:
            conn.execute("UPDATE probe_streams SET published=true WHERE run_id=%s", (run_id,))
        with store.connect() as conn:
            conn.execute("UPDATE probe_streams SET invalidated=true WHERE run_id=%s", (run_id,))
    return "stream_failed"


def test_first_delta_persistence_prevents_new_attempt_after_reconstruction(pg_schema):
    dsn, _ = pg_schema
    store = ProbeStore(dsn)
    assert generate_attempt(store, "published", publish_before_failure=True) == "stream_failed"
    assert (
        generate_attempt(ProbeStore(dsn), "published", publish_before_failure=True)
        == "rejected_after_publish"
    )
    assert generate_attempt(store, "not-published", publish_before_failure=False) == "stream_failed"
    assert generate_attempt(store, "not-published", publish_before_failure=False) == "stream_failed"
    with store.connect() as conn:
        rows = conn.execute("SELECT * FROM probe_streams ORDER BY run_id").fetchall()
    assert rows[0]["attempts"] == 2 and not rows[0]["published"]
    assert rows[1]["attempts"] == 1 and rows[1]["invalidated"]


def test_create_agent_hooks_can_gate_inputs_and_own_tool_dispatch(pg_schema):
    dsn, _ = pg_schema
    store = ProbeStore(dsn)
    order = []

    class InspectMiddleware(AgentMiddleware):
        def before_model(self, state, runtime):
            order.append("before_model")
            return None

        def after_model(self, state, runtime):
            order.append("after_model")
            return None

    def tool_proxy(order_id: str) -> str:
        """所有实际执行必须被统一工具包装接管。"""
        raise AssertionError("不应进入默认工具体")

    @wrap_tool_call
    def dispatch(request, handler):
        order.append("before_tool")
        if request.tool_call["args"]["order_id"] != "sandbox-order-001":
            raise PermissionError("工具参数超出授权范围")
        receipt = store.dispatch("hook-operation", request.tool_call["args"])
        order.append("after_tool")
        return ToolMessage(content=str(receipt), tool_call_id=request.tool_call["id"])

    model = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_proxy",
                        "args": {"order_id": "sandbox-order-001"},
                        "id": "hook-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已完成"),
        ]
    )
    graph = create_agent(model, [tool_proxy], middleware=[InspectMiddleware(), dispatch])
    graph.invoke({"messages": [{"role": "user", "content": "查询"}]})
    assert order == [
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "before_model",
        "after_model",
    ]
    assert store.counts()["sandbox_writes"] == 1
