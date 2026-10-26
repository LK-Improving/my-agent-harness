"""同一治理探针比较 create_agent 与自定义图，不作为业务服务。"""

from typing import Any

from harness.contracts.hashing import canonical_hash
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from tests.poc.support import ProbeStore, UnknownOutcome


class ScriptedToolModel(FakeMessagesListChatModel):
    """受控模型只验证框架协议，不冒充真实供应商。"""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 以恢复后的消息为准，不能用进程内响应游标代表持久执行位置。
        answer = self.responses[1] if messages[-1].type == "tool" else self.responses[0]
        return ChatResult(generations=[ChatGeneration(message=answer)])


def governed_call(store: ProbeStore, operation_id: str, args: dict[str, Any]):
    proposal = {"kind": "approval", "operation_id": operation_id, "args_hash": canonical_hash(args)}
    approved = interrupt(proposal)
    if not isinstance(approved, dict) or approved != {"approved_args_hash": proposal["args_hash"]}:
        raise PermissionError("审批不精确匹配当前参数")
    try:
        return store.dispatch(operation_id, args, lose_response=True)
    except UnknownOutcome:
        interrupt({"kind": "reconciliation", "operation_id": operation_id})
        conclusion = store.reconcile(operation_id)
        if conclusion["conclusion"] != "confirmed_succeeded":
            raise UnknownOutcome(operation_id) from None
        return conclusion["evidence"]


class WorkState(TypedDict, total=False):
    args: dict[str, Any]
    result: dict[str, Any]


def custom_graph(store: ProbeStore, operation_id: str, checkpointer):
    def execute(state: WorkState):
        return {"result": governed_call(store, operation_id, state["args"])}

    return (
        StateGraph(WorkState)
        .add_node("governed_tool", execute)
        .add_edge(START, "governed_tool")
        .add_edge("governed_tool", END)
        .compile(checkpointer=checkpointer)
    )


def agent_graph(store: ProbeStore, operation_id: str, checkpointer, args):
    def create_work_order(order_id: str) -> str:
        """由受控 middleware 提供结果，默认工具体不能直接派发。"""
        raise AssertionError("工具绕过统一治理入口")

    @wrap_tool_call
    def governed_tool(request, handler):
        result = governed_call(store, operation_id, request.tool_call["args"])
        return ToolMessage(content=str(result), tool_call_id=request.tool_call["id"])

    model = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "create_work_order", "args": args, "id": "call-1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="已取得合法回执"),
        ]
    )
    return create_agent(
        model, tools=[create_work_order], middleware=[governed_tool], checkpointer=checkpointer
    )
