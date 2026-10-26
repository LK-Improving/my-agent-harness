"""独立进程模拟 checkpoint 与业务提交之间的真实终止窗口。"""

import os
import sys

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tests.poc.support import ProbeStore, saver


class CrashState(TypedDict, total=False):
    result: dict[str, str]


def main() -> None:
    phase = sys.argv[1]
    dsn = os.environ["HARNESS_POC_SCOPED_DSN"]
    store = ProbeStore(dsn)

    def execute(state):
        return {"result": store.dispatch("crash-operation", {"order_id": "sandbox-order-001"})}

    def finalize(state):
        if phase == "before_commit":
            os._exit(31)
        result = store.commit("crash-run", state["result"])
        if phase == "after_commit":
            os._exit(32)
        return {"result": result}

    with saver(dsn) as checkpointer:
        graph = (
            StateGraph(CrashState)
            .add_node("execute", execute)
            .add_node("finalize", finalize)
            .add_edge(START, "execute")
            .add_edge("execute", "finalize")
            .add_edge("finalize", END)
            .compile(checkpointer=checkpointer)
        )
        graph.invoke(
            None if phase == "resume" else {},
            {"configurable": {"thread_id": "crash-thread"}},
            durability="sync",
        )


if __name__ == "__main__":
    main()
