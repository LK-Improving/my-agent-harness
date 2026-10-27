"""rag_chat准备器；通过统一执行器检索一次，保持完整assistant/tool事务组。"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness.contracts.hashing import canonical_json
from harness.contracts.artifacts import ArtifactProjector
from harness.contracts.runtime import ClaimedRun
from harness.tools.reads import ReadCoordinator

from domain_tools.knowledge.completion import KnowledgeCompletionValidator


class KnowledgeContext:
    def __init__(self, reads: ReadCoordinator, validator: KnowledgeCompletionValidator,
                 artifacts: ArtifactProjector | None = None) -> None:
        self.reads, self.validator = reads, validator
        self.artifacts = artifacts

    async def prepare(self, run: ClaimedRun, owner: str) -> list[Any]:
        receipt = await self.reads.execute(run, owner, "retrieve_knowledge", "rag-retrieve@1", {"query": run.request.message})
        call_id = "knowledge-" + receipt.operation_id
        data = (await self.artifacts.project(run, owner, receipt)).materialize() if self.artifacts else receipt.data
        # 首版只送当前问题和本轮证据，旧历史仍保存在PG，但不把可能已撤权的旧正文送入模型。
        return [HumanMessage(content=run.request.message), AIMessage(content="", id=call_id, tool_calls=[{
            "id": call_id, "name": "retrieve_knowledge", "args": {"query": run.request.message}, "type": "tool_call",
        }]), ToolMessage(content=canonical_json({"operation_id": receipt.operation_id, "data": data,
                                                "status": receipt.status}), tool_call_id=call_id)]

    async def authorize(self, run: ClaimedRun, owner: str) -> None:
        await self.validator.authorized(run, owner)
