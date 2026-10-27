"""只接受原始用户query，KB/快照和身份一律从服务端run绑定取得。"""

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.retrieval import KnowledgeRetrieval, RetrievalQuery, RetrievalResult
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ToolSpec
from harness.contracts.state import ReceiptState, ToolReceipt
from harness.tools.reads import RegisteredRead


class KnowledgeInput(Contract):
    query: str = Field(min_length=1, max_length=8000)


class KnowledgeOutput(Contract):
    source: Literal["authorized-knowledge@1"] = "authorized-knowledge@1"
    results: tuple[RetrievalResult, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_results(self) -> Self:
        if len({result.retrieval_id for result in self.results}) != len(self.results):
            raise ValueError("知识工具不能重复计入同一检索结果")
        return self


def require_query(run: ClaimedRun, spec: ToolSpec, args: dict[str, Any]) -> None:
    if args != {"query": run.request.message}:
        raise RuntimeFault("KNOWLEDGE_QUERY_CHANGED", 403)
    if not run.manifest.spec.knowledge_bindings or run.manifest.spec.graph_template != "rag_chat":
        raise RuntimeFault("KNOWLEDGE_BINDING_REQUIRED", 403)


def receipt_output(run: ClaimedRun, receipt: ToolReceipt) -> KnowledgeOutput:
    if (receipt.status != ReceiptState.SUCCEEDED or receipt.cost_status != "unknown"
        or receipt.args_hash != canonical_hash({"query": run.request.message})):
        raise RuntimeFault("KNOWLEDGE_TOOL_RECEIPT_INVALID", 409)
    return KnowledgeOutput.model_validate_json(canonical_json(receipt.data), strict=True)


def registered_knowledge(service: KnowledgeRetrieval) -> RegisteredRead:
    async def forbidden(actor: Principal, args: dict[str, Any]) -> Any:
        raise RuntimeFault("KNOWLEDGE_CONTEXT_REQUIRED")

    async def query(run: ClaimedRun, owner: str, logical: str, args: dict[str, Any]) -> dict[str, Any]:
        results = []
        bindings = sorted(run.manifest.spec.knowledge_bindings, key=lambda item: item.knowledge_base_id)
        if len(bindings) > 4 or len({b.knowledge_base_id for b in bindings}) != len(bindings):
            raise RuntimeFault("KNOWLEDGE_BINDINGS_INVALID", 422)
        for binding in bindings:
            results.append(await service.retrieve(
                run.principal, owner, run.run_id, binding.knowledge_base_id,
                RetrievalQuery(request_id=logical, query=args["query"]),
            ))
        return KnowledgeOutput(results=tuple(results)).model_dump(mode="json")

    async def revalidate(run: ClaimedRun, owner: str, receipt: ToolReceipt) -> None:
        output = receipt_output(run, receipt)
        if len(output.results) != len(run.manifest.spec.knowledge_bindings):
            raise RuntimeFault("KNOWLEDGE_RESULT_COUNT_MISMATCH", 409)
        for result in output.results:
            await service.authorize_result(run.principal, owner, run.run_id, result)

    spec = ToolSpec(
        name="retrieve_knowledge", version="1", handler_ref="domain_tools.knowledge.retrieve@1", effect="read",
        input_schema=KnowledgeInput.model_json_schema(), output_schema=KnowledgeOutput.model_json_schema(),
        authorization_policy_ref="knowledge-run-snapshot@1", timeout_ms=125000,
        cost={"billable": True}, sensitive_fields=("query", "results.evidence.body"),
    )
    return RegisteredRead(spec=JsonSnapshot.capture(spec.model_dump(mode="json")), input_type=KnowledgeInput,
                           output_type=KnowledgeOutput, handler=forbidden,
                           authorize=lambda actor, args: bool(actor.roles & {"resource_reader", "knowledge_manager"}),
                           contextual_handler=query, revalidate=revalidate, accounting_ref="child-model-ledger@1")
