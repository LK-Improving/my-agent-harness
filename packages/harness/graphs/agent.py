"""create_agent的模型/工具包装与实际interrupt恢复。"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt
from pydantic import Field

from harness.context.projection import initial_context
from harness.contracts.artifacts import ArtifactProjector
from harness.contracts.execution import ExecutionLedger
from harness.contracts.hashing import canonical_hash, canonical_json, checkpoint_thread_id
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ToolSpec
from harness.contracts.state import TaskState, ValidationReport
from harness.contracts.writes import ExecutionSuspended, SuspensionProof, WriteOperation
from harness.graphs.langchain_model import GovernedChatModel as GovernedChatModel
from harness.runtime.extensions import ExtensionChain
from harness.runtime.models import GovernedModelGateway
from harness.runtime.releases import InstalledReleases
from harness.tools.reads import ReadCoordinator, RegisteredRead
from harness.tools.writes import RegisteredWrite, WriteCoordinator


class HarnessState(AgentState):
    harness_run_id: str
    harness_manifest_hash: str


class GovernanceBlocked(RuntimeFault, GraphBubbleUp):
    """通过官方bubble-up边界阻断，不被ToolNode转换成可继续执行的错误文本。"""


class FinalEnvelope(Contract):
    answer: str = Field(min_length=1, max_length=64000)
    answer_kind: Literal["answer", "clarification", "abstention"]
    task_outcome: Literal["achieved", "needs_input", "not_achieved"]
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=40)


class AgentEngine:
    def __init__(
        self,
        ledger: ExecutionLedger,
        gateway: GovernedModelGateway,
        reads: ReadCoordinator,
        writes: WriteCoordinator | None,
        checkpointer: Any,
        owner: str,
        manifests: dict[str, Any],
        prompts: dict[str, str],
        validate: Callable[[ClaimedRun, AnswerCandidate], Awaitable[ValidationReport]],
        task_guard: Callable[[ClaimedRun, ToolSpec, dict[str, Any]], None] | None = None,
        extensions: ExtensionChain | None = None,
        *,
        release_guard: Callable[[ReleaseManifest], None] | None = None,
        prepare_context: Callable[[ClaimedRun, str], Awaitable[list[Any]]] | None = None,
        authorize_context: Callable[[ClaimedRun, str], Awaitable[None]] | None = None,
        artifact_projector: ArtifactProjector | None = None,
    ) -> None:
        self.ledger, self.gateway, self.reads, self.writes = ledger, gateway, reads, writes
        self.checkpointer, self.owner, self.validate = (
            checkpointer,
            owner,
            validate,
        )
        self.task_guard = task_guard
        self.extensions = extensions or ExtensionChain()
        self.releases = InstalledReleases(tuple(manifests.values()))
        self.manifests = MappingProxyType(
            {item.spec.release_id: item for item in self.releases.manifests}
        )
        self.prompts = MappingProxyType(dict(prompts))
        self.release_guard = release_guard
        self.prepare_context, self.authorize_context = prepare_context, authorize_context
        self.artifact_projector = artifact_projector

    def observed_manifest_hash(self, release: Any) -> str:
        installed = self.releases.require(release)
        if self.release_guard is not None:
            self.release_guard(installed)
        return installed.content_hash()

    def execution_available(self) -> bool:
        if self.release_guard is not None:
            try:
                for release in self.releases.manifests:
                    self.release_guard(release)
            except RuntimeFault:
                return False
        conn = getattr(self.checkpointer, "conn", None)
        return conn is None or (not conn.closed and not conn.broken)

    @staticmethod
    def config(run: ClaimedRun) -> dict[str, Any]:
        # 每个run单独checkpoint线程；失败草稿不会污染其他run的会话历史。
        return {
            "configurable": {
                "thread_id": checkpoint_thread_id(run.principal.tenant_id, run.principal.application_id, run.run_id)
            },
            "recursion_limit": 64,
            "max_concurrency": 1,
        }

    def graph(self, run: ClaimedRun) -> Any:
        self.observed_manifest_hash(run.manifest)

        async def model(messages: JsonSnapshot, tools: JsonSnapshot) -> Any:
            if self.authorize_context is not None:
                await self.authorize_context(run, self.owner)
            return await self.gateway.complete(run, self.owner, messages, tools)

        async def forbidden(**kwargs: Any) -> str:
            raise RuntimeFault("TOOL_DISPATCH_BYPASS")

        catalog: dict[str, RegisteredRead | RegisteredWrite] = dict(self.reads.entries)
        if self.writes:
            for binding in run.manifest.spec.allowed_tools:
                if binding.name not in catalog:
                    catalog[binding.name] = self.writes.registry.get(binding.name)
        tools = []
        for binding in run.manifest.spec.allowed_tools:
            entry = catalog[binding.name]
            tools.append(
                StructuredTool.from_function(
                    coroutine=forbidden,
                    name=binding.name,
                    description=f"版本化受控工具 {binding.name}，只能经统一执行器调用",
                    args_schema=entry.input_type,
                )
            )

        async def governed_dispatch(request: Any) -> ToolMessage:
            call = request.tool_call
            last = next(
                message
                for message in reversed(request.state["messages"])
                if isinstance(message, AIMessage)
            )
            if len(last.tool_calls) != 1:
                raise RuntimeFault("PARALLEL_TOOL_BATCH_DISABLED")
            logical = "call-" + canonical_hash({"producer": last.id, "call_id": call["id"]})
            if call["name"] not in catalog:
                raise RuntimeFault("TOOL_NOT_ALLOWED", 403)
            # 统一准备器在扩展前后分别检查Schema、授权和领域目标。
            if call["name"] in self.reads.entries:
                receipt = await self.reads.execute(
                    run, self.owner, call["name"], logical, call["args"], task_guard=self.task_guard
                )
            else:
                if self.writes is None:
                    raise RuntimeFault("WRITE_CAPABILITY_NOT_INSTALLED", 503)
                proposal = await self.writes.prepare_proposal(
                    run,
                    call["name"],
                    logical,
                    call["args"],
                    canonical_hash({"request": run.request.business_hash(), "call": call}),
                    task_guard=self.task_guard,
                )
                operation = await self.writes.ledger.prepare_write(
                    run.principal, self.owner, proposal
                )
                payload = {
                    "operation_id": operation.operation_id,
                    "run_id": run.run_id,
                    "manifest_hash": run.manifest.content_hash(),
                    "binding_hash": operation.binding_hash,
                }
                resumed = interrupt({**payload, "kind": "approval"})
                if resumed != {
                    "operation_id": operation.operation_id,
                    "binding_hash": operation.binding_hash,
                }:
                    raise RuntimeFault("INTERRUPT_RESUME_BINDING_MISMATCH")
                spec = ToolSpec.model_validate(operation.tool.as_object())
                receipt = None
                for index in range(spec.retry.max_attempts_per_operation):
                    state, attempts = await self.ledger.operation_state(
                        run.principal, operation.operation_id
                    )
                    operation = WriteOperation.model_validate(state.as_object())
                    history = attempts.as_object_array()
                    if len(history) > index and history[index]["requires_reconciliation"]:
                        resumed = interrupt(
                            {
                                **payload,
                                "kind": "reconciliation",
                                "attempt_id": history[index]["attempt_id"],
                            }
                        )
                        if resumed != {
                            "operation_id": operation.operation_id,
                            "binding_hash": operation.binding_hash,
                        }:
                            raise RuntimeFault("INTERRUPT_RESUME_BINDING_MISMATCH")
                    if len(history) > index + 1:
                        continue
                    try:
                        receipt = await self.writes.execute(
                            run,
                            self.owner,
                            call["name"],
                            logical,
                            proposal.arguments.as_object(),
                            proposal.plan_hash,
                            prepared_proposal=proposal,
                            task_guard=self.task_guard,
                        )
                    except Exception:
                        state, history_snapshot = await self.ledger.operation_state(
                            run.principal, operation.operation_id
                        )
                        current = WriteOperation.model_validate(state.as_object())
                        if current.state not in {"unknown", "dispatched"}:
                            raise
                        history = history_snapshot.as_object_array()
                        # 异常后的gate在重放时由持久requires_reconciliation标记保证相同次序。
                        interrupt(
                            {
                                **payload,
                                "kind": "reconciliation",
                                "attempt_id": history[-1]["attempt_id"],
                            }
                        )
                        continue
                    break
                if receipt is None:
                    raise RuntimeFault("WRITE_ATTEMPTS_EXHAUSTED")
            if receipt is None:
                raise RuntimeFault("TOOL_RECEIPT_MISSING")
            data = (await self.artifact_projector.project(run, self.owner, receipt)).materialize() if self.artifact_projector else receipt.data
            return ToolMessage(
                content=canonical_json(
                    {
                        "operation_id": receipt.operation_id,
                        "data": data,
                        "status": receipt.status,
                    }
                ),
                tool_call_id=call["id"],
            )

        @wrap_tool_call
        async def dispatch(request: Any, handler: Any) -> ToolMessage:
            try:
                return await governed_dispatch(request)
            except GraphBubbleUp:
                raise
            except RuntimeFault as exc:
                raise GovernanceBlocked(exc.code, exc.status) from None
            except Exception:
                raise GovernanceBlocked("TOOL_GOVERNANCE_FAILED") from None

        return create_agent(
            GovernedChatModel(dispatch=model, cache=False, disable_streaming=True),
            tools=tools,
            system_prompt=self.prompts[run.manifest.spec.release_id],
            middleware=[dispatch],
            state_schema=HarnessState,
            checkpointer=self.checkpointer,
        )

    async def execute(self, run: ClaimedRun) -> AnswerCandidate:
        self.observed_manifest_hash(run.manifest)
        await self.extensions.run(
            "before_run",
            run,
            JsonSnapshot.capture(
                {
                    "session_id": run.session_id,
                    "input_hash": run.request.business_hash(),
                }
            ),
        )
        candidate = await self.ledger.candidate_for_run(run.principal, run.run_id)
        if candidate is not None:
            return await self.validated_candidate(run, candidate)
        prepared_context = (await self.prepare_context(run, self.owner)
                            if self.prepare_context is not None else [])
        graph, config = self.graph(run), self.config(run)
        state = await graph.aget_state(config)
        if state.values:
            if (
                state.values.get("harness_manifest_hash") != run.manifest.content_hash()
                or state.values.get("harness_run_id") != run.run_id
            ):
                raise RuntimeFault("CHECKPOINT_BINDING_MISMATCH")
        value: Any
        if run.suspension:
            proof = run.suspension.as_object()
            checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id")
            if checkpoint_id != proof.get("checkpoint_ref"):
                raise RuntimeFault("CHECKPOINT_HEAD_CHANGED")
            if proof.get("kind") == "continuation":
                value = None
            else:
                matches = [
                    item
                    for task in state.tasks
                    for item in task.interrupts
                    if item.id == proof.get("interrupt_ref")
                    and item.value.get("operation_id") == proof.get("operation_id")
                ]
                if len(matches) != 1:
                    raise RuntimeFault("CHECKPOINT_INTERRUPT_MISMATCH")
                binding = matches[0].value["binding_hash"]
                value = Command(
                    resume={
                        matches[0].id: {
                            "operation_id": proof["operation_id"],
                            "binding_hash": binding,
                        }
                    }
                )
        elif state.values:
            # 已完成图但业务提交前崩溃，只复用checkpoint的最终输出。
            if state.next:
                raise RuntimeFault("UNCONTROLLED_CHECKPOINT_RESUME")
            value = None
        else:
            value = {
                "messages": prepared_context if self.prepare_context is not None else initial_context(run.history, run.request.message),
                "harness_run_id": run.run_id,
                "harness_manifest_hash": run.manifest.content_hash(),
            }
        result = await graph.ainvoke(value, config, durability="sync")
        if result.get("__interrupt__"):
            raise ExecutionSuspended(result["__interrupt__"][0].value["operation_id"])
        final = result["messages"][-1]
        if (
            not isinstance(final, AIMessage)
            or final.tool_calls
            or not isinstance(final.content, str)
        ):
            raise RuntimeFault("FINAL_ANSWER_INVALID", 422)
        envelope = FinalEnvelope.model_validate_json(final.content)
        candidate = AnswerCandidate(
            draft_id=final.additional_kwargs["harness_draft_id"],
            model_call_id=final.additional_kwargs["harness_call_id"],
            model_attempt_id=final.additional_kwargs["harness_attempt_id"],
            answer=envelope.answer,
            answer_kind=envelope.answer_kind,
            task_outcome=envelope.task_outcome,
            task=TaskState(objective=run.request.message, source_turn_id=run.run_id, evidence_refs=envelope.evidence_refs),
            validation=ValidationReport(
                validator_ref=run.manifest.spec.completion_validator_ref,
                input_hash="0" * 64,
                verdict="fail",
                checks={},
                checked_at=datetime.now(UTC),
            ),
        )
        return await self.validated_candidate(run, candidate)

    async def validated_candidate(
        self, run: ClaimedRun, candidate: AnswerCandidate
    ) -> AnswerCandidate:
        await self.extensions.run(
            "validate_completion", run, JsonSnapshot.capture(candidate.validation_input())
        )
        return candidate.model_copy(update={"validation": await self.validate(run, candidate)})

    async def before_commit(self, run: ClaimedRun, candidate: AnswerCandidate) -> None:
        self.observed_manifest_hash(run.manifest)
        await self.extensions.run(
            "before_commit",
            run,
            JsonSnapshot.capture(
                {
                    "candidate": candidate.model_dump(mode="json"),
                }
            ),
        )

    async def suspension_proof(self, run: ClaimedRun, operation_id: str) -> SuspensionProof:
        state = await self.graph(run).aget_state(self.config(run))
        items = [
            item
            for task in state.tasks
            for item in task.interrupts
            if isinstance(item.value, dict) and item.value.get("operation_id") == operation_id
        ]
        if (
            len(items) != 1
            or state.values.get("harness_manifest_hash") != run.manifest.content_hash()
        ):
            raise RuntimeFault("CHECKPOINT_SUSPENSION_UNVERIFIED")
        return SuspensionProof(
            run_id=run.run_id,
            operation_id=operation_id,
            manifest_hash=run.manifest.content_hash(),
            checkpoint_ref=state.config["configurable"]["checkpoint_id"],
            interrupt_ref=items[0].id,
        )
