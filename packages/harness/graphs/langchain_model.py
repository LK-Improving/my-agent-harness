"""LangChain消息/模型桥接；只能调用注入的治理网关，不拥有供应商客户端。"""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from harness.context.projection import project_model_messages
from harness.contracts.models import ModelResult
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot

ModelDispatch = Callable[[JsonSnapshot, JsonSnapshot], Awaitable[ModelResult]]


def framework_context(messages: Sequence[BaseMessage]) -> JsonSnapshot:
    """先拒绝不支持的框架消息，再投影；私有additional_kwargs不发送给模型。"""
    for message in messages:
        if message.type not in {"human", "system", "ai", "tool"}:
            raise RuntimeFault("MODEL_MESSAGE_ROLE_UNSUPPORTED", 422)
        if not isinstance(message.content, str):
            raise RuntimeFault("MODEL_MESSAGE_CONTENT_UNSUPPORTED", 422)
        if isinstance(message, AIMessage) and message.invalid_tool_calls:
            raise RuntimeFault("MODEL_TOOL_CALL_INVALID", 422)
        if (
            isinstance(message, AIMessage)
            and message.response_metadata.get("finish_reason") in {"error", "aborted", "cancelled"}
        ) or (isinstance(message, ToolMessage) and message.status == "error"):
            raise RuntimeFault("MODEL_FAILED_MESSAGE_NOT_CONTEXT", 422)
    return project_model_messages(convert_to_openai_messages(list(messages)))


def result_message(result: ModelResult) -> AIMessage:
    if result.finish_reason in {"error", "aborted", "cancelled"}:
        raise RuntimeFault("MODEL_FAILED_MESSAGE_NOT_CONTEXT", 422)
    calls = result.tool_calls.as_object_array()
    if len(calls) > 1:
        raise RuntimeFault("PARALLEL_TOOL_BATCH_DISABLED", 422)
    normalized = []
    for call in calls:
        function = call.get("function")
        if (
            call.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
            or not isinstance(call.get("id"), str)
            or not call["id"]
        ):
            raise RuntimeFault("MODEL_TOOL_CALL_INVALID", 422)
        try:
            arguments = json.loads(str(function["arguments"]))
        except (ValueError, TypeError):
            raise RuntimeFault("MODEL_TOOL_ARGUMENTS_INVALID", 422) from None
        if not isinstance(arguments, dict):
            raise RuntimeFault("MODEL_TOOL_ARGUMENTS_INVALID", 422)
        normalized.append(
            {"name": function["name"], "args": arguments, "id": call["id"], "type": "tool_call"}
        )
    return AIMessage(
        content=result.content or "",
        tool_calls=normalized,
        id=result.model_attempt_id,
        additional_kwargs={
            "harness_call_id": result.model_call_id,
            "harness_attempt_id": result.model_attempt_id,
            "harness_draft_id": result.draft_id,
        },
    )


class GovernedChatModel(BaseChatModel):
    dispatch: ModelDispatch = Field(exclude=True, repr=False)
    tool_snapshot: JsonSnapshot = Field(default_factory=lambda: JsonSnapshot.capture([]))

    @property
    def _llm_type(self) -> str:
        return "harness-governed-single-attempt"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        if tool_choice is not None or kwargs:
            raise RuntimeFault("MODEL_PARAMETER_OVERRIDE_DISABLED")
        return self.model_copy(
            update={
                "tool_snapshot": JsonSnapshot.capture(
                    [convert_to_openai_tool(tool) for tool in tools]
                )
            }
        )

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        raise RuntimeFault("SYNC_MODEL_DISPATCH_DISABLED")

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if stop is not None or kwargs:
            raise RuntimeFault("MODEL_PARAMETER_OVERRIDE_DISABLED")
        result = await self.dispatch(framework_context(messages), self.tool_snapshot)
        return ChatResult(generations=[ChatGeneration(message=result_message(result))])
