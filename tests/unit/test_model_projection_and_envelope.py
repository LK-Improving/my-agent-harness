"""M2R-02投影/桥接/最终封装用例定义；未执行，不产生真实模型请求。"""

import json

import pytest
from harness.adapters.models.config import QwenSettings
from harness.adapters.models.qwen import QwenClient
from harness.context.projection import initial_context, project_model_messages
from harness.context.requests import envelope_tokens
from harness.contracts.failures import ProviderError
from harness.contracts.model_envelope import (
    chat_envelope,
    chat_envelope_tokens,
    prepared_chat_envelope,
)
from harness.contracts.models import (
    ModelParameters,
    ModelResult,
    ModelUsage,
    PreparedModelRequest,
    model_input_hash,
)
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.graphs.langchain_model import GovernedChatModel, framework_context, result_message
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.unit.test_m0_boundaries import prepared


def tool_group() -> list[dict]:
    return [
        {"role": "user", "content": "查订单"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"order_id":"合成订单"}'},
                }
            ],
            "additional_kwargs": {"harness_draft_id": "private-draft"},
            "response_metadata": {"private": "secret"},
        },
        {"role": "tool", "content": "合成结果", "tool_call_id": "call-a"},
    ]


def measured_request() -> PreparedModelRequest:
    request = prepared()
    parameters = ModelParameters(temperature=0, top_p=1, seed=0, stop=("结束" * 30,))
    wire = chat_envelope(
        request.route.model, request.messages, request.tools, request.max_output_tokens, parameters
    )
    budget = request.budget.model_copy(
        update={
            "input_tokens": chat_envelope_tokens(wire),
            "remaining_total_tokens": 10000,
            "measured_input_hash": model_input_hash(
                request.route,
                request.messages,
                request.tools,
                parameters,
                request.max_output_tokens,
            ),
        }
    )
    return PreparedModelRequest.model_validate(
        {**request.model_dump(mode="python"), "parameters": parameters, "budget": budget}
    )


def test_projection_strips_private_metadata_without_mutating_source() -> None:
    messages = tool_group()
    projected = project_model_messages(messages)
    assert "private-draft" not in projected.canonical_text
    assert "secret" not in projected.canonical_text
    assert "additional_kwargs" in messages[1]
    messages[2]["content"] = "changed"
    assert projected.as_object_array()[2]["content"] == "合成结果"
    assert initial_context(JsonSnapshot.capture([]), "新问题") == [
        {"role": "user", "content": "新问题"}
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "audit", "content": "private approval"}],
        [{"role": "user", "content": "text", "authorization": "secret"}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": "private"}]}],
        [{"role": "tool", "content": "orphan", "tool_call_id": "missing"}],
        tool_group()[:2],
        [*tool_group()[:2], {"role": "tool", "content": "wrong", "tool_call_id": "wrong"}],
    ],
)
def test_projection_rejects_untrusted_roles_fields_content_and_partial_groups(messages) -> None:
    with pytest.raises(RuntimeFault):
        project_model_messages(messages)


def test_framework_bridge_keeps_ids_private_and_tool_group_complete() -> None:
    messages = [
        SystemMessage(content="受控提示"),
        HumanMessage(content="查询"),
        AIMessage(
            content="",
            id="attempt",
            tool_calls=[
                {
                    "name": "lookup",
                    "args": {"order_id": "demo"},
                    "id": "call-a",
                    "type": "tool_call",
                }
            ],
            additional_kwargs={"harness_draft_id": "private-draft"},
        ),
        ToolMessage(content="结果", tool_call_id="call-a"),
    ]
    wire = framework_context(messages)
    assert "private-draft" not in wire.canonical_text
    assert wire.as_object_array()[-1]["tool_call_id"] == "call-a"
    with pytest.raises(RuntimeFault):
        framework_context(messages[:-1])
    with pytest.raises(RuntimeFault, match="MODEL_FAILED_MESSAGE_NOT_CONTEXT"):
        framework_context(
            [AIMessage(content="failed draft", response_metadata={"finish_reason": "error"})]
        )


def test_envelope_includes_all_parameters_and_preserves_default_count() -> None:
    request = measured_request()
    wire = prepared_chat_envelope(request)
    body = wire.as_object()
    assert body["temperature"] == 0
    assert body["seed"] == 0
    assert body["stop"] == list(request.parameters.stop)
    assert body["stream"] is False
    assert "tools" not in body
    assert chat_envelope_tokens(wire) == envelope_tokens(
        request.route.model,
        request.messages.as_object_array(),
        [],
        request.max_output_tokens,
        request.parameters,
    )
    default = chat_envelope(request.route.model, request.messages, request.tools, 128)
    old_wire = JsonSnapshot.capture(
        {
            "model": request.route.model,
            "messages": request.messages.materialize(),
            "max_tokens": 128,
            "stream": False,
        }
    )
    assert default == old_wire
    assert chat_envelope_tokens(default) == len(old_wire.canonical_text.encode("utf-8")) + 320
    assert chat_envelope_tokens(wire) > chat_envelope_tokens(default)


def test_counted_request_rejects_underquote_or_post_count_parameter_change() -> None:
    request = measured_request()
    with pytest.raises(ValueError, match="已计数"):
        prepared_chat_envelope(
            request.model_copy(
                update={"budget": request.budget.model_copy(update={"input_tokens": 1})}
            )
        )
    with pytest.raises(ValueError):
        prepared_chat_envelope(
            request.model_copy(update={"parameters": ModelParameters(stop=("后续篡改",))})
        )
    with pytest.raises(ValueError):
        chat_envelope(
            "model",
            JsonSnapshot.capture([{"role": "user", "content": "query", "audit": "private"}]),
            JsonSnapshot.capture([]),
            128,
        )


async def test_qwen_sends_exact_counted_envelope_and_blocks_before_transport() -> None:
    request = measured_request()

    class CapturingClient(QwenClient):
        def __init__(self) -> None:
            super().__init__(QwenSettings())
            self.sent: list[JsonSnapshot] = []

        async def _post(self, url, body, *, settings):
            self.sent.append(JsonSnapshot.capture(body))
            return json.dumps(
                {
                    "model": request.route.model,
                    "choices": [{"message": {"content": "结果"}, "finish_reason": "stop"}],
                }
            ).encode(), "controlled-request"

    client = CapturingClient()
    result = await client.complete(request)
    assert client.sent == [prepared_chat_envelope(request)]
    assert result.usage.input_tokens is None
    assert result.request_fingerprint == request.fingerprint()
    underquoted = request.model_copy(
        update={"budget": request.budget.model_copy(update={"input_tokens": 1})}
    )
    with pytest.raises(ProviderError) as blocked:
        await client.complete(underquoted)
    assert blocked.value.failure.dispatch_status == "not_sent"
    assert len(client.sent) == 1


async def test_model_bridge_calls_only_injected_gateway_and_rejects_overrides() -> None:
    captured = []

    async def dispatch(messages, tools):
        captured.append((messages, tools))
        return ModelResult(
            model="controlled",
            content="结果",
            finish_reason="stop",
            request_id=None,
            usage=ModelUsage(),
            model_call_id="call",
            model_attempt_id="attempt",
            request_fingerprint="a" * 64,
            draft_id="draft",
        )

    model = GovernedChatModel(dispatch=dispatch, cache=False, disable_streaming=True)
    response = await model.ainvoke([HumanMessage(content="问题")])
    assert response.id == "attempt"
    assert response.additional_kwargs["harness_draft_id"] == "draft"
    assert len(captured) == 1
    with pytest.raises(RuntimeFault):
        model.bind_tools([], tool_choice="required")
    with pytest.raises(RuntimeFault):
        await model.ainvoke([HumanMessage(content="问题")], temperature=1)
    assert len(captured) == 1


def test_tool_result_bridge_rejects_batch_before_any_tool_dispatch() -> None:
    result = ModelResult(
        model="controlled",
        content=None,
        finish_reason="tool_calls",
        request_id=None,
        usage=ModelUsage(),
        model_call_id="call",
        model_attempt_id="attempt",
        request_fingerprint="a" * 64,
        tool_calls=JsonSnapshot.capture(
            [
                {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            ]
        ),
    )
    with pytest.raises(RuntimeFault, match="PARALLEL_TOOL_BATCH_DISABLED"):
        result_message(result)
    with pytest.raises(RuntimeFault, match="MODEL_FAILED_MESSAGE_NOT_CONTEXT"):
        result_message(result.model_copy(update={"finish_reason": "error"}))
