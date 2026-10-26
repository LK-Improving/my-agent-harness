"""Completions缓冲请求的纯封装；计数与传输从同一深快照取值。"""

from pydantic import JsonValue

from harness.contracts.models import ModelParameters, PreparedModelRequest
from harness.contracts.snapshots import JsonSnapshot


def chat_envelope(
    model: str,
    messages: JsonSnapshot,
    tools: JsonSnapshot,
    max_output: int,
    parameters: ModelParameters | None = None,
) -> JsonSnapshot:
    """不读取环境、不加认证header、不进行网络调用或协议外参数补丁。"""
    if not model or type(max_output) is not int or max_output <= 0:
        raise ValueError("模型和输出上限必须明确有效")
    projected = messages.as_object_array()
    if not projected:
        raise ValueError("请求封装消息不能为空")
    for message in projected:
        role = message.get("role")
        allowed = {"role", "content", "name"}
        if role == "assistant":
            allowed.add("tool_calls")
        elif role == "tool":
            allowed.add("tool_call_id")
        if role not in {"system", "user", "assistant", "tool"} or set(message) - allowed:
            raise ValueError("最终请求只接受已投影的消息字段")
    params = ModelParameters.model_validate_json(
        (parameters or ModelParameters()).model_dump_json()
    )
    body: dict[str, JsonValue] = {
        "model": model,
        "messages": [*projected],
        "max_tokens": max_output,
        "stream": False,
    }
    body.update(params.model_dump(mode="json", exclude_none=True))
    declared_tools = tools.as_object_array()
    if declared_tools:
        body["tools"] = [*declared_tools]
    return JsonSnapshot.capture(body)


def chat_envelope_tokens(envelope: JsonSnapshot) -> int:
    """沿用utf8-envelope@1保守估计；计入全部参数，并非供应商精确Tokenizer。"""
    body = envelope.as_object()
    messages, tools = body.get("messages"), body.get("tools", [])
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise ValueError("请求封装必须包含消息和工具数组")
    return len(envelope.canonical_text.encode("utf-8")) + 256 + 64 * (len(messages) + len(tools))


def prepared_chat_envelope(request: PreparedModelRequest) -> JsonSnapshot:
    """重验原请求指纹依据，拒绝最终封装超过已预留输入，不改v1请求字段/hash。"""
    request.validate_binding()
    envelope = chat_envelope(
        request.route.model,
        request.messages,
        request.tools,
        request.max_output_tokens,
        request.parameters,
    )
    if chat_envelope_tokens(envelope) > request.budget.input_tokens:
        raise ValueError("实际请求封装超出已计数输入")
    return envelope
