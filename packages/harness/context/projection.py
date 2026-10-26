"""文本/工具上下文白名单投影；审计和框架元数据不能直接成为模型输入。"""

import json
from typing import Any

from harness.context.requests import require_complete_tool_groups
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot

_METADATA = {"id", "additional_kwargs", "response_metadata", "usage_metadata", "type"}


def project_model_messages(messages: list[dict[str, Any]]) -> JsonSnapshot:
    """只接受已授权上下文，复制允许字段；未知角色/内容阻断，不静默丢消息。"""
    projected: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise RuntimeFault("MODEL_MESSAGE_ROLE_UNSUPPORTED", 422)
        allowed = {"role", "content", "name"}
        if role == "assistant":
            allowed.add("tool_calls")
        elif role == "tool":
            allowed.add("tool_call_id")
        if set(message) - allowed - _METADATA:
            raise RuntimeFault("MODEL_MESSAGE_FIELDS_UNSUPPORTED", 422)
        content = message.get("content", "")
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = ""
        if not isinstance(content, str):
            raise RuntimeFault("MODEL_MESSAGE_CONTENT_UNSUPPORTED", 422)
        value: dict[str, Any] = {"role": role, "content": content}
        if "name" in message:
            if not isinstance(message["name"], str) or not message["name"]:
                raise RuntimeFault("MODEL_MESSAGE_NAME_INVALID", 422)
            value["name"] = message["name"]
        if role == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise RuntimeFault("MODEL_TOOL_CALL_INVALID", 422)
            normalized = []
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"id", "type", "function"}
                    or call.get("type") != "function"
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                ):
                    raise RuntimeFault("MODEL_TOOL_CALL_INVALID", 422)
                function = call["function"]
                if (
                    not isinstance(function, dict)
                    or set(function) != {"name", "arguments"}
                    or not isinstance(function.get("name"), str)
                    or not function["name"]
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise RuntimeFault("MODEL_TOOL_CALL_INVALID", 422)
                try:
                    arguments = json.loads(function["arguments"])
                except (ValueError, TypeError):
                    raise RuntimeFault("MODEL_TOOL_ARGUMENTS_INVALID", 422) from None
                if not isinstance(arguments, dict):
                    raise RuntimeFault("MODEL_TOOL_ARGUMENTS_INVALID", 422)
                # 保留实际参数字符串，避免计数后再次编码改变输入。
                normalized.append(call)
            value["tool_calls"] = normalized
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeFault("TOOL_CALL_ID_INVALID", 422)
            value["tool_call_id"] = call_id
        projected.append(value)
    if not projected:
        raise RuntimeFault("MODEL_MESSAGES_EMPTY", 422)
    require_complete_tool_groups(projected)
    return JsonSnapshot.capture(projected)


def initial_context(history: JsonSnapshot, user_input: str) -> list[dict[str, Any]]:
    """只使用成功历史与当前输入，运行审计/审批事件不在本入口接收。"""
    return project_model_messages(
        [*history.as_object_array(), {"role": "user", "content": user_input}]
    ).as_object_array()
