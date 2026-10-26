"""不截断工具事务组；计数包含完整供应商请求封装。"""

from typing import Any

from harness.contracts.model_envelope import chat_envelope, chat_envelope_tokens
from harness.contracts.models import ModelParameters
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


def require_complete_tool_groups(messages: list[dict[str, Any]]) -> None:
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise RuntimeFault("TOOL_TRANSACTION_GROUP_INVALID", 422)
            pending.remove(call_id)
        else:
            if pending:
                raise RuntimeFault("TOOL_TRANSACTION_GROUP_INCOMPLETE", 422)
            calls = message.get("tool_calls", [])
            for call in calls:
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    raise RuntimeFault("TOOL_CALL_ID_INVALID", 422)
                pending.add(call_id)
                seen.add(call_id)
    if pending:
        raise RuntimeFault("TOOL_TRANSACTION_GROUP_INCOMPLETE", 422)


def envelope_tokens(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_output: int,
    parameters: ModelParameters | None = None,
) -> int:
    # 此处是保守开发估计，非供应商精确Tokenizer。未验证模型能力时仍不能派发。
    return chat_envelope_tokens(
        chat_envelope(
            model,
            JsonSnapshot.capture(messages),
            JsonSnapshot.capture(tools),
            max_output,
            parameters,
        )
    )
