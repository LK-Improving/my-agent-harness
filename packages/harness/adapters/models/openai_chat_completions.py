"""固定路由的单次Completions客户端；不选模型、不fallback、不自行记账。"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import JsonValue, SecretStr

from harness.adapters.models.http import decode_completion, failure, post_json
from harness.contracts.failures import ProviderErrorCode
from harness.contracts.model_envelope import prepared_chat_envelope
from harness.contracts.models import ModelResult, PreparedModelRequest
from harness.contracts.providers import (
    InstalledModelProtocol,
    ModelCapabilityEvidence,
    ProviderConnection,
    ProviderDefinition,
    ResolvedModelSelection,
)
from harness.contracts.release import BudgetProfile
from harness.providers.catalog import CHAT_ADAPTER, CHAT_PROTOCOL


def installed_chat_protocol() -> InstalledModelProtocol:
    """声明当前代码提供的实现；依赖实际版本由组合根另与uv.lock核对。"""
    return InstalledModelProtocol(
        protocol_ref=CHAT_PROTOCOL,
        adapter_ref=CHAT_ADAPTER,
        task_type="chat",
        counter_ref="utf8-envelope@1",
        envelope_ref="openai-chat-buffered@1",
    )


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsClient:
    selection: ResolvedModelSelection
    credential: Callable[[], SecretStr] = field(repr=False)

    def __post_init__(self) -> None:
        selection = ResolvedModelSelection.model_validate_json(self.selection.model_dump_json())
        data = selection.snapshot.as_object()
        connection = ProviderConnection.model_validate(data["connection"])
        definition = ProviderDefinition.model_validate(data["definition"])
        proof = ModelCapabilityEvidence.model_validate(data["evidence"])
        if (
            connection.base_url not in definition.allowed_base_urls
            or connection.provider_id != definition.provider_id
            or proof.implementation != installed_chat_protocol()
        ):
            raise ValueError("模型连接与已安装协议不一致")
        object.__setattr__(self, "selection", selection)

    async def _post(
        self, body: dict[str, JsonValue], secret: SecretStr
    ) -> tuple[bytes, str | None]:
        connection = ProviderConnection.model_validate(
            self.selection.snapshot.as_object()["connection"]
        )
        return await post_json(
            connection.base_url + "/chat/completions",
            body,
            api_key=secret,
            timeout_seconds=connection.timeout_seconds,
        )

    async def complete(self, request: PreparedModelRequest) -> ModelResult:
        data = self.selection.snapshot.as_object()
        proof = ModelCapabilityEvidence.model_validate(data["evidence"])
        budget = BudgetProfile.model_validate(data["budget"])
        if (
            request.route != self.selection.route
            or request.environment_profile_hash != proof.environment_profile_hash
            or request.budget.policy_ref != budget.version
            or request.budget.safety_margin_tokens != data["safety_margin_tokens"]
            or request.budget.input_tokens > budget.max_input_tokens
            or request.budget.remaining_total_tokens > budget.max_total_tokens
            or request.max_output_tokens != budget.max_output_tokens
        ):
            raise failure(ProviderErrorCode.REQUEST_BINDING_MISMATCH)
        try:
            request.require_dispatchable(datetime.now(UTC))
            if not proof.valid_at(datetime.now(UTC)):
                raise ValueError("模型证据不在有效期")
            body = prepared_chat_envelope(request).as_object()
            secret = await asyncio.to_thread(self.credential)
            # 文件读取期间取消/到期也不能越过最后派发检查。
            request.require_dispatchable(datetime.now(UTC))
            if not proof.valid_at(datetime.now(UTC)):
                raise ValueError("模型证据已到期")
        except (ValueError, OSError):
            raise failure(ProviderErrorCode.REQUEST_NOT_DISPATCHABLE) from None
        if not secret.get_secret_value():
            raise failure(ProviderErrorCode.CONFIGURATION_MISSING)
        raw, reference = await self._post(body, secret)
        return decode_completion(raw, reference, request)
