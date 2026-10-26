"""有限静态候选目录；元数据声明不等于能力验证，不产生客户端。"""

from harness.contracts.providers import (
    ModelCatalog,
    ModelDescriptor,
    ProviderDefinition,
    ProviderProtocol,
)
from harness.providers.registry import ProviderRegistry

CHAT_PROTOCOL = "openai-chat-completions@1"
CHAT_ADAPTER = "openai-chat-completions-http@1"


def builtin_registry() -> ProviderRegistry:
    """仅登记当前重构目标；窗口/价格未知，无内置能力证据。"""
    definitions = (
        ProviderDefinition(
            provider_id="dashscope",
            version="dashscope-definition@1",
            display_name="通义千问（DashScope）",
            endpoint_policy_ref="dashscope-official-https@1",
            allowed_base_urls=("https://dashscope.aliyuncs.com/compatible-mode/v1",),
            compatibility_ref="dashscope-chat@1",
            protocols=(
                ProviderProtocol(
                    task_type="chat", protocol_ref=CHAT_PROTOCOL, adapter_refs=(CHAT_ADAPTER,)
                ),
            ),
        ),
        ProviderDefinition(
            provider_id="deepseek",
            version="deepseek-definition@1",
            display_name="DeepSeek",
            endpoint_policy_ref="deepseek-official-https@1",
            allowed_base_urls=("https://api.deepseek.com",),
            compatibility_ref="deepseek-chat@1",
            protocols=(
                ProviderProtocol(
                    task_type="chat", protocol_ref=CHAT_PROTOCOL, adapter_refs=(CHAT_ADAPTER,)
                ),
            ),
        ),
    )
    catalogs = tuple(
        ModelCatalog(
            version=f"{provider}-catalog@1",
            provider_id=provider,
            models=(
                ModelDescriptor(
                    provider_id=provider,
                    model_id=model,
                    task_type="chat",
                    declared_capabilities=("tool_calling",),
                ),
            ),
        )
        for provider, model in (("dashscope", "qwen-plus"), ("deepseek", "deepseek-chat"))
    )
    return ProviderRegistry(definitions, catalogs)
