"""供应商连接参数；只定义类型，不读取环境或建立连接。"""

from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.specs import Contract, Identifier

QWEN_ADAPTER_REF = "dashscope-http@1"


class QwenSettings(Contract):
    api_key: SecretStr = SecretStr("")
    credential_profile_ref: Identifier = "dashscope-env@1"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = Field(default=1024, gt=0, le=2048)
    rerank_model: str = "gte-rerank-v2"
    rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_output_tokens: int = Field(default=128, gt=0, le=8192)

    def chat_configuration_hash(self) -> str:
        """固定实际聊天路由与非秘密参数；凭据仅绑定引用身份。"""
        return canonical_hash(
            {
                "provider": "dashscope",
                "adapter_ref": QWEN_ADAPTER_REF,
                "base_url": self.base_url.rstrip("/"),
                "model": self.chat_model,
                "credential_profile_ref": self.credential_profile_ref,
                "timeout_seconds": self.timeout_seconds,
                "max_output_tokens": self.max_output_tokens,
                "transport_retries": 0,
                "follow_redirects": False,
            }
        )

    @model_validator(mode="after")
    def validate_endpoint(self) -> "QwenSettings":
        allowed = {
            "dashscope.aliyuncs.com",
            "dashscope-intl.aliyuncs.com",
            "dashscope-us.aliyuncs.com",
        }
        for address in (self.base_url, self.rerank_url):
            parsed = urlparse(address)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in allowed
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.port not in {None, 443}
            ):
                raise ValueError("通义服务地址必须为显式允许的官方 HTTPS 地址")
        return self
