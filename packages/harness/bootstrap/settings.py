"""通义千问配置入口；允许空密钥保存，发起调用前必须显式校验。"""

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError

from harness.adapters.models.config import QwenSettings


def load_qwen_settings(env_file: Path, *, include_retrieval: bool = True) -> QwenSettings:
    """只读显式文件，环境变量优先；不搜索父目录或其他项目密钥。"""
    values = {key: value for key, value in dotenv_values(env_file).items() if value is not None}
    values.update(os.environ)
    mapping = {
        "DASHSCOPE_API_KEY": "api_key",
        "QWEN_CREDENTIAL_PROFILE_REF": "credential_profile_ref",
        "QWEN_BASE_URL": "base_url",
        "QWEN_CHAT_MODEL": "chat_model",
        "QWEN_EMBEDDING_MODEL": "embedding_model",
        "QWEN_EMBEDDING_DIMENSIONS": "embedding_dimensions",
        "QWEN_RERANK_MODEL": "rerank_model",
        "QWEN_RERANK_URL": "rerank_url",
        "QWEN_TIMEOUT_SECONDS": "timeout_seconds",
        "QWEN_MAX_OUTPUT_TOKENS": "max_output_tokens",
    }
    selected = {
        target: values[key]
        for key, target in mapping.items()
        if values.get(key)
        and (
            include_retrieval
            or not (key.startswith("QWEN_EMBEDDING") or key.startswith("QWEN_RERANK"))
        )
    }
    try:
        return QwenSettings.model_validate(selected)
    except ValidationError:
        # 不输出包含原始api_key输入的Pydantic异常正文。
        raise ValueError("通义配置未通过校验，请检查本地配置字段") from None
