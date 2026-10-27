"""规范化业务输入并生成稳定身份；不对消息做截断或空白清理。"""

import hashlib
import json
import unicodedata
from typing import cast

from pydantic import Field, JsonValue

from harness.contracts.specs import Contract, Identifier


def _normalize(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in normalized:
                raise ValueError("规范化后出现重复 JSON 键")
            normalized[canonical_key] = _normalize(item)
        return normalized
    return value


def canonical_json(value: object) -> str:
    """v1 使用 NFC、排序键、紧凑 JSON；非有限数值禁止参与身份。"""
    return json.dumps(
        _normalize(cast(JsonValue, value)),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def checkpoint_thread_id(tenant_id: str, application_id: str, run_id: str) -> str:
    """执行和清理共用固定线程身份，不接受外部checkpoint路径。"""
    return "harness-" + canonical_hash({"tenant": tenant_id, "application": application_id, "run": run_id})


class RunRequest(Contract):
    request_id: Identifier
    message: str = Field(min_length=1, max_length=16000)
    expected_revision: int = Field(ge=0, strict=True)
    attachment_versions: tuple[str, ...] = ()
    business_options: dict[str, JsonValue] = Field(default_factory=dict)

    def business_hash(self) -> str:
        return canonical_hash(
            {
                "message": self.message,
                "attachment_versions": list(self.attachment_versions),
                "business_options": self.business_options,
            }
        )
