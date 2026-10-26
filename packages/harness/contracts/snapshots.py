"""以不可变规范 JSON 文本保存深快照，不共享调用方嵌套字典或列表。"""

import json
from typing import Self

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.specs import Contract

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class JsonSnapshot(Contract):
    canonical_text: str = Field(repr=False)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @classmethod
    def capture(cls, value: object) -> Self:
        """复制并规范化；拒绝非 JSON 类型、循环引用和非有限数值。"""
        detached = _JSON_VALUE.validate_python(value, strict=True)
        return cls(canonical_text=canonical_json(detached), content_hash=canonical_hash(detached))

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        value = _JSON_VALUE.validate_json(self.canonical_text)
        if (
            canonical_json(value) != self.canonical_text
            or canonical_hash(value) != self.content_hash
        ):
            raise ValueError("JSON 快照内容、规范编码与指纹不一致")
        return self

    def materialize(self) -> JsonValue:
        """每次返回独立解析的内容；修改返回值不影响已固定快照。"""
        return _JSON_VALUE.validate_python(json.loads(self.canonical_text), strict=True)

    def as_object(self) -> dict[str, JsonValue]:
        value = self.materialize()
        if not isinstance(value, dict):
            raise ValueError("快照必须为 JSON 对象")
        return value

    def as_object_array(self) -> list[dict[str, JsonValue]]:
        value = self.materialize()
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("快照必须为 JSON 对象数组")
        return [item for item in value if isinstance(item, dict)]
