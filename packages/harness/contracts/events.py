"""公开事件、回答草稿与授权分页读取的数据契约。"""

from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from harness.contracts.specs import Contract, Identifier


class RuntimeEvent(Contract):
    schema_version: Literal[1] = 1
    run_id: Identifier
    seq: int = Field(ge=1, strict=True)
    type: str
    timestamp: AwareDatetime
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        required: dict[str, tuple[str, ...]] = {
            "run.accepted": ("release_id", "manifest_hash", "environment_profile_hash"),
            "run.started": ("observed_manifest_hash", "execution_identity"),
            "answer.started": ("draft_id", "producer"),
            "answer.delta": ("draft_id", "model_attempt_id", "delta"),
            "answer.invalidated": ("draft_id", "reason"),
            "run.completed": ("committed_draft_id", "commit_id", "result_hash"),
        }
        for key in required.get(self.type, ()):
            if self.payload.get(key) is None:
                raise ValueError(f"事件 {self.type} 缺少 {key}")
        if self.type == "answer.started" and self.payload.get("producer") == "model":
            if not self.payload.get("model_call_id") or not self.payload.get("model_attempt_id"):
                raise ValueError("模型草稿必须包含调用与尝试身份")
        return self


class ArtifactPage(Contract):
    ref: Identifier
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_ref: Identifier
    offset: int = Field(ge=0, strict=True)
    next_offset: int = Field(ge=0, strict=True)
    has_more: bool
    content: str
    truncated: bool

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.next_offset != self.offset + len(self.content):
            raise ValueError("分页游标必须按实际 Unicode 字符数推进")
        return self
