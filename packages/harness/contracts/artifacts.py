"""不可变文本产物、统一字符分页及模型投影端口。"""

from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, Field, model_validator

from harness.contracts.runtime import ClaimedRun, Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier
from harness.contracts.state import ToolReceipt


class ArtifactProfile(Contract):
    version: Literal["pg-artifact@1"] = "pg-artifact@1"
    max_artifact_bytes: int = Field(default=1048576, gt=0, le=1048576, strict=True)
    max_run_bytes: int = Field(default=5242880, gt=0, le=5242880, strict=True)
    projection_chars: int = Field(default=2000, gt=0, le=4000, strict=True)
    page_chars: int = Field(default=4000, gt=0, le=4000, strict=True)
    page_bytes: int = Field(default=16000, ge=4, le=16000, strict=True)
    ttl_seconds: int = Field(default=604800, gt=0, le=604800, strict=True)

    @model_validator(mode="after")
    def consistent_limits(self) -> Self:
        if self.projection_chars > self.page_chars or self.max_artifact_bytes > self.max_run_bytes:
            raise ValueError("产物投影和单体容量不得超过分页和运行上限")
        return self


class ArtifactRead(Contract):
    artifact_ref: str = Field(pattern=r"^artifact-[a-f0-9]{32}$")
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=2000, ge=1, le=4000, strict=True)


class ArtifactPage(Contract):
    schema_version: Literal[2] = 2
    artifact_ref: str = Field(pattern=r"^artifact-[a-f0-9]{32}$")
    media_type: Literal["text/plain", "application/json"]
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_run_id: Identifier
    source_operation_id: Identifier
    total_chars: int = Field(ge=0, strict=True)
    total_bytes: int = Field(ge=0, strict=True)
    expires_at: AwareDatetime
    offset: int = Field(ge=0, strict=True)
    next_offset: int | None = Field(default=None, ge=0, strict=True)
    text: str = Field(max_length=4000)

    @model_validator(mode="after")
    def consistent_offsets(self) -> Self:
        end = self.offset + len(self.text)
        if end > self.total_chars or self.offset > self.total_chars:
            raise ValueError("分页不能超过正文字符边界")
        if self.next_offset != (end if end < self.total_chars else None):
            raise ValueError("分页游标必须按实际Unicode字符数推进")
        if self.next_offset is not None and not self.text:
            raise ValueError("非终页必须推进字符游标")
        return self


class ArtifactReader(Protocol):
    async def read(self, actor: Principal, session_id: str, request: ArtifactRead) -> ArtifactPage: ...


class ArtifactProjector(Protocol):
    async def project(self, run: ClaimedRun, owner: str, receipt: ToolReceipt) -> JsonSnapshot: ...


class ArtifactDeletionRequest(Contract):
    request_id: Identifier
    expected_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ArtifactRetentionPolicy(Contract):
    version: Literal["artifact-retention@1"] = "artifact-retention@1"
    recheck_seconds: int = Field(default=300, ge=1, le=86400, strict=True)


class ArtifactRetentionFacts(Contract):
    source_run_active: bool
    session_active: bool
    unresolved_write: bool
    pending_approval: bool
    active_evaluation: bool
    source_receipt_retained: bool
    readback_reference_retained: bool
    model_request_retained: bool
    session_history_retained: bool
    checkpoint_scope_retained: bool


class ArtifactRetentionView(Contract):
    artifact_ref: str = Field(pattern=r"^artifact-[a-f0-9]{32}$")
    policy_version: Literal["artifact-retention@1"] = "artifact-retention@1"
    state: Literal["retained", "cleanup_pending", "purged"]
    purged_at: AwareDatetime | None = None
    deletion_requested: bool
    revoked_at: AwareDatetime | None
    expires_at: AwareDatetime
    readable: bool
    read_denial_code: str | None
    retention_reasons: tuple[str, ...]
    checked_at: AwareDatetime
    next_check_at: AwareDatetime


class ArtifactRetention(Protocol):
    async def inspect(self, actor: Principal, session_id: str, artifact_ref: str) -> ArtifactRetentionView: ...

    async def request_deletion(self, actor: Principal, session_id: str, artifact_ref: str,
                               request: ArtifactDeletionRequest) -> ArtifactRetentionView: ...
