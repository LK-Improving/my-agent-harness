"""版本化保留、不可变清理请求与副本传播端口。"""

from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, Field, model_validator

from harness.contracts.runtime import Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier

CleanupKind = Literal["session", "events", "artifact", "trace", "extensions"]


class RetentionPolicy(Contract):
    version: Literal["core-retention@1"] = "core-retention@1"
    session_seconds: int = Field(default=2592000, ge=1, strict=True)
    event_seconds: int = Field(default=604800, ge=1, strict=True)
    trace_seconds: int = Field(default=604800, ge=1, strict=True)
    private_seconds: int = Field(default=7776000, ge=1, strict=True)
    max_runs: int = Field(default=256, ge=1, le=1000, strict=True)
    max_jobs: int = Field(default=256, ge=1, le=1000, strict=True)
    max_attempts: int = Field(default=3, ge=1, le=3, strict=True)
    recheck_seconds: int = Field(default=300, ge=1, le=86400, strict=True)
    external_trace_copies: bool = False

    @model_validator(mode="after")
    def audit_window(self) -> Self:
        if self.private_seconds < max(self.session_seconds, self.event_seconds, self.trace_seconds):
            raise ValueError("私有审计窗口不能短于普通正文保留窗口")
        return self


class CleanupRequest(Contract):
    request_id: Identifier
    kind: CleanupKind
    reason: Literal["ttl", "delete"] = "ttl"
    expected_revision: int = Field(ge=0, strict=True)
    artifact_ref: str | None = Field(default=None, pattern=r"^artifact-[a-f0-9]{32}$")
    expected_content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def exact_target(self) -> Self:
        if (self.kind == "artifact") != (self.artifact_ref is not None):
            raise ValueError("产物清理必须且只能指定一个精确ref")
        if (self.kind == "artifact") != (self.expected_content_hash is not None):
            raise ValueError("产物清理必须绑定内容hash")
        if self.reason == "delete" and self.kind not in {"session", "artifact"}:
            raise ValueError("明确删除只接受会话或精确产物")
        return self


class CleanupView(Contract):
    job_id: Identifier
    session_id: Identifier
    kind: CleanupKind
    state: Literal["pending", "blocked", "running", "retry", "manual", "completed"]
    phase: Literal["copies", "payloads", "done"]
    attempts: int = Field(ge=0)
    reasons: tuple[str, ...] = ()
    next_check_at: AwareDatetime | None = None
    revoked_at: AwareDatetime | None = None
    request_hash: str


class CleanupCopies(Protocol):
    async def erase(self, target: JsonSnapshot) -> None: ...
    async def absent(self, target: JsonSnapshot) -> bool: ...


class RetentionManager(Protocol):
    async def request(self, actor: Principal, session_id: str, request: CleanupRequest) -> CleanupView: ...
    async def inspect(self, actor: Principal, job_id: str) -> CleanupView: ...
    async def advance(self, actor: Principal, job_id: str) -> CleanupView: ...
