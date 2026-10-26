"""M1 服务身份、准入策略与运行端口。"""

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import AwareDatetime, Field, JsonValue, field_serializer

from harness.contracts.hashing import RunRequest
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier, PositiveLimit
from harness.contracts.state import TaskState, ValidationReport

if TYPE_CHECKING:
    from harness.contracts.writes import SuspensionProof


class Principal(Contract):
    tenant_id: Identifier
    subject_id: Identifier
    application_id: Identifier
    roles: frozenset[str] = frozenset()
    resource_scopes: frozenset[str] = frozenset()

    @field_serializer("roles", "resource_scopes")
    def serialize_sets(self, value: frozenset[str]) -> list[str]:
        return sorted(value)


class AdmissionProfile(Contract):
    version: Identifier = "development-admission@1"
    max_queued_per_tenant: PositiveLimit = 20
    max_open_runs_per_tenant: PositiveLimit = 20
    requests_per_subject_minute: PositiveLimit = 60
    queue_timeout_seconds: PositiveLimit = 300
    execution_timeout_seconds: PositiveLimit = 120
    run_timeout_seconds: PositiveLimit = 900
    heartbeat_stale_seconds: PositiveLimit = 30
    event_page_size: int = Field(default=100, ge=1, le=500)
    reads_per_subject_minute: PositiveLimit = 120
    max_sse_per_subject: PositiveLimit = 2
    sse_lease_seconds: PositiveLimit = 30


class SessionRequest(Contract):
    agent_id: Identifier
    evaluation_context_id: Identifier | None = None


class CancelRequest(Contract):
    request_id: Identifier
    expected_run_revision: int = Field(ge=0, strict=True)


class EvaluationRequest(Contract):
    """仅评测管理身份可创建；主体与应用始终取服务端身份。"""

    candidate_release_id: Identifier
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    gate_policy_version: Identifier
    resource_scopes: frozenset[str]
    expires_at: AwareDatetime


class RunView(Contract):
    run_id: Identifier
    session_id: Identifier
    status: str
    revision: int
    release_id: Identifier
    manifest_hash: str
    environment_profile_hash: str
    cancel_requested: bool
    recovery_required: bool
    result: JsonValue = None
    error_code: str | None = None
    created_at: AwareDatetime
    model_steps: int = 0
    tool_dispatches: int = 0
    model_reserved_tokens: int = 0
    execution_used_ms: int = 0


class ClaimedRun(Contract):
    run_id: Identifier
    session_id: Identifier
    principal: Principal
    request: RunRequest
    manifest: ReleaseManifest
    execution_deadline: AwareDatetime
    history: JsonSnapshot
    suspension: JsonSnapshot | None = None


class AnswerCandidate(Contract):
    """候选与验证证据分开传递；存储层再次核对哈希。"""

    draft_id: Identifier
    model_call_id: Identifier
    model_attempt_id: Identifier
    answer: str = Field(min_length=1, max_length=64000)
    answer_kind: Literal["answer", "clarification", "abstention"]
    task_outcome: Literal["achieved", "needs_input", "not_achieved"]
    task: TaskState
    validation: ValidationReport

    def validation_input(self) -> dict[str, JsonValue]:
        return {
            "answer": self.answer,
            "answer_kind": self.answer_kind,
            "task_outcome": self.task_outcome,
            "task": self.task.model_dump(mode="json", exclude={"validation"}),
        }


class RuntimeFault(Exception):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


class RunEngine(Protocol):
    """装配必须对实际执行版本求哈希；不能直接返回输入 manifest。"""

    def observed_manifest_hash(self, release: ReleaseManifest) -> str: ...

    async def execute(self, run: ClaimedRun) -> AnswerCandidate: ...


class RuntimeStore(Protocol):
    async def admit_read(self, actor: Principal) -> None: ...

    async def acquire_sse(self, actor: Principal) -> str: ...

    async def renew_sse(self, actor: Principal, token: str) -> None: ...

    async def release_sse(self, token: str) -> None: ...
    async def register_release(self, actor: Principal, release: ReleaseManifest) -> str: ...

    async def create_evaluation(
        self, actor: Principal, request: EvaluationRequest
    ) -> EvaluationContext: ...

    async def create_session(self, actor: Principal, request: SessionRequest) -> str: ...

    async def create_run(
        self, actor: Principal, session_id: str, request: RunRequest
    ) -> RunView: ...

    async def get_run(self, actor: Principal, run_id: str) -> RunView: ...

    async def cancel_run(
        self, actor: Principal, run_id: str, request: CancelRequest
    ) -> RunView: ...

    async def events(
        self, actor: Principal, run_id: str, after: int
    ) -> list[dict[str, JsonValue]]: ...

    async def register_worker(self, identity: str) -> None: ...

    async def heartbeat(self, identity: str) -> None: ...

    async def worker_unready(self, identity: str) -> None: ...

    async def claim(self, identity: str) -> ClaimedRun | None: ...

    async def started(self, identity: str, run: ClaimedRun, observed_hash: str) -> None: ...

    async def continue_run(self, identity: str, run_id: str) -> bool: ...

    async def complete(
        self, identity: str, run: ClaimedRun, candidate: AnswerCandidate
    ) -> None: ...

    async def fail(self, identity: str, run_id: str, code: str) -> None: ...

    async def release_worker(self, identity: str) -> None: ...

    async def finish_slot(self, identity: str) -> None: ...

    async def park_write(self, owner: str, proof: "SuspensionProof") -> None: ...

    async def health(self) -> dict[str, bool | datetime | None]: ...
