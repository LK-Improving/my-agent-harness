"""模型、只读工具与本机恢复的执行端口。"""

from typing import Literal, Protocol

from pydantic import Field

from harness.contracts.models import ModelResult, PreparedModelRequest, ResolvedModelRoute
from harness.contracts.release import BudgetProfile
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier
from harness.contracts.state import ToolReceipt


class ExecutionProfile(Contract):
    version: Identifier = "development-execution@1"
    route: ResolvedModelRoute
    budget: BudgetProfile
    counter_ref: Literal["utf8-envelope@1"] = "utf8-envelope@1"
    safety_margin_tokens: int = Field(default=512, ge=0)
    checkpoint_verification_timeout_seconds: int = Field(default=10, gt=0, le=60)


class ModelReservation(Contract):
    request: PreparedModelRequest
    result: ModelResult | None = None


class ReadReservation(Contract):
    operation_id: Identifier
    attempt_id: Identifier
    receipt: ToolReceipt | None = None


class ProcessIdentity(Contract):
    host_id: str
    boot_time: float
    pid: int = Field(gt=0)
    started_at: float
    namespace_id: str
    execution_class: Literal["local-no-subprocess@1"]
    boot_id: str
    start_marker: str


class TerminationProof(Contract):
    process: ProcessIdentity
    reason: str


class ExecutionLedger(Protocol):
    async def reserve_model(
        self,
        run: ClaimedRun,
        owner: str,
        messages: JsonSnapshot,
        tools: JsonSnapshot,
        profile: ExecutionProfile,
    ) -> ModelReservation: ...

    async def save_model_result(
        self, request: PreparedModelRequest, result: ModelResult
    ) -> None: ...

    async def fail_model(self, request: PreparedModelRequest, failure: JsonSnapshot) -> None: ...

    async def reserve_read(
        self,
        run: ClaimedRun,
        owner: str,
        logical_id: str,
        spec: JsonSnapshot,
        arguments: JsonSnapshot,
    ) -> ReadReservation: ...

    async def save_read(
        self, reservation: ReadReservation, data: JsonSnapshot | None, error: str | None
    ) -> ToolReceipt: ...

    async def operation_state(
        self, actor: Principal, operation_id: str
    ) -> tuple[JsonSnapshot, JsonSnapshot]: ...

    async def read_receipts(self, actor: Principal, run_id: str) -> tuple[ToolReceipt, ...]: ...

    async def candidate_for_run(self, actor: Principal, run_id: str) -> AnswerCandidate | None: ...
