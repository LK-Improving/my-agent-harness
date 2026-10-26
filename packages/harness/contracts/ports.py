"""M0 所需最小端口，不依赖实现或供应商 SDK。"""

from typing import Protocol

from pydantic import JsonValue

from harness.contracts.events import ArtifactPage
from harness.contracts.specs import ExecutionContext
from harness.contracts.state import TaskState, ToolReceipt, ValidationReport


class ToolExecutor(Protocol):
    def execute(
        self,
        name: str,
        arguments: dict[str, JsonValue],
        logical_call_id: str,
        context: ExecutionContext,
    ) -> ToolReceipt: ...


class ArtifactReader(Protocol):
    def read(
        self,
        ref: str,
        offset: int,
        limit: int,
        context: ExecutionContext,
    ) -> ArtifactPage: ...


class CompletionValidator(Protocol):
    def validate(
        self,
        task: TaskState,
        candidate: JsonValue,
        context: ExecutionContext,
    ) -> ValidationReport: ...
