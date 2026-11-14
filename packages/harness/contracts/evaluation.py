"""冻结评测资产与采集报告契约；不承担发布授权。"""

from collections.abc import AsyncIterator
from typing import Literal, Protocol, Self

from pydantic import Field, JsonValue, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.specs import Contract, Identifier
from harness.contracts.events import RuntimeEvent
from harness.contracts.hashing import RunRequest
from harness.contracts.runtime import EvaluationRequest, RunView, SessionRequest
from harness.contracts.release import EvaluationContext

class EvaluationAPI(Protocol):
    async def create_context(self, request: EvaluationRequest) -> EvaluationContext: ...

    async def create_session(self, request: SessionRequest) -> str: ...

    async def create_run(self, session_id: str, request: RunRequest) -> RunView: ...

    async def get_run(self, run_id: str) -> RunView: ...

    def events(self, run_id: str) -> AsyncIterator[RuntimeEvent]: ...


class CollectionFault(Exception):
    """可公开的采集错误码；不得携带响应正文或认证信息。"""


class EvaluationCase(Contract):
    case_id: Identifier
    message: str = Field(min_length=1, max_length=16000)
    attachment_versions: tuple[str, ...] = ()
    business_options: dict[str, JsonValue] = Field(default_factory=dict)
    expected: dict[str, JsonValue] = Field(min_length=1)

    def input_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"case_id", "expected"}))


class JudgePolicy(Contract):
    version: Identifier
    required: bool = True
    implementation: Identifier | None = None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_binding_ref: Identifier | None = None
    judge_manifest_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dimensions: tuple[Identifier, ...] = Field(min_length=1)
    missing_outcome: Literal["incomplete"] = "incomplete"


class MetricPolicy(Contract):
    version: Identifier
    metrics: tuple[Identifier, ...] = Field(min_length=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    thresholds_confirmed: bool = False


class GatePolicy(Contract):
    version: Identifier
    hard_failures: tuple[Identifier, ...] = Field(min_length=1)
    required_evidence: tuple[Identifier, ...] = Field(min_length=1)
    missing_outcome: Literal["incomplete"] = "incomplete"


class FrozenFile(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]*\.(json|jsonl)$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=1, le=16777216)


class EvaluationFreeze(Contract):
    version: Literal["evaluation-freeze@1"] = "evaluation-freeze@1"
    files: tuple[FrozenFile, ...] = Field(min_length=5)
    development_ids: tuple[Identifier, ...] = Field(min_length=1)
    holdout_ids: tuple[Identifier, ...] = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    holdout_for_tuning: Literal[False] = False

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        names = [item.name for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("冻结包文件重复")
        ids = self.development_ids + self.holdout_ids
        if len(ids) != len(set(ids)):
            raise ValueError("冻结包用例身份重复")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))
