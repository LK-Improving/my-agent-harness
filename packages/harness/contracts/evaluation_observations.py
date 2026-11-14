"""离线引入受信独立观测的输入契约。"""

from typing import Literal, Self

from pydantic import Field, model_validator

from harness.contracts.evaluation_gates import ExternalObservation, JudgeExecutionEvidence, JudgeReceipt
from harness.contracts.evaluation_metrics import Digest, MetricObservation
from harness.contracts.specs import Contract, Identifier


class CaseObservations(Contract):
    case_id: Identifier
    external_observation: ExternalObservation | None = None
    judge_receipt: JudgeReceipt | None = None
    judge_execution: JudgeExecutionEvidence | None = None
    metric_observation: MetricObservation | None = None

    @model_validator(mode="after")
    def require_observation(self) -> Self:
        if all(value is None for value in (
            self.external_observation, self.judge_receipt, self.judge_execution, self.metric_observation
        )):
            raise ValueError("每个用例至少提供一项独立观测")
        return self


class ObservationBundle(Contract):
    version: Literal["evaluation-observations@1"] = "evaluation-observations@1"
    freeze_hash: Digest
    manifest_hash: Digest
    cases: tuple[CaseObservations, ...] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("独立观测不能包含重复用例")
        return self
