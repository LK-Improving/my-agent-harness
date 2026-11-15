"""分项报告使用的独立观测契约。"""

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from harness.contracts.specs import Contract, Identifier

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Count = Annotated[int, Field(ge=0, strict=True)]
CostCategory = Literal["model", "tool", "retrieval", "judge"]


class CostObservation(Contract):
    state: Literal["known", "unknown", "not_billable"]
    amount: Decimal | None = Field(default=None, ge=0, max_digits=24, decimal_places=12)
    currency: Literal["CNY", "USD"] | None = None
    price_version: Identifier | None = None
    # 模型和Judge填call_id，读工具填operation_id，写工具填attempt_id，检索填call_id。
    covered_call_ids: tuple[Identifier, ...] | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def check_amount(self) -> Self:
        if self.state == "known":
            if self.amount is None or self.currency is None or self.price_version is None:
                raise ValueError("已知费用必须包含金额、币种和价格版本")
        elif any(value is not None for value in (self.amount, self.currency, self.price_version)):
            raise ValueError("未知或不计费项不能填报金额")
        if self.covered_call_ids is not None and len(self.covered_call_ids) != len(set(self.covered_call_ids)):
            raise ValueError("费用覆盖调用身份不能重复")
        return self


class MemoryObservation(Contract):
    key_fact_retained: StrictBool | None = None
    erroneous_deletion: StrictBool | None = None
    correction_followed: StrictBool | None = None
    before_tokens: Count | None = None
    after_tokens: Count | None = None


class MetricObservation(Contract):
    version: Literal["evaluation-metric-observation@1"] = "evaluation-metric-observation@1"
    source_ref: Identifier
    case_id: Identifier
    run_id: Identifier
    freeze_hash: Digest
    manifest_hash: Digest
    result_hash: Digest | None
    event_seq: Count
    ranked_chunk_aliases: tuple[Identifier, ...] | None = Field(default=None, max_length=1000)
    memory: MemoryObservation | None = None
    costs: dict[CostCategory, CostObservation] = Field(default_factory=dict)
