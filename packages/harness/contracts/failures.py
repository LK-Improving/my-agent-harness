"""供应商安全错误分类；重试提示不代表已通过审批、预算和草稿守卫。"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from harness.contracts.specs import Contract


class ProviderErrorCode(StrEnum):
    CONFIGURATION_MISSING = "configuration_missing"
    AUTHENTICATION_FAILED = "authentication_failed"
    ACCESS_DENIED = "access_denied"
    INVALID_REQUEST = "invalid_request"
    ENDPOINT_NOT_FOUND = "endpoint_not_found"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    PROTOCOL_ERROR = "protocol_error"
    REQUEST_BINDING_MISMATCH = "request_binding_mismatch"
    REQUEST_NOT_DISPATCHABLE = "request_not_dispatchable"


class ProviderFailure(Contract):
    code: ProviderErrorCode
    http_status: int | None = Field(default=None, ge=100, le=599, strict=True)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    retry_hint: Literal["never", "policy_required"] = "never"
    dispatch_status: Literal["not_sent", "unknown", "response_received"]
    billing_status: Literal["not_billable", "unknown"]

    @model_validator(mode="after")
    def validate_uncertainty(self) -> Self:
        if self.dispatch_status != "not_sent" and self.billing_status == "not_billable":
            raise ValueError("可能已派发时不能无依据认定未计费")
        return self


class ProviderError(RuntimeError):
    """异常文本仅显示稳定错误码，结构化事实不接收响应正文或凭据。"""

    def __init__(self, failure: ProviderFailure) -> None:
        self.failure = failure
        super().__init__(failure.code.value)
