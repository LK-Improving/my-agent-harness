"""模型单次HTTP传输与Completions响应解码；无重试、跳转或正文日志。"""

import re

import httpx
from pydantic import BaseModel, Field, JsonValue, SecretStr, ValidationError

from harness.contracts.failures import ProviderError, ProviderErrorCode, ProviderFailure
from harness.contracts.models import ModelResult, ModelUsage, PreparedModelRequest
from harness.contracts.snapshots import JsonSnapshot


class ProviderUnavailable(ProviderError):
    """保留旧通义异常入口；异常文本只包含分类码。"""


def request_id(value: str | None) -> str | None:
    return value if value and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) else None


def failure(code: ProviderErrorCode, *, sent: bool = False) -> ProviderUnavailable:
    return ProviderUnavailable(
        ProviderFailure(
            code=code,
            dispatch_status="response_received" if sent else "not_sent",
            billing_status="unknown" if sent else "not_billable",
        )
    )


def protocol_failure(reference: str | None = None) -> ProviderUnavailable:
    return ProviderUnavailable(
        ProviderFailure(
            code=ProviderErrorCode.PROTOCOL_ERROR,
            request_id=request_id(reference),
            dispatch_status="response_received",
            billing_status="unknown",
        )
    )


def http_failure(status: int, reference: str | None) -> ProviderUnavailable:
    categories = {
        400: ProviderErrorCode.INVALID_REQUEST,
        401: ProviderErrorCode.AUTHENTICATION_FAILED,
        403: ProviderErrorCode.ACCESS_DENIED,
        404: ProviderErrorCode.ENDPOINT_NOT_FOUND,
        408: ProviderErrorCode.TIMEOUT,
        413: ProviderErrorCode.INVALID_REQUEST,
        422: ProviderErrorCode.INVALID_REQUEST,
        429: ProviderErrorCode.RATE_LIMITED,
        502: ProviderErrorCode.PROVIDER_UNAVAILABLE,
        503: ProviderErrorCode.PROVIDER_UNAVAILABLE,
        504: ProviderErrorCode.TIMEOUT,
    }
    return ProviderUnavailable(
        ProviderFailure(
            code=categories.get(status, ProviderErrorCode.PROVIDER_ERROR),
            http_status=status,
            request_id=request_id(reference),
            retry_hint="policy_required" if status in {408, 429, 500, 502, 503, 504} else "never",
            dispatch_status="response_received",
            billing_status="unknown",
        )
    )


async def post_json(
    url: str, body: dict[str, JsonValue], *, api_key: SecretStr, timeout_seconds: float
) -> tuple[bytes, str | None]:
    secret = api_key.get_secret_value()
    if not secret:
        raise failure(ProviderErrorCode.CONFIGURATION_MISSING)
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=httpx.AsyncHTTPTransport(retries=0),
        ) as client:
            async with client.stream(
                "POST", url, json=body, headers={"Authorization": f"Bearer {secret}"}
            ) as response:
                reference = request_id(response.headers.get("x-request-id"))
                if response.status_code != 200:
                    raise http_failure(response.status_code, reference)
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 2 * 1024 * 1024:
                        raise protocol_failure(reference)
                return bytes(raw), reference
    except (httpx.ConnectTimeout, httpx.PoolTimeout):
        raise ProviderUnavailable(
            ProviderFailure(
                code=ProviderErrorCode.TIMEOUT,
                retry_hint="policy_required",
                dispatch_status="not_sent",
                billing_status="not_billable",
            )
        ) from None
    except httpx.ConnectError:
        raise ProviderUnavailable(
            ProviderFailure(
                code=ProviderErrorCode.TRANSPORT_ERROR,
                retry_hint="policy_required",
                dispatch_status="not_sent",
                billing_status="not_billable",
            )
        ) from None
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise ProviderUnavailable(
            ProviderFailure(
                code=ProviderErrorCode.TIMEOUT
                if isinstance(exc, httpx.TimeoutException)
                else ProviderErrorCode.TRANSPORT_ERROR,
                retry_hint="policy_required",
                dispatch_status="unknown",
                billing_status="unknown",
            )
        ) from None


class _Usage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class _Message(BaseModel):
    content: str | None = None
    tool_calls: tuple[dict[str, JsonValue], ...] = ()


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


class _Completion(BaseModel):
    model: str
    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=1)
    usage: _Usage | None = None


def decode_completion(
    raw: bytes, reference: str | None, request: PreparedModelRequest
) -> ModelResult:
    try:
        result = _Completion.model_validate_json(raw)
    except ValidationError:
        raise protocol_failure(reference) from None
    usage, choice = result.usage, result.choices[0]
    return ModelResult(
        model=result.model,
        content=choice.message.content,
        tool_calls=JsonSnapshot.capture(list(choice.message.tool_calls)),
        finish_reason=choice.finish_reason,
        request_id=request_id(reference),
        usage=ModelUsage(
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        ),
        model_call_id=request.model_call_id,
        model_attempt_id=request.model_attempt_id,
        request_fingerprint=request.fingerprint(),
        draft_id=request.draft_id,
    )
