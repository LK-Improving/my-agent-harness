"""M0-03a 的边界反例；按用户要求仅编写，尚未运行。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harness.adapters.models.config import QWEN_ADAPTER_REF, QwenSettings
from harness.adapters.models.qwen import _http_failure
from harness.contracts.failures import ProviderErrorCode, ProviderFailure
from harness.contracts.guards import (
    GuardDecision,
    GuardedToolCall,
    GuardVerdict,
    compose_guard_decisions,
)
from harness.contracts.models import (
    ModelBudgetQuote,
    ModelParameters,
    PreparedModelRequest,
    ResolvedModelRoute,
    model_input_hash,
)
from harness.contracts.release import ReleaseManifest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec, ExecutionContext
from pydantic import ValidationError


def context() -> ExecutionContext:
    return ExecutionContext(
        tenant_id="tenant-a",
        subject_id="operator-a",
        application_id="app-a",
        run_id="run-1",
        release_id="release-1",
        trace_id="trace-1",
    )


def prepared() -> PreparedModelRequest:
    settings = QwenSettings()
    route = ResolvedModelRoute(
        provider="dashscope",
        model="qwen-plus",
        adapter_ref=QWEN_ADAPTER_REF,
        configuration_hash=settings.chat_configuration_hash(),
        capabilities=("tool_calling",),
        capabilities_verified=True,
        context_window_tokens=4096,
        max_output_tokens=128,
    )
    messages = JsonSnapshot.capture([{"role": "user", "content": "测试输入"}])
    tools = JsonSnapshot.capture([])
    parameters = ModelParameters()
    quote = ModelBudgetQuote(
        quote_id="quote-1",
        policy_ref="budget@1",
        measured_input_hash=model_input_hash(route, messages, tools, parameters, 128),
        input_tokens=100,
        output_reserved_tokens=128,
        remaining_total_tokens=1000,
        safety_margin_tokens=50,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    return PreparedModelRequest(
        context=context(),
        manifest_hash="a" * 64,
        environment_profile_hash="b" * 64,
        model_call_id="call-1",
        model_attempt_id="attempt-1",
        draft_id="draft-1",
        route=route,
        messages=messages,
        tools=tools,
        parameters=parameters,
        max_output_tokens=128,
        budget=quote,
    )


def test_snapshot_does_not_share_nested_containers() -> None:
    source = {"outer": [{"value": "原值"}]}
    snapshot = JsonSnapshot.capture(source)
    source["outer"][0]["value"] = "变更原输入"
    exported = snapshot.as_object()
    exported["outer"] = "变更读取副本"
    assert snapshot.as_object() == {"outer": [{"value": "原值"}]}
    with pytest.raises(ValidationError):
        JsonSnapshot(canonical_text=snapshot.canonical_text, content_hash="f" * 64)


def test_manifest_keeps_wire_shape_but_freezes_mapping_content() -> None:
    root = Path(__file__).resolve().parents[2]
    spec = AgentSpec.model_validate_json((root / "agents/operations.json").read_text("utf-8"))
    profile = {"provider": "controlled-stub", "options": {"retries": 0}}
    manifest = ReleaseManifest.model_validate(
        {
            "spec": spec,
            "source_revision": None,
            "source_tree_hash": "a" * 64,
            "dependency_lock_hash": "b" * 64,
            "prompt_hash": "c" * 64,
            "tool_schema_hashes": {
                f"{tool.name}@{tool.version}": "d" * 64 for tool in spec.allowed_tools
            },
            "model_profile": profile,
            "profiles": {},
            "environment_profile_hash": "e" * 64,
        }
    )
    before = manifest.content_hash()
    profile["options"]["retries"] = 99
    assert manifest.content_hash() == before
    assert isinstance(manifest.model_dump(mode="json")["model_profile"], dict)
    assert ReleaseManifest.model_validate_json(manifest.model_dump_json()).content_hash() == before


def test_request_binds_measurement_and_rejects_changed_input() -> None:
    request = prepared()
    request.require_dispatchable(datetime.now(UTC))
    modified = request.model_dump()
    modified["messages"] = JsonSnapshot.capture([{"role": "user", "content": "不同输入"}])
    with pytest.raises(ValidationError, match="计数依据"):
        PreparedModelRequest.model_validate(modified)


def test_unknown_capabilities_or_expired_quote_cannot_dispatch() -> None:
    request = prepared()
    with pytest.raises(ValueError, match="过期"):
        request.require_dispatchable(datetime.now(UTC) + timedelta(hours=1))
    unverified_route = request.route.model_copy(update={"capabilities_verified": False})
    quote = request.budget.model_copy(
        update={
            "measured_input_hash": model_input_hash(
                unverified_route,
                request.messages,
                request.tools,
                request.parameters,
                128,
            ),
        }
    )
    unverified = request.model_copy(update={"route": unverified_route, "budget": quote})
    with pytest.raises(ValueError, match="尚未确认"):
        unverified.require_dispatchable(datetime.now(UTC))


def test_request_and_returned_view_are_independent() -> None:
    request = prepared()
    before = request.fingerprint()
    view = request.messages.as_object_array()
    view[0]["content"] = "调用方改写副本"
    assert request.fingerprint() == before
    changed_route = request.route.model_copy(update={"model": "other-model"})
    changed = request.model_copy(update={"route": changed_route})
    with pytest.raises(ValueError, match="计数依据"):
        changed.require_dispatchable(datetime.now(UTC))


def test_guard_denial_is_monotonic_and_bound_to_arguments() -> None:
    call = GuardedToolCall(
        context=context(),
        operation_id="op-1",
        logical_call_id="tool-call-1",
        tool_name="create_work_order",
        tool_version="1",
        arguments=JsonSnapshot.capture({"order_id": "order-1"}),
    )
    deny = GuardDecision(
        guard_ref="scope@1",
        execution_fingerprint=call.fingerprint(),
        verdict=GuardVerdict.DENY,
        reason_code="scope_denied",
    )
    abstain = GuardDecision(
        guard_ref="extension@1",
        execution_fingerprint=call.fingerprint(),
        verdict=GuardVerdict.ABSTAIN,
    )
    assert compose_guard_decisions(call, (deny, abstain)).denied
    assert compose_guard_decisions(call, (abstain, deny)).denied
    other = call.model_copy(update={"arguments": JsonSnapshot.capture({"order_id": "order-2"})})
    with pytest.raises(ValidationError, match="同一固定工具调用"):
        compose_guard_decisions(other, (deny,))
    with pytest.raises(ValidationError):
        GuardDecision.model_validate({**deny.model_dump(), "verdict": "allow"})


def test_response_or_unknown_dispatch_cannot_be_marked_unbilled() -> None:
    with pytest.raises(ValidationError, match="未计费"):
        ProviderFailure(
            code=ProviderErrorCode.TIMEOUT, dispatch_status="unknown", billing_status="not_billable"
        )


@pytest.mark.parametrize(
    "status,code",
    [
        (401, ProviderErrorCode.AUTHENTICATION_FAILED),
        (403, ProviderErrorCode.ACCESS_DENIED),
        (429, ProviderErrorCode.RATE_LIMITED),
        (503, ProviderErrorCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_http_error_category_does_not_grant_retry_or_free_usage(status, code) -> None:
    failure = _http_failure(status, "request-id").failure
    assert failure.code == code
    assert failure.billing_status == "unknown"
    assert failure.dispatch_status == "response_received"
    assert failure.retry_hint == ("policy_required" if status in {429, 503} else "never")


def test_connection_hash_excludes_secret_but_binds_route_settings() -> None:
    base = QwenSettings()
    rotated = base.model_copy(update={"api_key": "不可写入指纹的值"})
    assert base.chat_configuration_hash() == rotated.chat_configuration_hash()
    changed = base.model_copy(update={"chat_model": "qwen-turbo"})
    assert base.chat_configuration_hash() != changed.chat_configuration_hash()
