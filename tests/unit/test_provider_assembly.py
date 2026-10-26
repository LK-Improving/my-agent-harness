"""M2R-03装配/单次协议反例定义；合成证据与控制传输，不执行真实请求。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harness.adapters.models.qwen import QwenClient
from harness.bootstrap.providers import ModelConfigurationError, load_model_runtime
from harness.contracts.execution import ExecutionProfile
from harness.contracts.failures import ProviderError
from harness.contracts.model_envelope import chat_envelope, chat_envelope_tokens
from harness.contracts.models import PreparedModelRequest, model_input_hash
from harness.contracts.providers import ModelProfile, ModelProviderConfiguration, ProviderConnection
from harness.providers.catalog import CHAT_PROTOCOL

from tests.unit.test_m0_boundaries import prepared
from tests.unit.test_provider_resolution import ENVIRONMENT_HASH, fixture_values


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), "utf-8")


def deployment(tmp_path: Path, monkeypatch, provider: str = "dashscope"):
    # 完全替换配置读取结果，宿主机已有环境密钥不会进入fixture。
    monkeypatch.setattr("harness.bootstrap.providers.os.environ", {})
    registry, _, _implementation, base_proof, budget = fixture_values()
    model = "qwen-plus" if provider == "dashscope" else "deepseek-chat"
    endpoint = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if provider == "dashscope"
        else "https://api.deepseek.com"
    )
    connection = ProviderConnection(
        ref=f"{provider}-main@1",
        provider_id=provider,
        protocol_ref=CHAT_PROTOCOL,
        base_url=endpoint,
        credential_profile_ref=f"{provider}-env@1",
    )
    profile = ModelProfile(
        ref="qwen-tool-model@1",
        connection_ref=connection.ref,
        model_id=model,
        required_capabilities=("tool_calling",),
        capability_evidence_ref="controlled-proof@1",
        max_output_tokens=128,
    )
    configuration = ModelProviderConfiguration(connections=(connection,), profiles=(profile,))
    instant = datetime.now(UTC)
    proof = base_proof.model_copy(
        update={
            "ref": "controlled-proof@1",
            "provider_id": provider,
            "model_id": model,
            "definition_hash": registry.get(provider).content_hash(),
            "catalog_hash": registry.catalog(provider).content_hash(),
            "connection_hash": connection.content_hash(),
            "verified_at": instant - timedelta(hours=1),
            "expires_at": instant + timedelta(hours=1),
        }
    )
    dummy = prepared().route.model_dump(mode="json")
    execution = ExecutionProfile.model_validate(
        {
            "route": dummy,
            "budget": budget.model_dump(mode="json"),
            "safety_margin_tokens": 512,
        }
    )
    write_json(tmp_path / "execution.json", execution.model_dump(mode="json"))
    write_json(tmp_path / "providers.json", configuration.model_dump(mode="json"))
    write_json(
        tmp_path / "evidence.json",
        {
            "version": "model-evidence@1",
            "evidence": [proof.model_dump(mode="json")],
            "revoked_refs": [],
        },
    )
    secret_var = "DASHSCOPE_API_KEY" if provider == "dashscope" else "DEEPSEEK_API_KEY"
    (tmp_path / ".env").write_text(
        "HARNESS_EXECUTION_PROFILE_FILE=execution.json\n"
        "HARNESS_MODEL_PROVIDERS_FILE=providers.json\n"
        "HARNESS_MODEL_EVIDENCE_FILE=evidence.json\n" + secret_var + "=controlled-key\n",
        "utf-8",
    )
    return configuration, proof


def load(root: Path):
    return load_model_runtime(
        root,
        model_profile_ref="qwen-tool-model@1",
        environment_profile_hash=ENVIRONMENT_HASH,
        now=datetime.now(UTC),
    )


def request_for(runtime) -> PreparedModelRequest:
    original = prepared()
    route = runtime.profile.route
    wire = chat_envelope(route.model, original.messages, original.tools, 128, original.parameters)
    quote = original.budget.model_copy(
        update={
            "policy_ref": runtime.profile.budget.version,
            "input_tokens": chat_envelope_tokens(wire),
            "remaining_total_tokens": 3000,
            "safety_margin_tokens": runtime.profile.safety_margin_tokens,
            "measured_input_hash": model_input_hash(
                route, original.messages, original.tools, original.parameters, 128
            ),
        }
    )
    return PreparedModelRequest.model_validate(
        {
            **original.model_dump(mode="python"),
            "route": route,
            "budget": quote,
            "environment_profile_hash": ENVIRONMENT_HASH,
        }
    )


@pytest.mark.parametrize("provider", ["dashscope", "deepseek"])
async def test_two_providers_share_one_protocol_without_cross_account_credentials(
    tmp_path, monkeypatch, provider
):
    deployment(tmp_path, monkeypatch, provider)
    runtime = load(tmp_path)
    captured = []

    async def controlled_post(url, body, *, api_key, timeout_seconds):
        captured.append((url, body, api_key.get_secret_value(), timeout_seconds))
        return json.dumps(
            {
                "model": body["model"],
                "choices": [{"message": {"content": "controlled"}, "finish_reason": "stop"}],
            }
        ).encode(), "controlled-reference"

    monkeypatch.setattr(
        "harness.adapters.models.openai_chat_completions.post_json", controlled_post
    )
    assert runtime.ready()
    result = await runtime.transport.complete(request_for(runtime))
    assert len(captured) == 1 and captured[0][2] == "controlled-key"
    assert captured[0][0].endswith("/chat/completions")
    assert (
        "dashscope.aliyuncs.com" if provider == "dashscope" else "api.deepseek.com"
    ) in captured[0][0]
    assert result.usage.input_tokens is None
    assert "controlled-key" not in runtime.binding_snapshot.canonical_text
    assert not isinstance(runtime.transport, QwenClient)


def test_empty_new_entry_retains_legacy_qwen_and_ignores_broken_optional_files(
    tmp_path, monkeypatch
):
    deployment(tmp_path, monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        env.read_text("utf-8").replace(
            "HARNESS_MODEL_PROVIDERS_FILE=providers.json", "HARNESS_MODEL_PROVIDERS_FILE="
        )
        + "QWEN_EMBEDDING_DIMENSIONS=not-an-integer\n",
        "utf-8",
    )
    (tmp_path / "evidence.json").write_text("invalid-json", "utf-8")
    runtime = load(tmp_path)
    assert isinstance(runtime.transport, QwenClient)
    assert runtime.profile.route.adapter_ref == "dashscope-http@1"
    assert (
        runtime.profile.route.configuration_hash
        == runtime.transport.settings.chat_configuration_hash()
    )
    assert runtime.binding_snapshot is None


@pytest.mark.parametrize("change", ["provider_mode", "model", "credential", "execution"])
def test_legacy_runtime_drift_cannot_silently_switch(tmp_path, monkeypatch, change):
    deployment(tmp_path, monkeypatch)
    env = tmp_path / ".env"
    env.write_text(
        env.read_text("utf-8").replace(
            "HARNESS_MODEL_PROVIDERS_FILE=providers.json", "HARNESS_MODEL_PROVIDERS_FILE="
        ),
        "utf-8",
    )
    runtime = load(tmp_path)
    assert runtime.credential_ready()
    original = runtime.profile.route
    if change == "execution":
        (tmp_path / "execution.json").write_text("{}", "utf-8")
    else:
        additions = {
            "provider_mode": "HARNESS_MODEL_PROVIDERS_FILE=providers.json",
            "model": "QWEN_CHAT_MODEL=qwen-other",
            "credential": "DASHSCOPE_API_KEY=",
        }
        env.write_text(env.read_text("utf-8") + additions[change] + "\n", "utf-8")
    assert not runtime.credential_ready()
    assert runtime.profile.route == original and isinstance(runtime.transport, QwenClient)


@pytest.mark.parametrize(
    "change",
    [
        "missing_evidence",
        "bad_json",
        "duplicate_keys",
        "unknown_model",
        "unsafe_endpoint",
        "wrong_environment",
        "revoked",
        "unknown_profile",
    ],
)
def test_explicit_invalid_configuration_never_falls_back(tmp_path, monkeypatch, change):
    configuration, proof = deployment(tmp_path, monkeypatch)
    if change == "missing_evidence":
        (tmp_path / "evidence.json").unlink()
    elif change == "bad_json":
        (tmp_path / "providers.json").write_text('{"secret":"controlled-key",', "utf-8")
    elif change == "duplicate_keys":
        (tmp_path / "providers.json").write_text('{"connections":[],"connections":[]}', "utf-8")
    elif change in {"unknown_model", "unsafe_endpoint", "unknown_profile"}:
        raw = configuration.model_dump(mode="json")
        if change == "unknown_model":
            raw["profiles"][0]["model_id"] = "not-installed"
        elif change == "unsafe_endpoint":
            raw["connections"][0]["base_url"] = "https://127.0.0.1"
        else:
            raw["profiles"][0]["ref"] = "other-profile@1"
        write_json(tmp_path / "providers.json", raw)
    else:
        changed = proof.model_dump(mode="json")
        if change == "wrong_environment":
            changed["environment_profile_hash"] = "f" * 64
        write_json(
            tmp_path / "evidence.json",
            {
                "evidence": [changed],
                "revoked_refs": [proof.ref] if change == "revoked" else [],
            },
        )
    with pytest.raises(ModelConfigurationError) as error:
        load(tmp_path)
    assert "controlled-key" not in str(error.value)


@pytest.mark.parametrize(
    "change", ["removed_key", "revoked", "changed_connection", "changed_budget", "changed_binding"]
)
async def test_dispatch_rechecks_configuration_and_revocation_without_sending(
    tmp_path, monkeypatch, change
):
    configuration, proof = deployment(tmp_path, monkeypatch)
    runtime = load(tmp_path)
    called = []

    async def forbidden_post(*args, **kwargs):
        called.append(True)
        raise AssertionError("不应跨越派发边界")

    monkeypatch.setattr("harness.adapters.models.openai_chat_completions.post_json", forbidden_post)
    if change == "removed_key":
        env = tmp_path / ".env"
        env.write_text(
            env.read_text("utf-8").replace(
                "DASHSCOPE_API_KEY=controlled-key", "DASHSCOPE_API_KEY="
            ),
            "utf-8",
        )
    elif change == "revoked":
        write_json(
            tmp_path / "evidence.json",
            {"evidence": [proof.model_dump(mode="json")], "revoked_refs": [proof.ref]},
        )
    elif change == "changed_connection":
        raw = configuration.model_dump(mode="json")
        raw["connections"][0]["timeout_seconds"] = 10
        write_json(tmp_path / "providers.json", raw)
    elif change == "changed_budget":
        path = tmp_path / "execution.json"
        raw = json.loads(path.read_text("utf-8"))
        raw["budget"]["max_total_tokens"] = 6000
        write_json(path, raw)
    else:
        env = tmp_path / ".env"
        env.write_text(
            env.read_text("utf-8").replace(
                "HARNESS_MODEL_PROVIDERS_FILE=providers.json", "HARNESS_MODEL_PROVIDERS_FILE="
            ),
            "utf-8",
        )
    assert not runtime.ready()
    with pytest.raises(ProviderError) as error:
        await runtime.transport.complete(request_for(runtime))
    assert error.value.failure.dispatch_status == "not_sent" and not called


async def test_network_failure_is_one_attempt_and_does_not_select_another_provider(
    tmp_path, monkeypatch
):
    deployment(tmp_path, monkeypatch, "deepseek")
    runtime = load(tmp_path)
    called = []

    async def failed_post(*args, **kwargs):
        from harness.adapters.models.http import http_failure

        called.append(True)
        raise http_failure(429, "controlled-reference")

    monkeypatch.setattr("harness.adapters.models.openai_chat_completions.post_json", failed_post)
    with pytest.raises(ProviderError) as error:
        await runtime.transport.complete(request_for(runtime))
    assert error.value.failure.http_status == 429 and len(called) == 1


def test_additional_account_binding_is_server_side_nonsecret_and_has_new_route_identity(
    tmp_path, monkeypatch
):
    configuration, proof = deployment(tmp_path, monkeypatch)
    first = load(tmp_path)
    raw = configuration.model_dump(mode="json")
    raw["connections"][0]["credential_profile_ref"] = "dashscope-secondary-env@1"
    changed = ModelProviderConfiguration.model_validate(raw)
    proof = proof.model_copy(update={"connection_hash": changed.connections[0].content_hash()})
    write_json(tmp_path / "providers.json", raw)
    write_json(tmp_path / "evidence.json", {"evidence": [proof.model_dump(mode="json")]})
    write_json(
        tmp_path / "credentials.json",
        {
            "bindings": [
                {
                    "ref": "dashscope-secondary-env@1",
                    "provider_id": "dashscope",
                    "env_var": "DASHSCOPE_SECONDARY_API_KEY",
                }
            ]
        },
    )
    env = tmp_path / ".env"
    env.write_text(
        env.read_text("utf-8") + "HARNESS_MODEL_CREDENTIALS_FILE=credentials.json\n"
        "DASHSCOPE_SECONDARY_API_KEY=second-controlled-key\n",
        "utf-8",
    )
    second = load(tmp_path)
    assert second.ready()
    assert second.profile.route.configuration_hash != first.profile.route.configuration_hash
    assert "second-controlled-key" not in second.binding_snapshot.canonical_text
    write_json(
        tmp_path / "credentials.json",
        {
            "bindings": [
                {
                    "ref": "dashscope-secondary-env@1",
                    "provider_id": "dashscope",
                    "env_var": "HARNESS_OPERATOR_TOKEN",
                }
            ]
        },
    )
    assert not second.ready()


async def test_http_transport_has_zero_retry_rejects_redirect_and_redacts_error(
    tmp_path, monkeypatch
):
    import httpx
    from harness.adapters.models.http import post_json
    from pydantic import SecretStr

    calls = []
    transports = []

    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://untrusted.invalid"})

    def transport_factory(*, retries):
        transports.append(retries)
        return httpx.MockTransport(handler)

    monkeypatch.setattr("harness.adapters.models.http.httpx.AsyncHTTPTransport", transport_factory)
    # 实际HTTP编解码经MockTransport；即使误开重定向也不能访问真实网络。
    with pytest.raises(ProviderError) as error:
        await post_json(
            "https://api.deepseek.com/chat/completions",
            {"model": "controlled"},
            api_key=SecretStr("controlled-key"),
            timeout_seconds=1,
        )
    assert transports == [0] and len(calls) == 1
    assert error.value.failure.http_status == 302
    assert "controlled-key" not in str(error.value)
