"""M2R-01纯解析反例；证据为合成fixture，未运行、不证明真实供应商能力。"""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest
from harness.contracts.providers import (
    InstalledModelProtocol,
    ModelCapabilityEvidence,
    ModelCatalog,
    ModelProfile,
    ModelProviderConfiguration,
    ProviderConnection,
    ResolvedModelSelection,
)
from harness.contracts.release import BudgetProfile
from harness.providers.catalog import CHAT_ADAPTER, CHAT_PROTOCOL, builtin_registry
from harness.providers.registry import ProviderRegistry, ProviderResolutionError
from harness.providers.resolver import resolve_model_profile
from pydantic import ValidationError

NOW = datetime(2026, 10, 26, tzinfo=UTC)
ENVIRONMENT_HASH = "a" * 64


def fixture_values() -> tuple[
    ProviderRegistry,
    ModelProviderConfiguration,
    InstalledModelProtocol,
    ModelCapabilityEvidence,
    BudgetProfile,
]:
    registry = builtin_registry()
    connection = ProviderConnection(
        ref="dashscope-mainland@1",
        provider_id="dashscope",
        protocol_ref=CHAT_PROTOCOL,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        credential_profile_ref="dashscope-env@1",
    )
    profile = ModelProfile(
        ref="operations-chat@2",
        connection_ref=connection.ref,
        model_id="qwen-plus",
        required_capabilities=("tool_calling",),
        capability_evidence_ref="synthetic-proof@1",
        max_output_tokens=128,
    )
    configuration = ModelProviderConfiguration(connections=(connection,), profiles=(profile,))
    implementation = InstalledModelProtocol(
        protocol_ref=CHAT_PROTOCOL,
        adapter_ref=CHAT_ADAPTER,
        task_type="chat",
        counter_ref="utf8-envelope@1",
        envelope_ref="openai-chat-buffered@1",
    )
    proof = ModelCapabilityEvidence(
        ref="synthetic-proof@1",
        provider_id="dashscope",
        model_id="qwen-plus",
        task_type="chat",
        definition_hash=registry.get("dashscope").content_hash(),
        catalog_hash=registry.catalog("dashscope").content_hash(),
        connection_hash=connection.content_hash(),
        implementation=implementation,
        environment_profile_hash=ENVIRONMENT_HASH,
        verified_capabilities=("tool_calling",),
        context_window_tokens=4096,
        max_output_tokens=256,
        report_hash="b" * 64,
        verified_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
    )
    budget = BudgetProfile(
        version="test-budget@1",
        max_input_tokens=1000,
        max_output_tokens=128,
        max_total_tokens=3000,
        execution_budget_ms=120000,
    )
    return registry, configuration, implementation, proof, budget


def test_selected_route_is_frozen_and_uses_nonsecret_snapshot() -> None:
    registry, config, implementation, proof, budget = fixture_values()
    result = resolve_model_profile(
        registry,
        config,
        "operations-chat@2",
        installed=(implementation,),
        evidence=(proof,),
        budget=budget,
        environment_profile_hash=ENVIRONMENT_HASH,
        now=NOW,
        safety_margin_tokens=512,
    )
    assert result.route.provider == "dashscope"
    assert result.route.adapter_ref == CHAT_ADAPTER
    assert result.route.configuration_hash == result.snapshot.content_hash
    materialized = result.snapshot.as_object()
    materialized["connection"] = {"credential_profile_ref": "changed"}
    assert result.snapshot.as_object() != materialized
    assert "api_key" not in result.snapshot.canonical_text
    assert "Authorization" not in result.snapshot.canonical_text
    assert config.connections[0].base_url.endswith("/v1")
    with pytest.raises(ValidationError):
        ResolvedModelSelection(
            route=result.route.model_copy(update={"provider": "deepseek"}),
            snapshot=result.snapshot,
        )


def test_registry_is_frozen_and_rejects_duplicate_definitions() -> None:
    registry = builtin_registry()
    definition = registry.get("dashscope")
    catalog = registry.catalog("dashscope")
    with pytest.raises(ProviderResolutionError, match="PROVIDER_ALREADY_REGISTERED"):
        ProviderRegistry((definition, definition), (catalog,))
    with pytest.raises(ProviderResolutionError, match="MODEL_CATALOG_ALREADY_REGISTERED"):
        ProviderRegistry((definition,), (catalog, catalog))
    with pytest.raises(ProviderResolutionError, match="MODEL_CATALOG_MISSING"):
        ProviderRegistry((definition,), ())
    with pytest.raises(FrozenInstanceError):
        registry._definitions = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        registry._definitions["replacement"] = definition  # type: ignore[index]
    with pytest.raises(ProviderResolutionError, match="PROVIDER_NOT_REGISTERED"):
        registry.get("unknown")
    with pytest.raises(ProviderResolutionError, match="MODEL_NOT_REGISTERED"):
        registry.model("dashscope", "deepseek-chat", "chat")


@pytest.mark.parametrize(
    "address",
    [
        "http://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://user:secret@dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1?key=secret",
        "https://dashscope.aliyuncs.com/compatible-mode/v1#fragment",
        "https://dashscope.aliyuncs.com:444/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/%2e%2e/other",
        "https://dashscope.aliyuncs.com/compatible-mode/v1\n",
    ],
)
def test_connection_rejects_unsafe_endpoint_shapes(address: str) -> None:
    with pytest.raises(ValidationError):
        ProviderConnection(
            ref="connection@1",
            provider_id="dashscope",
            protocol_ref=CHAT_PROTOCOL,
            base_url=address,
            credential_profile_ref="credential@1",
        )


def test_configuration_rejects_duplicates_secrets_and_duplicate_model_entries() -> None:
    registry, config, _, _, _ = fixture_values()
    raw = config.model_dump(mode="json")
    raw["connections"][0]["api_key"] = "synthetic-secret"
    with pytest.raises(ValidationError):
        ModelProviderConfiguration.model_validate(raw)
    with pytest.raises(ValidationError):
        ModelProviderConfiguration(connections=config.connections * 2, profiles=config.profiles)
    with pytest.raises(ValidationError):
        ModelProviderConfiguration(connections=config.connections, profiles=config.profiles * 2)
    catalog = registry.catalog("dashscope")
    with pytest.raises(ValidationError):
        ModelCatalog(provider_id="dashscope", version="bad@1", models=catalog.models * 2)
    with pytest.raises(ValidationError):
        ModelCatalog(provider_id="deepseek", version="bad@1", models=catalog.models)


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        ("missing_profile", "MODEL_PROFILE_NOT_REGISTERED"),
        ("missing_connection", "PROVIDER_CONNECTION_NOT_REGISTERED"),
        ("unknown_provider", "PROVIDER_NOT_REGISTERED"),
        ("unknown_model", "MODEL_NOT_REGISTERED"),
        ("unknown_protocol", "PROTOCOL_NOT_SUPPORTED"),
        ("unsupported_task", "MODEL_TASK_NOT_SUPPORTED"),
        ("unlisted_endpoint", "PROVIDER_ENDPOINT_NOT_ALLOWED"),
        ("no_evidence", "MODEL_CAPABILITY_UNVERIFIED"),
        ("revoked_evidence", "MODEL_CAPABILITY_UNVERIFIED"),
        ("expired_evidence", "MODEL_EVIDENCE_EXPIRED_OR_FUTURE"),
        ("future_evidence", "MODEL_EVIDENCE_EXPIRED_OR_FUTURE"),
        ("no_implementation", "PROTOCOL_NOT_INSTALLED"),
        ("duplicate_implementation", "PROTOCOL_IMPLEMENTATION_DUPLICATE"),
        ("duplicate_evidence", "MODEL_EVIDENCE_DUPLICATE"),
        ("changed_connection", "MODEL_EVIDENCE_BINDING_MISMATCH"),
        ("changed_environment", "MODEL_EVIDENCE_BINDING_MISMATCH"),
        ("changed_catalog", "MODEL_EVIDENCE_BINDING_MISMATCH"),
        ("changed_definition", "MODEL_EVIDENCE_BINDING_MISMATCH"),
        ("changed_adapter", "MODEL_EVIDENCE_BINDING_MISMATCH"),
        ("changed_counter", "MODEL_ENVELOPE_NOT_SUPPORTED"),
        ("unverified_capability", "MODEL_CAPABILITY_UNVERIFIED"),
        ("undeclared_capability", "MODEL_CAPABILITY_UNVERIFIED"),
        ("output_mismatch", "MODEL_LIMITS_MISMATCH"),
        ("input_over_window", "MODEL_CONTEXT_BUDGET_EXCEEDED"),
        ("bad_safety_margin", "MODEL_SAFETY_MARGIN_INVALID"),
    ],
)
def test_resolution_fails_closed(scenario: str, code: str) -> None:
    registry, config, implementation, proof, budget = fixture_values()
    connection, profile = config.connections[0], config.profiles[0]
    selected = profile.ref
    instant = NOW
    installed = (implementation,)
    evidence = (proof,)
    environment = ENVIRONMENT_HASH
    revoked: frozenset[str] = frozenset()
    margin = 512
    if scenario == "missing_profile":
        selected = "missing@1"
    elif scenario == "missing_connection":
        profile = profile.model_copy(update={"connection_ref": "missing@1"})
    elif scenario == "unknown_provider":
        connection = connection.model_copy(update={"provider_id": "unknown"})
    elif scenario == "unknown_model":
        profile = profile.model_copy(update={"model_id": "unknown"})
    elif scenario == "unknown_protocol":
        connection = connection.model_copy(update={"protocol_ref": "responses@1"})
    elif scenario == "unsupported_task":
        profile = profile.model_copy(update={"task_type": "embedding"})
    elif scenario == "unlisted_endpoint":
        connection = connection.model_copy(update={"base_url": "https://127.0.0.1"})
    elif scenario == "no_evidence":
        evidence = ()
    elif scenario == "revoked_evidence":
        revoked = frozenset({proof.ref})
    elif scenario == "expired_evidence":
        instant = proof.expires_at
    elif scenario == "future_evidence":
        instant = proof.verified_at - timedelta(seconds=1)
    elif scenario == "no_implementation":
        installed = ()
    elif scenario == "duplicate_implementation":
        installed = (implementation, implementation)
    elif scenario == "duplicate_evidence":
        evidence = (proof, proof)
    elif scenario == "changed_connection":
        connection = connection.model_copy(update={"credential_profile_ref": "other-account@1"})
    elif scenario == "changed_environment":
        environment = "c" * 64
    elif scenario == "changed_catalog":
        evidence = (proof.model_copy(update={"catalog_hash": "c" * 64}),)
    elif scenario == "changed_definition":
        evidence = (proof.model_copy(update={"definition_hash": "c" * 64}),)
    elif scenario == "changed_adapter":
        changed = implementation.model_copy(update={"adapter_ref": "other-implementation@1"})
        installed = (changed,)
        evidence = (proof.model_copy(update={"implementation": changed}),)
    elif scenario == "changed_counter":
        changed = implementation.model_copy(update={"counter_ref": "other-counter@1"})
        installed = (changed,)
        evidence = (proof.model_copy(update={"implementation": changed}),)
    elif scenario == "unverified_capability":
        evidence = (proof.model_copy(update={"verified_capabilities": ()}),)
    elif scenario == "undeclared_capability":
        evidence = (proof.model_copy(update={"verified_capabilities": ("vision",)}),)
    elif scenario == "output_mismatch":
        profile = profile.model_copy(update={"max_output_tokens": 129})
    elif scenario == "input_over_window":
        budget = budget.model_copy(update={"max_input_tokens": 3500, "max_total_tokens": 5000})
    elif scenario == "bad_safety_margin":
        margin = -1
    config = ModelProviderConfiguration(connections=(connection,), profiles=(profile,))
    with pytest.raises(ProviderResolutionError, match=code):
        resolve_model_profile(
            registry,
            config,
            selected,
            installed=installed,
            evidence=evidence,
            budget=budget,
            environment_profile_hash=environment,
            now=instant,
            safety_margin_tokens=margin,
            revoked_evidence_refs=revoked,
        )


def test_unselected_connection_needs_no_credentials_and_cannot_borrow_proof() -> None:
    registry, config, implementation, proof, budget = fixture_values()
    original = config.connections[0]
    other = original.model_copy(update={"ref": "second-account@1", "credential_profile_ref": "b@1"})
    extra = ProviderConnection(
        ref="deepseek@1",
        provider_id="deepseek",
        protocol_ref=CHAT_PROTOCOL,
        base_url="https://api.deepseek.com",
        credential_profile_ref="deepseek-env@1",
    )
    config = ModelProviderConfiguration(
        connections=(*config.connections, other, extra), profiles=config.profiles
    )
    result = resolve_model_profile(
        registry,
        config,
        config.profiles[0].ref,
        installed=(implementation,),
        evidence=(proof,),
        budget=budget,
        environment_profile_hash=ENVIRONMENT_HASH,
        now=NOW,
        safety_margin_tokens=512,
    )
    assert result.route.provider == "dashscope"
    changed_profile = config.profiles[0].model_copy(update={"connection_ref": other.ref})
    changed = ModelProviderConfiguration(
        connections=config.connections, profiles=(changed_profile,)
    )
    with pytest.raises(ProviderResolutionError, match="MODEL_EVIDENCE_BINDING_MISMATCH"):
        resolve_model_profile(
            registry,
            changed,
            changed_profile.ref,
            installed=(implementation,),
            evidence=(proof,),
            budget=budget,
            environment_profile_hash=ENVIRONMENT_HASH,
            now=NOW,
            safety_margin_tokens=512,
        )
    other_proof = proof.model_copy(update={"connection_hash": other.content_hash()})
    other_result = resolve_model_profile(
        registry,
        changed,
        changed_profile.ref,
        installed=(implementation,),
        evidence=(other_proof,),
        budget=budget,
        environment_profile_hash=ENVIRONMENT_HASH,
        now=NOW,
        safety_margin_tokens=512,
    )
    assert other_result.route.configuration_hash != result.route.configuration_hash
