"""提供商组合根：本项目显式文件、凭据引用与实际单次客户端装配。"""

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr, model_validator

from harness.adapters.models.config import QWEN_ADAPTER_REF
from harness.adapters.models.openai_chat_completions import (
    OpenAIChatCompletionsClient,
    installed_chat_protocol,
)
from harness.adapters.models.qwen import QwenClient
from harness.bootstrap.settings import load_qwen_settings
from harness.contracts.execution import ExecutionProfile
from harness.contracts.models import ModelGateway
from harness.contracts.providers import (
    ModelCapabilityEvidence,
    ModelProfile,
    ModelProviderConfiguration,
    ProviderConnection,
    ResolvedModelSelection,
)
from harness.contracts.release import BudgetProfile
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier
from harness.providers.catalog import builtin_registry
from harness.providers.registry import ProviderResolutionError
from harness.providers.resolver import resolve_model_profile


class ModelConfigurationError(ValueError):
    """稳定本地配置错误；不显示文件内容、密钥或Pydantic输入。"""


class EvidenceBundle(Contract):
    version: Literal["model-evidence@1"] = "model-evidence@1"
    evidence: tuple[ModelCapabilityEvidence, ...] = Field(default=(), max_length=256)
    revoked_refs: tuple[Identifier, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_refs(self) -> "EvidenceBundle":
        if len({item.ref for item in self.evidence}) != len(self.evidence):
            raise ValueError("能力证据引用不能重复")
        if len(set(self.revoked_refs)) != len(self.revoked_refs):
            raise ValueError("撤销引用不能重复")
        return self


class CredentialBinding(Contract):
    ref: Identifier
    provider_id: Literal["dashscope", "deepseek"]
    env_var: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def allowed_variable(self) -> "CredentialBinding":
        prefix = "DASHSCOPE" if self.provider_id == "dashscope" else "DEEPSEEK"
        if not re.fullmatch(prefix + r"(?:_[A-Z0-9]+)*_API_KEY", self.env_var):
            raise ValueError("凭据变量必须使用对应提供商的API_KEY前缀")
        return self


class CredentialBindings(Contract):
    version: Literal["model-credentials@1"] = "model-credentials@1"
    bindings: tuple[CredentialBinding, ...] = Field(max_length=64)


def model_values(root: Path) -> dict[str, str]:
    """仅根.env和进程环境；不搜索父目录、PI或其他项目凭据。"""
    values = {
        key: value for key, value in dotenv_values(root / ".env").items() if value is not None
    }
    values.update(os.environ)
    return values


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("配置JSON含重复键")
        result[key] = value
    return result


def _json_file(root: Path, ref: str, *, project_only: bool = True) -> dict[str, Any]:
    """只读本项目内显式路径；有界读取，符号链接不得逃逸工作区。"""
    try:
        path = (root / ref).resolve()
        if project_only and not path.is_relative_to(root.resolve()):
            raise ValueError("配置路径不属于本项目")
        with path.open("rb") as stream:
            raw = stream.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            raise ValueError("配置文件过大")
        value = json.loads(raw, object_pairs_hook=_pairs)
        if not isinstance(value, dict):
            raise ValueError("配置必须是JSON对象")
        return value
    except (ValueError, OSError):
        raise ModelConfigurationError("MODEL_CONFIGURATION_FILE_INVALID") from None


def _binding(
    root: Path, values: Mapping[str, str], connection: ProviderConnection
) -> CredentialBinding:
    defaults = (
        CredentialBinding(
            ref="dashscope-env@1", provider_id="dashscope", env_var="DASHSCOPE_API_KEY"
        ),
        CredentialBinding(ref="deepseek-env@1", provider_id="deepseek", env_var="DEEPSEEK_API_KEY"),
    )
    entries = list(defaults)
    if values.get("HARNESS_MODEL_CREDENTIALS_FILE"):
        entries.extend(
            CredentialBindings.model_validate(
                _json_file(root, values["HARNESS_MODEL_CREDENTIALS_FILE"])
            ).bindings
        )
    if len({entry.ref for entry in entries}) != len(entries):
        raise ModelConfigurationError("MODEL_CREDENTIAL_BINDING_DUPLICATE")
    matches = [entry for entry in entries if entry.ref == connection.credential_profile_ref]
    if len(matches) != 1 or matches[0].provider_id != connection.provider_id:
        raise ModelConfigurationError("MODEL_CREDENTIAL_BINDING_MISMATCH")
    return matches[0]


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    profile: ExecutionProfile
    transport: ModelGateway = field(repr=False)
    binding_snapshot: JsonSnapshot | None
    credential_ready: Callable[[], bool] = field(repr=False)

    def ready(self) -> bool:
        route, budget = self.profile.route, self.profile.budget
        return bool(
            route.capabilities_verified
            and "tool_calling" in route.capabilities
            and route.context_window_tokens is not None
            and budget.max_output_tokens <= route.max_output_tokens
            and budget.max_input_tokens
            + budget.max_output_tokens
            + self.profile.safety_margin_tokens
            <= route.context_window_tokens
            and self.credential_ready()
        )


def load_model_runtime(
    root: Path, *, model_profile_ref: str, environment_profile_hash: str, now: datetime
) -> ModelRuntime:
    """显式新配置失败即阻断；未启用时保留旧Qwen字段和hash路径。"""
    try:
        values = model_values(root)
        providers_ref = values.get("HARNESS_MODEL_PROVIDERS_FILE")
        execution_ref = (
            values.get("HARNESS_EXECUTION_PROFILE_FILE")
            or "examples/execution-profile.example.json"
        )
        # 旧执行profile保留显式外部路径语义；新提供商模式仅接受项目内配置。
        raw = _json_file(root, execution_ref, project_only=bool(providers_ref))
        original_execution = JsonSnapshot.capture(raw)
        if not providers_ref:
            settings = load_qwen_settings(root / ".env", include_retrieval=False)
            if not isinstance(raw.get("route"), dict):
                raise ModelConfigurationError("MODEL_EXECUTION_PROFILE_INVALID")
            raw["route"].update(
                {
                    "provider": "dashscope",
                    "model": settings.chat_model,
                    "adapter_ref": QWEN_ADAPTER_REF,
                    "configuration_hash": settings.chat_configuration_hash(),
                    "max_output_tokens": settings.max_output_tokens,
                }
            )
            legacy_profile = ExecutionProfile.model_validate(raw)

            def legacy_ready() -> bool:
                try:
                    current = model_values(root)
                    current_ref = (
                        current.get("HARNESS_EXECUTION_PROFILE_FILE")
                        or "examples/execution-profile.example.json"
                    )
                    actual = load_qwen_settings(root / ".env", include_retrieval=False)
                    return bool(
                        not current.get("HARNESS_MODEL_PROVIDERS_FILE")
                        and current_ref == execution_ref
                        and JsonSnapshot.capture(
                            _json_file(root, execution_ref, project_only=False)
                        )
                        == original_execution
                        and actual.chat_configuration_hash() == settings.chat_configuration_hash()
                        and actual.api_key == settings.api_key
                        and settings.api_key.get_secret_value()
                    )
                except (OSError, ValueError):
                    return False

            return ModelRuntime(legacy_profile, QwenClient(settings), None, legacy_ready)
        configuration = ModelProviderConfiguration.model_validate(_json_file(root, providers_ref))
        evidence_ref = values.get("HARNESS_MODEL_EVIDENCE_FILE")
        if not evidence_ref:
            raise ModelConfigurationError("MODEL_EVIDENCE_FILE_REQUIRED")
        bundle = EvidenceBundle.model_validate(_json_file(root, evidence_ref))
        selection = resolve_model_profile(
            builtin_registry(),
            configuration,
            model_profile_ref,
            installed=(installed_chat_protocol(),),
            evidence=bundle.evidence,
            budget=BudgetProfile.model_validate(raw["budget"]),
            environment_profile_hash=environment_profile_hash,
            now=now,
            safety_margin_tokens=raw.get("safety_margin_tokens", 512),
            revoked_evidence_refs=frozenset(bundle.revoked_refs),
        )
        selected = selection.snapshot.as_object()
        connection = ProviderConnection.model_validate(selected["connection"])
        profile = ModelProfile.model_validate(selected["profile"])
        proof = ModelCapabilityEvidence.model_validate(selected["evidence"])
        credential_binding = _binding(root, values, connection)
        # 非秘密账户变量绑定进入新路由身份；不更改旧Qwen hash。
        selected["credential_binding"] = credential_binding.model_dump(mode="json")
        snapshot = JsonSnapshot.capture(selected)
        selection = ResolvedModelSelection(
            route=selection.route.model_copy(update={"configuration_hash": snapshot.content_hash}),
            snapshot=snapshot,
        )
        raw["route"] = selection.route.model_dump(mode="json")
        execution = ExecutionProfile.model_validate(raw)

        def credential() -> SecretStr:
            current = model_values(root)
            if (
                current.get("HARNESS_MODEL_PROVIDERS_FILE") != providers_ref
                or current.get("HARNESS_MODEL_EVIDENCE_FILE") != evidence_ref
            ):
                raise ModelConfigurationError("MODEL_CONFIGURATION_CHANGED")
            current_execution_ref = (
                current.get("HARNESS_EXECUTION_PROFILE_FILE")
                or "examples/execution-profile.example.json"
            )
            if (
                current_execution_ref != execution_ref
                or JsonSnapshot.capture(_json_file(root, execution_ref)) != original_execution
            ):
                raise ModelConfigurationError("MODEL_EXECUTION_PROFILE_CHANGED")
            active = ModelProviderConfiguration.model_validate(_json_file(root, providers_ref))
            current_profiles = {item.ref: item for item in active.profiles}
            current_connections = {item.ref: item for item in active.connections}
            if (
                current_profiles.get(profile.ref) != profile
                or current_connections.get(connection.ref) != connection
            ):
                raise ModelConfigurationError("MODEL_CONFIGURATION_CHANGED")
            actual = EvidenceBundle.model_validate(_json_file(root, evidence_ref))
            if (
                proof not in actual.evidence
                or proof.ref in actual.revoked_refs
                or not proof.valid_at(datetime.now(UTC))
            ):
                raise ModelConfigurationError("MODEL_CAPABILITY_REVOKED_OR_EXPIRED")
            if _binding(root, current, connection) != credential_binding:
                raise ModelConfigurationError("MODEL_CREDENTIAL_BINDING_CHANGED")
            return SecretStr(current.get(credential_binding.env_var, ""))

        def ready() -> bool:
            try:
                return bool(credential().get_secret_value())
            except (ValueError, OSError):
                return False

        return ModelRuntime(
            execution, OpenAIChatCompletionsClient(selection, credential), selection.snapshot, ready
        )
    except ModelConfigurationError:
        raise
    except ProviderResolutionError as exc:
        raise ModelConfigurationError(exc.code) from None
    except (ValueError, KeyError, TypeError, OSError):
        raise ModelConfigurationError("MODEL_CONFIGURATION_INVALID") from None
