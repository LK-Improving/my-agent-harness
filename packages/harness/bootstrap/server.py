"""从工作区根目录的显式配置装配 API；不自动迁移或启动 Worker。"""

import os
from pathlib import Path
from typing import Literal, cast

from dotenv import dotenv_values
from fastapi import FastAPI
from pydantic import Field, SecretStr

from harness.adapters.development_auth import DevelopmentAuthenticator, LiveDevelopmentAuthenticator
from harness.adapters.postgres.activation import PostgresReleaseActivation
from harness.bootstrap.api import create_app
from harness.bootstrap.observability import langfuse_sink
from harness.contracts.runtime import AdmissionProfile
from harness.contracts.retention import RetentionPolicy
from harness.contracts.specs import Contract


class ServerSettings(Contract):
    agent_profile: Literal["operations", "travel", "judge"] = "operations"
    rag_enabled: bool = False
    database_dsn: SecretStr
    environment: str = "development"
    auth_mode: str = "development"
    identity_file: Path
    admission: AdmissionProfile = Field(default_factory=AdmissionProfile)
    enable_write_sandbox: bool = False
    retention: RetentionPolicy = Field(default_factory=RetentionPolicy)
    activation_directory: Path | None = None
    judge_freeze_directory: Path | None = None


def load_server_settings(root: Path) -> ServerSettings:
    values = {
        key: value for key, value in dotenv_values(root / ".env").items() if value is not None
    }
    values.update(os.environ)
    dsn = values.get("HARNESS_DATABASE_DSN", "")
    identity = values.get("HARNESS_DEV_IDENTITY_FILE", "")
    if not dsn or not identity:
        raise ValueError("请在工作区 .env 设置 HARNESS_DATABASE_DSN 和 HARNESS_DEV_IDENTITY_FILE")
    profile = values.get("HARNESS_ADMISSION_PROFILE_FILE")
    admission = AdmissionProfile()
    if profile:
        admission = AdmissionProfile.model_validate_json(
            (root / profile).read_text(encoding="utf-8")
        )
    selected = values.get("HARNESS_AGENT_PROFILE", "operations")
    retention = RetentionPolicy()
    retention_ref = values.get("HARNESS_RETENTION_PROFILE_FILE")
    activation_ref = values.get("HARNESS_ACTIVATION_DIRECTORY")
    judge_freeze_ref = values.get("HARNESS_JUDGE_FREEZE_DIRECTORY") if selected == "judge" else None
    if retention_ref:
        retention = RetentionPolicy.model_validate_json((root / retention_ref).read_text("utf-8"))
    if selected not in {"operations", "travel", "judge"}:
        raise ValueError("HARNESS_AGENT_PROFILE仅支持operations、travel或judge")
    judge_freeze = (root / judge_freeze_ref).resolve() if judge_freeze_ref else None
    if judge_freeze is not None and not judge_freeze.is_relative_to(root.resolve()):
        raise ValueError("Judge 冻结包必须位于当前工作区")
    return ServerSettings(
        agent_profile=cast(Literal["operations", "travel", "judge"], selected),
        rag_enabled=values.get("HARNESS_RAG_ENABLED", "false").lower() == "true",
        database_dsn=SecretStr(dsn),
        environment=values.get("HARNESS_ENVIRONMENT", "development"),
        auth_mode=values.get("HARNESS_AUTH_MODE", "development"),
        identity_file=(root / identity).resolve(),
        admission=admission,
        enable_write_sandbox=values.get("HARNESS_ENABLE_WRITE_SANDBOX", "false").lower() == "true",
        retention=retention,
        activation_directory=(root / activation_ref).resolve() if activation_ref else None,
        judge_freeze_directory=judge_freeze,
    )


def application() -> FastAPI:
    root = Path.cwd().resolve()
    settings = load_server_settings(root)
    if settings.auth_mode != "development":
        raise ValueError("正式身份适配器尚未安装；拒绝回退开发认证")
    auth = LiveDevelopmentAuthenticator(settings.identity_file, settings.environment)
    DevelopmentAuthenticator(auth.identities(), environment=settings.environment)
    from harness.bootstrap.assembly import DevelopmentAssembly

    assembly = DevelopmentAssembly(root, owner="api-no-execution", checkpointer=None)
    observations = langfuse_sink(root, settings.retention.external_trace_copies)
    if observations is not None:
        assembly.store.install_observation_sink(observations)
    activation = None
    if settings.activation_directory is not None:
        activation = PostgresReleaseActivation(
            assembly.store, settings.activation_directory, assembly.observe_execution_version, auth.current,
        )
    return create_app(
        assembly.store,
        auth.authenticate,
        writes=assembly.store,
        reconcile=assembly.writes.reconcile if assembly.writes else None,
        model_ready=lambda: assembly.model_ready,
        artifacts=assembly.artifacts,
        artifact_retention=assembly.artifact_retention,
        retention=assembly.retention,
        activate=activation.activate if activation else None,
        close_observations=observations.aclose if observations else None,
    )
