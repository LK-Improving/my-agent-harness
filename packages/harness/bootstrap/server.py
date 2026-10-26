"""从工作区根目录的显式配置装配 API；不自动迁移或启动 Worker。"""

import os
from pathlib import Path

from dotenv import dotenv_values
from fastapi import FastAPI
from pydantic import Field, SecretStr

from harness.adapters.development_auth import DevelopmentAuthenticator, LiveDevelopmentAuthenticator
from harness.bootstrap.api import create_app
from harness.contracts.runtime import AdmissionProfile
from harness.contracts.specs import Contract


class ServerSettings(Contract):
    database_dsn: SecretStr
    environment: str = "development"
    auth_mode: str = "development"
    identity_file: Path
    admission: AdmissionProfile = Field(default_factory=AdmissionProfile)
    enable_write_sandbox: bool = False


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
    return ServerSettings(
        database_dsn=SecretStr(dsn),
        environment=values.get("HARNESS_ENVIRONMENT", "development"),
        auth_mode=values.get("HARNESS_AUTH_MODE", "development"),
        identity_file=(root / identity).resolve(),
        admission=admission,
        enable_write_sandbox=values.get("HARNESS_ENABLE_WRITE_SANDBOX", "false").lower() == "true",
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
    return create_app(
        assembly.store,
        auth.authenticate,
        writes=assembly.store,
        reconcile=assembly.writes.reconcile if assembly.writes else None,
        model_ready=lambda: assembly.model_ready,
    )
