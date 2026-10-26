"""显式开发身份映射；客户端不能指定租户、角色或资源范围。"""

import hashlib
import hmac
import json
from collections.abc import Mapping
from pathlib import Path

from harness.contracts.runtime import Principal, RuntimeFault


class DevelopmentAuthenticator:
    def __init__(self, identities: Mapping[str, Principal], *, environment: str) -> None:
        if environment != "development" or not identities:
            raise ValueError("开发认证仅允许显式开发环境与非空服务端身份配置")
        if any(len(token) < 32 for token in identities):
            raise ValueError("开发 Bearer Token 至少需要 32 个字符")
        if any(token.startswith("REPLACE_") for token in identities):
            raise ValueError("身份模板中的占位 Token 必须替换为随机值")
        self._identities = tuple(
            (hashlib.sha256(token.encode()).digest(), actor) for token, actor in identities.items()
        )

    def authenticate(self, authorization: str | None) -> Principal:
        if authorization is None or not authorization.startswith("Bearer "):
            raise RuntimeFault("UNAUTHENTICATED", 401)
        token = authorization[7:]
        if len(token) > 1024:
            raise RuntimeFault("UNAUTHENTICATED", 401)
        digest = hashlib.sha256(token.encode()).digest()
        for expected, actor in self._identities:
            if hmac.compare_digest(digest, expected):
                return actor
        raise RuntimeFault("UNAUTHENTICATED", 401)


class LiveDevelopmentAuthenticator:
    """每次读取本地身份文件；同主体多Token时权限取交集，避免旧Token扩大权限。"""

    def __init__(self, path: Path, environment: str) -> None:
        self.path, self.environment = path, environment

    def identities(self) -> dict[str, Principal]:
        try:
            if self.environment != "development" or self.path.stat().st_size > 262144:
                raise RuntimeFault("IDENTITY_CONFIGURATION_UNAVAILABLE", 503)
            return {
                token: Principal.model_validate(value)
                for token, value in json.loads(self.path.read_text("utf-8")).items()
            }
        except (OSError, ValueError, AttributeError):
            raise RuntimeFault("IDENTITY_CONFIGURATION_UNAVAILABLE", 503) from None

    def authenticate(self, authorization: str | None) -> Principal:
        actor = DevelopmentAuthenticator(
            self.identities(), environment=self.environment
        ).authenticate(authorization)
        return self.current(actor)

    def current(self, identity: Principal) -> Principal:
        matches = [
            item
            for item in self.identities().values()
            if (item.tenant_id, item.subject_id, item.application_id)
            == (identity.tenant_id, identity.subject_id, identity.application_id)
        ]
        if not matches:
            raise RuntimeFault("IDENTITY_REVOKED", 403)
        roles, scopes = matches[0].roles, matches[0].resource_scopes
        for item in matches[1:]:
            roles, scopes = roles & item.roles, scopes & item.resource_scopes
        return identity.model_copy(update={"roles": roles, "resource_scopes": scopes})
