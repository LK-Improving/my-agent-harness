"""M1 身份边界和 HTTP 输入保护；不替代 PostgreSQL 集成证据。"""

import pytest
from harness.adapters.development_auth import DevelopmentAuthenticator
from harness.bootstrap.api import create_app
from harness.contracts.runtime import Principal, RuntimeFault
from httpx import ASGITransport, AsyncClient


def test_development_auth_never_accepts_client_identity_headers() -> None:
    actor = Principal(tenant_id="server-tenant", subject_id="server-user", application_id="app")
    auth = DevelopmentAuthenticator({"a" * 40: actor}, environment="development")
    assert auth.authenticate("Bearer " + "a" * 40) == actor
    for token in (None, "Bearer another-tenant", "Basic " + "a" * 40):
        with pytest.raises(RuntimeFault) as error:
            auth.authenticate(token)
        assert error.value.status == 401
    with pytest.raises(ValueError):
        DevelopmentAuthenticator({"a" * 40: actor}, environment="production")


@pytest.mark.asyncio
async def test_api_rejects_identity_override_without_echoing_input() -> None:
    actor = Principal(tenant_id="server", subject_id="u", application_id="app")
    auth = DevelopmentAuthenticator({"a" * 40: actor}, environment="development")
    # 请求必须在触达存储之前拒绝多余字段，因此本例不使用伪造数据库结论。
    app = create_app(None, auth.authenticate)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer " + "a" * 40},
            json={"agent_id": "ops", "tenant_id": "sensitive-input-marker"},
        )
        assert response.status_code == 422
        assert "sensitive-input-marker" not in response.text
        unauthorized = await client.post("/api/v1/sessions", json={"agent_id": "ops"})
        assert unauthorized.status_code == 401
        too_large = await client.post("/api/v1/sessions", content=b"x" * 262145)
        assert too_large.status_code == 413


@pytest.mark.asyncio
async def test_sse_cursor_validation_occurs_before_store_access() -> None:
    actor = Principal(tenant_id="t", subject_id="u", application_id="app")
    auth = DevelopmentAuthenticator({"a" * 40: actor}, environment="development")
    app = create_app(None, auth.authenticate)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for cursor in ("-1", "١", "9999999999999999999", "1.5"):
            # HTTP 头只允许 ASCII；Unicode 数字由服务方法的 isascii 边界另外覆盖。
            if not cursor.isascii():
                continue
            response = await client.get(
                "/api/v1/runs/r1/events",
                headers={
                    "Authorization": "Bearer " + "a" * 40,
                    "Last-Event-ID": cursor,
                },
            )
            assert response.status_code == 422
