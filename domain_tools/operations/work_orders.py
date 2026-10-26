"""工单沙箱适配器；只允许本机显式配置的开发服务。"""

from typing import Any
from urllib.parse import urlparse

import httpx
from harness.contracts.runtime import Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ToolSpec
from harness.contracts.writes import (
    DispatchPermit,
    ExternalWriteResult,
    ReconciliationPermit,
    ReconciliationResult,
)
from harness.tools.writes import RegisteredWrite
from pydantic import Field, SecretStr


class CreateWorkOrderInput(Contract):
    order_id: str = Field(pattern=r"^synthetic-[A-Za-z0-9_-]{1,40}$")


class WorkOrderOutput(Contract):
    work_order_id: str = Field(pattern=r"^wo_[a-f0-9]{32}$")
    order_id: str = Field(pattern=r"^synthetic-[A-Za-z0-9_-]{1,40}$")


class WorkOrderSandboxHandler:
    def __init__(self, base_url: str, token: SecretStr, fault: str | None = None) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("运营沙箱仅允许本机HTTP源地址")
        if len(token.get_secret_value()) < 32:
            raise ValueError("沙箱需要至少32字符的独立访问Token")
        if fault not in {None, "before_write", "after_write", "hidden", "invalid_output"}:
            raise ValueError("未知沙箱故障模式")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.fault = fault

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            transport=httpx.AsyncHTTPTransport(retries=0),
            headers={"Authorization": "Bearer " + self.token.get_secret_value()},
        )

    async def invoke(self, permit: DispatchPermit) -> ExternalWriteResult:
        operation = permit.operation
        body = {
            "tenant_id": operation.principal.tenant_id,
            "application_id": operation.principal.application_id,
            "arguments": operation.arguments.as_object(),
        }
        headers = {"X-Sandbox-Fault": self.fault} if self.fault else {}
        async with self._client() as client:
            response = await client.post(
                f"/writes/{operation.operation_id}", json=body, headers=headers
            )
        if response.status_code != 200:
            raise RuntimeFault("SANDBOX_WRITE_UNCONFIRMED", 503)
        return ExternalWriteResult.model_validate(response.json())

    async def reconcile(self, permit: ReconciliationPermit) -> ReconciliationResult:
        operation = permit.operation
        async with self._client() as client:
            response = await client.get(
                f"/writes/{operation.operation_id}",
                params={
                    "tenant_id": operation.principal.tenant_id,
                    "application_id": operation.principal.application_id,
                },
            )
        if response.status_code != 200:
            # 包括暂不可见/404，绝不把缺少记录当作没有写入。
            return ReconciliationResult(
                operation_id=operation.operation_id,
                args_hash=operation.arguments.content_hash,
                conclusion="still_unknown",
                evidence_ref="sandbox-query-unconfirmed",
            )
        return ReconciliationResult.model_validate(response.json())


def registered_work_order(handler: WorkOrderSandboxHandler) -> RegisteredWrite:
    spec = ToolSpec(
        name="create_work_order",
        version="1",
        handler_ref="domain_tools.operations.create_work_order",
        effect="write",
        input_schema=CreateWorkOrderInput.model_json_schema(),
        output_schema=WorkOrderOutput.model_json_schema(),
        authorization_policy_ref="work-order-create-synthetic@1",
        approval_policy_ref="exact-operation@1",
        timeout_ms=10000,
        retry={"max_attempts_per_operation": 2, "strategy": "same_key_after_reconcile"},
        idempotency={"provider_supported": True, "retention_seconds": 604800},
        business_action_key_resolver_ref="operations.order-action@1",
        reconcile_handler_ref="domain_tools.operations.lookup_operation",
        cost={
            "billable": False,
            "upper_bound": "0",
            "currency": "CNY",
            "price_version": "sandbox@1",
        },
    )

    def authorize(actor: Principal, args: dict[str, Any]) -> bool:
        return "work_orders:write:synthetic" in actor.resource_scopes

    return RegisteredWrite(
        spec=JsonSnapshot.capture(spec.model_dump(mode="json")),
        input_type=CreateWorkOrderInput,
        output_type=WorkOrderOutput,
        handler=handler,
        authorize=authorize,
        action_identity=lambda args: "create-work-order:" + args["order_id"],
    )
