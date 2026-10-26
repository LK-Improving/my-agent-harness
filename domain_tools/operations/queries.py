"""核心独立运行使用的显式合成数据查询，不冒充真实业务数据。"""

from typing import Any, Literal

from harness.contracts.runtime import Principal
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, ToolSpec
from harness.tools.reads import RegisteredRead
from pydantic import Field


class OrderQuery(Contract):
    order_id: str = Field(pattern=r"^synthetic-[A-Za-z0-9_-]{1,40}$")


class OrderResult(Contract):
    order_id: str
    status: Literal["pending", "not_found", "created"]
    source: Literal["synthetic-fixture", "write-sandbox"]
    work_order_id: str | None = None


def registered_order_query() -> RegisteredRead:
    async def query(actor: Principal, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "order_id": args["order_id"],
            "status": "pending" if args["order_id"] == "synthetic-demo" else "not_found",
            "source": "synthetic-fixture",
        }

    spec = ToolSpec(
        name="get_work_order",
        version="1",
        handler_ref="domain_tools.operations.synthetic_query@1",
        effect="read",
        input_schema=OrderQuery.model_json_schema(),
        output_schema=OrderResult.model_json_schema(),
        authorization_policy_ref="work-order-read-synthetic@1",
        timeout_ms=3000,
        cost={"billable": False, "upper_bound": "0"},
    )
    return RegisteredRead(
        spec=JsonSnapshot.capture(spec.model_dump(mode="json")),
        input_type=OrderQuery,
        output_type=OrderResult,
        handler=query,
        authorize=lambda actor, args: "work_orders:read:synthetic" in actor.resource_scopes,
    )


def registered_sandbox_query(handler: Any) -> RegisteredRead:
    async def query(actor: Principal, args: dict[str, Any]) -> dict[str, Any]:
        async with handler._client() as client:
            response = await client.get(
                "/orders/" + args["order_id"],
                params={"tenant_id": actor.tenant_id, "application_id": actor.application_id},
            )
        if response.status_code != 200:
            from harness.contracts.runtime import RuntimeFault

            raise RuntimeFault("SANDBOX_QUERY_UNAVAILABLE", 503)
        return response.json()

    original = registered_order_query()
    spec = ToolSpec.model_validate(original.spec.as_object()).model_copy(
        update={"handler_ref": "domain_tools.operations.sandbox_query@1"}
    )
    return RegisteredRead(
        spec=JsonSnapshot.capture(spec.model_dump(mode="json")),
        input_type=OrderQuery,
        output_type=OrderResult,
        handler=query,
        authorize=original.authorize,
    )
