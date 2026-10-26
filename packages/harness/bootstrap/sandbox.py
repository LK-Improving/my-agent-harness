"""隔离运营沙箱组合根与显式建表入口；默认不启用。"""

import hmac
import os
import sys
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import dotenv_values
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi import Path as ApiPath
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import SecretStr

from domain_tools.operations.work_orders import CreateWorkOrderInput
from harness.bootstrap.api import BodyLimitMiddleware
from harness.bootstrap.event_loop import run
from harness.contracts.hashing import canonical_hash
from harness.contracts.specs import Contract, Identifier


class SandboxSettings(Contract):
    dsn: SecretStr
    token: SecretStr
    base_url: str


def load_sandbox_settings(root: Path) -> SandboxSettings:
    values = {
        key: value for key, value in dotenv_values(root / ".env").items() if value is not None
    }
    values.update(os.environ)
    if values.get("HARNESS_ENVIRONMENT") != "development":
        raise ValueError("运营沙箱只允许显式development")
    dsn = values.get("HARNESS_SANDBOX_DSN", "")
    parsed = urlparse(dsn)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.path.endswith(
        "_write_sandbox"
    ):
        raise ValueError("沙箱必须使用本机独立 *_write_sandbox 数据库")
    if dsn == values.get("HARNESS_DATABASE_DSN") or dsn == values.get("HARNESS_TEST_DSN"):
        raise ValueError("沙箱不能使用Harness业务库或M0探针库")
    token = values.get("HARNESS_SANDBOX_TOKEN", "")
    if len(token) < 32:
        raise ValueError("请配置至少32字符的独立HARNESS_SANDBOX_TOKEN")
    return SandboxSettings(
        dsn=SecretStr(dsn),
        token=SecretStr(token),
        base_url=values.get("HARNESS_SANDBOX_BASE_URL", "http://127.0.0.1:18081"),
    )


class SandboxWrite(Contract):
    tenant_id: Identifier
    application_id: Identifier
    arguments: CreateWorkOrderInput


def create_sandbox(settings: SandboxSettings) -> FastAPI:
    app = FastAPI(title="隔离运营工单沙箱", debug=False)
    app.add_middleware(BodyLimitMiddleware)
    ResourceId = Annotated[Identifier, ApiPath()]
    QueryId = Annotated[Identifier, Query()]

    async def authenticated(authorization: str | None = Header(default=None)) -> None:
        if authorization is None or not hmac.compare_digest(
            authorization.encode(), ("Bearer " + settings.token.get_secret_value()).encode()
        ):
            raise HTTPException(401, "UNAUTHENTICATED")

    @app.post("/writes/{operation_id}", dependencies=[Depends(authenticated)])
    async def write(
        operation_id: ResourceId,
        body: SandboxWrite,
        x_sandbox_fault: str | None = Header(default=None),
    ) -> dict[str, Any] | None:
        if x_sandbox_fault not in {None, "before_write", "after_write", "hidden", "invalid_output"}:
            raise HTTPException(422, "INVALID_FAULT_MODE")
        args_hash = canonical_hash(body.arguments.model_dump(mode="json"))
        fail_response = x_sandbox_fault in {"before_write", "after_write"}
        async with await AsyncConnection.connect(
            settings.dsn.get_secret_value(), row_factory=dict_row, connect_timeout=5
        ) as conn:
            async with conn.transaction():
                # 同operation请求串行；工单业务唯一约束独立防止跨key重复。
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (operation_id,)
                )
                row = await (
                    await conn.execute(
                        "SELECT * FROM work_order_sandbox.requests WHERE operation_id=%s",
                        (operation_id,),
                    )
                ).fetchone()
                if row:
                    if (row["tenant_id"], row["application_id"], row["args_hash"]) != (
                        body.tenant_id,
                        body.application_id,
                        args_hash,
                    ):
                        raise HTTPException(409, "IDEMPOTENCY_CONFLICT")
                    if row["state"] == "not_executed":
                        raise HTTPException(409, "OPERATION_FENCED_NOT_EXECUTED")
                    return dict(row["result"])
                if x_sandbox_fault == "before_write":
                    # 持久终止标记是正面未执行证明；该key之后永不写入，不是404推断。
                    await conn.execute(
                        (
                            "INSERT INTO "
                            "work_order_sandbox.requests(operation_id,tenant_id,application_id"
                            ",args_hash,state) VALUES(%s,%s,%s,%s,'not_executed')"
                        ),
                        (operation_id, body.tenant_id, body.application_id, args_hash),
                    )
                    result = None
                else:
                    work_order_id = "wo_" + uuid4().hex
                    data = {"work_order_id": work_order_id, "order_id": body.arguments.order_id}
                    result = {
                        "external_reference": work_order_id,
                        "data": data
                        if x_sandbox_fault != "invalid_output"
                        else {"malformed": True},
                    }
                    await conn.execute(
                        (
                            "INSERT INTO "
                            "work_order_sandbox.work_orders(work_order_id,operation_id,tenant_"
                            "id,application_id,order_id) VALUES(%s,%s,%s,%s,%s)"
                        ),
                        (
                            work_order_id,
                            operation_id,
                            body.tenant_id,
                            body.application_id,
                            body.arguments.order_id,
                        ),
                    )
                    await conn.execute(
                        (
                            "INSERT INTO "
                            "work_order_sandbox.requests(operation_id,tenant_id,application_id"
                            ",args_hash,state,result,visible_at) "
                            "VALUES(%s,%s,%s,%s,'succeeded',%s,clock_timestamp()+%s * "
                            "interval '1 second')"
                        ),
                        (
                            operation_id,
                            body.tenant_id,
                            body.application_id,
                            args_hash,
                            Jsonb(result),
                            5 if x_sandbox_fault == "hidden" else 0,
                        ),
                    )
        # 故障响应在数据库事务提交之后发生，独立持久写入数不会因HTTP失败回滚。
        if fail_response:
            raise HTTPException(503, "INJECTED_RESPONSE_LOSS")
        return result

    @app.get("/writes/{operation_id}", dependencies=[Depends(authenticated)])
    async def lookup(
        operation_id: ResourceId, tenant_id: QueryId, application_id: QueryId
    ) -> dict[str, Any]:
        async with await AsyncConnection.connect(
            settings.dsn.get_secret_value(), row_factory=dict_row, connect_timeout=5
        ) as conn:
            row = await (
                await conn.execute(
                    (
                        "SELECT * FROM work_order_sandbox.requests WHERE operation_id=%s "
                        "AND tenant_id=%s AND application_id=%s AND "
                        "visible_at<=clock_timestamp()"
                    ),
                    (operation_id, tenant_id, application_id),
                )
            ).fetchone()
        if row is None:
            raise HTTPException(404, "NOT_VISIBLE_OR_UNKNOWN")
        succeeded = row["state"] == "succeeded"
        return {
            "operation_id": operation_id,
            "args_hash": row["args_hash"],
            "conclusion": "confirmed_succeeded" if succeeded else "confirmed_not_executed",
            "authoritative_not_executed": not succeeded,
            "evidence_ref": "sandbox/request/" + operation_id,
            "result": row["result"],
        }

    @app.get("/counts", dependencies=[Depends(authenticated)])
    async def counts(tenant_id: QueryId, application_id: QueryId) -> dict[str, Any] | None:
        async with await AsyncConnection.connect(
            settings.dsn.get_secret_value(), row_factory=dict_row, connect_timeout=5
        ) as conn:
            row = await (
                await conn.execute(
                    (
                        "SELECT count(*) AS writes FROM work_order_sandbox.work_orders "
                        "WHERE tenant_id=%s AND application_id=%s"
                    ),
                    (tenant_id, application_id),
                )
            ).fetchone()
        return row

    @app.get("/orders/{order_id}", dependencies=[Depends(authenticated)])
    async def order(order_id: str, tenant_id: QueryId, application_id: QueryId) -> dict[str, Any]:
        args = CreateWorkOrderInput(order_id=order_id)
        async with await AsyncConnection.connect(
            settings.dsn.get_secret_value(), row_factory=dict_row, connect_timeout=5
        ) as conn:
            row = await (
                await conn.execute(
                    (
                        "SELECT work_order_id FROM work_order_sandbox.work_orders WHERE "
                        "tenant_id=%s AND application_id=%s AND order_id=%s"
                    ),
                    (tenant_id, application_id, args.order_id),
                )
            ).fetchone()
        return {
            "order_id": args.order_id,
            "status": "created" if row else "not_found",
            "source": "write-sandbox",
            "work_order_id": row["work_order_id"] if row else None,
        }

    return app


def application() -> FastAPI:
    return create_sandbox(load_sandbox_settings(Path.cwd().resolve()))


async def initialize(root: Path) -> None:
    settings = load_sandbox_settings(root)
    async with await AsyncConnection.connect(
        settings.dsn.get_secret_value(), connect_timeout=5
    ) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(184031,2)")
            exists = await (
                await conn.execute("SELECT to_regclass('work_order_sandbox.requests')")
            ).fetchone()
            if exists == (None,):
                await conn.execute(
                    (root / "migrations/sandbox/0001_work_orders.sql").read_text("utf-8"),
                    prepare=False,
                )


if __name__ == "__main__":
    if sys.argv[1:] != ["init"]:
        raise SystemExit("显式初始化请使用 python -m harness.bootstrap.sandbox init")
    run(initialize(Path.cwd().resolve()))
