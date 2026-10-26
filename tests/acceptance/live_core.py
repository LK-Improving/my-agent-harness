"""显式千问/API/SSE/审批核对验收；需先配置独立本机验收环境。"""

import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
from harness.adapters.local_process import prove_terminated
from harness.bootstrap.assembly import DevelopmentAssembly
from harness.bootstrap.event_loop import run
from harness.contracts.runtime import Principal

ROOT = Path.cwd().resolve()
FOLDER = ROOT / ".runtime"
PREFIX = os.environ.get("HARNESS_ACCEPTANCE_PREFIX", "m2r07")
if not re.fullmatch(r"[a-z0-9-]{1,64}", PREFIX):
    raise ValueError("验收证据前缀无效")
TERMINAL = {"succeeded", "failed", "cancelled", "expired", "partial"}


async def verify() -> dict:
    actors = json.loads((ROOT / os.environ["HARNESS_DEV_IDENTITY_FILE"]).read_text("utf-8"))
    tokens = {a["subject_id"]: key for key, a in actors.items()}
    auth = {"Authorization": "Bearer " + tokens["requester"]}
    approval_auth = {"Authorization": "Bearer " + tokens["approver"]}
    assembly = DevelopmentAssembly(ROOT, owner="acceptance-inspector", checkpointer=None)
    manifest = assembly.manifest
    expected_adapter = os.environ.get("HARNESS_ACCEPTANCE_EXPECT_ADAPTER")
    if expected_adapter:
        assert assembly.profile.route.adapter_ref == expected_adapter
        assert assembly.model_runtime.binding_snapshot is not None
        assert type(assembly.model_runtime.transport).__name__ == "OpenAIChatCompletionsClient"
    report = {
        "manifest_hash": manifest.content_hash(),
        "release_id": manifest.spec.release_id,
        "model": assembly.profile.route.model,
        "provider": assembly.profile.route.provider,
        "adapter_ref": assembly.profile.route.adapter_ref,
        "transport_class": type(assembly.model_runtime.transport).__name__,
        "binding_hash": assembly.model_runtime.binding_snapshot.content_hash
        if assembly.model_runtime.binding_snapshot
        else None,
        "source_tree_hash": manifest.source_tree_hash,
        "dependency_lock_hash": manifest.dependency_lock_hash,
        "run_ids": [],
        "checks": {},
    }
    (FOLDER / (PREFIX + "-live-manifest.json")).write_text(
        manifest.model_dump_json(indent=2), "utf-8"
    )
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:18080", timeout=30, trust_env=False
    ) as client:

        async def call(method, url, *, headers=None, body=None, expected=(200, 201, 202)):
            response = await client.request(method, url, headers=headers or auth, json=body)
            if response.status_code not in expected:
                raise AssertionError(f"HTTP {response.status_code}: {response.text[:300]}")
            return response.json()

        async def wait_run(run_id, desired, seconds=120):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                view = await call("GET", "/api/v1/runs/" + run_id)
                if view["status"] in desired:
                    return view
                if view["status"] in TERMINAL:
                    raise AssertionError(
                        f"run stopped: {view['status']} / {view.get('error_code')}"
                    )
                await asyncio.sleep(1)
            raise TimeoutError("run status wait expired")

        async def new_run(message):
            context = await call(
                "POST",
                "/api/v1/evaluation-contexts",
                body={
                    "candidate_release_id": manifest.spec.release_id,
                    "dataset_hash": "a" * 64,
                    "gate_policy_version": "m2r07-live@1",
                    "resource_scopes": [
                        "work_orders:read:synthetic",
                        "work_orders:write:synthetic",
                    ],
                    "expires_at": (datetime.now(UTC) + timedelta(minutes=20)).isoformat(),
                },
            )
            session = await call(
                "POST",
                "/api/v1/sessions",
                body={
                    "agent_id": manifest.spec.agent_id,
                    "evaluation_context_id": context["context_id"],
                },
            )
            request = {"request_id": uuid4().hex, "message": message, "expected_revision": 0}
            url = f"/api/v1/sessions/{session['session_id']}/runs"
            created = await call("POST", url, body=request)
            assert (await call("POST", url, body=request))["run_id"] == created["run_id"]
            report["run_ids"].append(created["run_id"])
            return created

        async def sse(run_id, cursor=0, first_only=False):
            events = []
            async with client.stream(
                "GET",
                f"/api/v1/runs/{run_id}/events",
                headers={**auth, "Last-Event-ID": str(cursor)},
            ) as response:
                assert response.status_code == 200, (
                    f"SSE HTTP {response.status_code}: {await response.aread()}"
                )
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        events.append(event)
                        if first_only or event["type"] in {
                            "run.completed",
                            "run.failed",
                            "run.cancelled",
                            "run.expired",
                        }:
                            break
            return events

        await call(
            "POST",
            f"/api/v1/agents/{manifest.spec.agent_id}/releases",
            body=manifest.model_dump(mode="json"),
        )
        # 核心、官方saver与真实供应商的就绪条件必须同时成立。
        for _ in range(40):
            ready = await client.get("/api/health/ready")
            if ready.status_code == 200:
                break
            await asyncio.sleep(0.5)
        assert ready.status_code == 200, "core did not become ready"
        report["checks"]["core_ready"] = True
        order = "synthetic-live-" + uuid4().hex[:10]
        sandbox_headers = {"Authorization": "Bearer " + os.environ["HARNESS_SANDBOX_TOKEN"]}
        sandbox_scope = {"tenant_id": "m2r07-local", "application_id": "m2r07"}
        # 只在独立沙箱建立合成查询夹具，业务请求仍通过实际模型和统一工具执行。
        async with httpx.AsyncClient(base_url="http://127.0.0.1:18081", trust_env=False) as sandbox:
            seeded = await sandbox.post(
                "/writes/" + uuid4().hex,
                headers=sandbox_headers,
                json={**sandbox_scope, "arguments": {"order_id": order}},
            )
            assert seeded.status_code == 200
            before = (
                await sandbox.get("/counts", headers=sandbox_headers, params=sandbox_scope)
            ).json()["writes"]
        read = await new_run(f"请调用工具查询 {order} 工单，并在答案中说明这是合成沙箱数据。")
        prefix = await sse(read["run_id"], first_only=True)
        assert prefix
        view = await wait_run(read["run_id"], {"succeeded"})
        assert view["result"]["task_outcome"] == "achieved"
        replay = await sse(read["run_id"], prefix[-1]["seq"])
        events = prefix + replay
        assert [x["seq"] for x in events] == list(range(1, events[-1]["seq"] + 1))
        assert sum(x["type"] == "run.completed" for x in events) == 1
        assert (
            "".join(x["payload"]["delta"] for x in events if x["type"] == "answer.delta")
            == view["result"]["answer"]
        )
        denied = await client.get(
            f"/api/v1/runs/{read['run_id']}",
            headers={"Authorization": "Bearer " + tokens["outsider"]},
        )
        assert denied.status_code == 404
        report["checks"].update(
            {
                "real_qwen_tool_query": True,
                "sse_disconnect_replay": True,
                "cross_tenant_denied": True,
            }
        )
        print("live query and SSE replay passed", flush=True)

        write_order = "synthetic-write-" + uuid4().hex[:10]
        write = await new_run(
            f"请调用工具为 {write_order} 创建一个合成沙箱工单，完成后告诉我订单编号及工单编号。"
        )
        await wait_run(write["run_id"], {"waiting_approval"})
        items = (await call("GET", "/api/v1/approvals", headers=approval_auth))["items"]
        approval = next(x for x in items if x["arguments"]["order_id"] == write_order)
        decision = {
            "request_id": uuid4().hex,
            "binding_hash": approval["binding_hash"],
            "expected_revision": approval["revision"],
            "decision": "approve",
        }
        url = f"/api/v1/approvals/{approval['approval_id']}/decisions"
        self_response = await client.post(url, headers=auth, json=decision)
        assert self_response.status_code == 403
        await call("POST", url, headers=approval_auth, body=decision)
        await wait_run(write["run_id"], {"reconciling"})
        async with httpx.AsyncClient(base_url="http://127.0.0.1:18081", trust_env=False) as sandbox:
            after_fault = (
                await sandbox.get("/counts", headers=sandbox_headers, params=sandbox_scope)
            ).json()["writes"]
        assert after_fault == before + 1
        reconciled = await call(
            "POST",
            f"/api/v1/runs/{write['run_id']}/reconciliations",
            headers=approval_auth,
            body={
                "request_id": "reconcile-" + uuid4().hex,
                "operation_id": approval["operation_id"],
            },
        )
        assert reconciled["state"] == "succeeded"
        completed = await wait_run(write["run_id"], {"succeeded"})
        assert completed["result"]["task_outcome"] == "achieved"
        async with httpx.AsyncClient(base_url="http://127.0.0.1:18081", trust_env=False) as sandbox:
            after = (
                await sandbox.get("/counts", headers=sandbox_headers, params=sandbox_scope)
            ).json()["writes"]
        assert after == after_fault
        write_events = await sse(write["run_id"])
        assert sum(x["type"] == "run.completed" for x in write_events) == 1
        report["checks"].update(
            {
                "independent_approval": True,
                "self_approval_denied": True,
                "real_http_write_unknown_reconciled": True,
                "sandbox_no_duplicate_write": True,
            }
        )
        report["sandbox_counts"] = {
            "before": before,
            "after_fault": after_fault,
            "after_resume": after,
        }
        async with assembly.store._transaction() as conn:
            usage = await (
                await conn.execute(
                    "SELECT run_id,state,result->'usage' AS usage,request->'route' AS route "
                    "FROM harness.model_calls "
                    "WHERE run_id=ANY(%s)",
                    (report["run_ids"],),
                )
            ).fetchall()
            commits = await (
                await conn.execute(
                    "SELECT run_id,count(*) AS commits FROM harness.commits "
                    "WHERE run_id=ANY(%s) GROUP BY run_id",
                    (report["run_ids"],),
                )
            ).fetchall()
        assert all(x["state"] == "succeeded" and x["usage"] is not None for x in usage)
        assert len(commits) == 2 and all(x["commits"] == 1 for x in commits)
        assert all(x["route"]["adapter_ref"] == assembly.profile.route.adapter_ref for x in usage)
        assert all(
            x["route"]["configuration_hash"] == assembly.profile.route.configuration_hash
            for x in usage
        )
        report["checks"]["persisted_route_matches_installed_adapter"] = True
        report["model_usage"] = usage
        report["checks"]["durable_unique_commits_and_usage"] = True
        # 断线与正常完成均应及时释放SSE名额，不等待30秒短租期才恢复可用。
        for _ in range(20):
            async with assembly.store._transaction() as conn:
                active = await assembly.store._one(
                    conn,
                    "SELECT count(*) AS count FROM harness.sse_connections "
                    "WHERE tenant_id='m2r07-local' AND expires_at>clock_timestamp()",
                )
            if active["count"] == 0:
                break
            await asyncio.sleep(0.1)
        assert active["count"] == 0
        report["checks"]["sse_disconnect_releases_slot"] = True
        print("live approval, response failure and reconciliation passed", flush=True)
    return report


async def retire_dead_validation_worker() -> None:
    assembly = DevelopmentAssembly(ROOT, owner="acceptance-cleanup", checkpointer=None)
    actors = json.loads((ROOT / os.environ["HARNESS_DEV_IDENTITY_FILE"]).read_text("utf-8"))
    actor = next(
        Principal.model_validate(value)
        for value in actors.values()
        if value["subject_id"] == "operator"
    )
    async with assembly.store._transaction() as conn:
        row = await (
            await conn.execute("SELECT identity FROM harness.worker_registration WHERE singleton")
        ).fetchone()
    if row is not None:
        process = await assembly.store.registered_process(actor, row["identity"])
        proof = prove_terminated(process)
        await assembly.store.retire_worker(actor, row["identity"], proof)
        print("已证明验收Worker进程终止并按协议退役登记。", flush=True)


def main() -> None:
    for port in (18080, 18081, 18082):
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"验收端口已占用：{port}")
    processes = []
    handles = []
    try:
        commands = {
            "sandbox": ["-m", "apps.sandbox"],
            "proxy": [
                "-m",
                "uvicorn",
                "tests.acceptance.fault_proxy:app",
                "--host",
                "127.0.0.1",
                "--port",
                "18082",
                "--no-access-log",
                "--log-level",
                "critical",
            ],
            "api": ["-m", "apps.api"],
            "worker": ["-m", "apps.worker"],
        }
        for name, arguments in commands.items():
            handle = (FOLDER / (PREFIX + "-" + name + ".log")).open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [sys.executable, *arguments],
                    stdout=handle,
                    stderr=handle,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0,
                )
            )
        time.sleep(3)
        assert all(p.poll() is None for p in processes), "验收子进程启动失败，查看本地脱敏日志"
        report = run(verify())
        report["status"] = "passed"
        (FOLDER / (PREFIX + "-live-report.json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
        )
        print("Live acceptance report: .runtime/" + PREFIX + "-live-report.json", flush=True)
    finally:
        # 只停止本脚本创建的进程；先给Worker排空机会，再关闭HTTP服务。
        for p in reversed(processes):
            if p.poll() is None:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.send_signal(signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    p.wait(timeout=5)
        for handle in handles:
            handle.close()
        run(retire_dead_validation_worker())


if __name__ == "__main__":
    main()
