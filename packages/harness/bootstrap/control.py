"""本机受控运维命令；不提供强制成功、任意checkpoint或自动接管选项。"""

import argparse
import getpass
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from harness.adapters.development_auth import LiveDevelopmentAuthenticator
from harness.adapters.local_process import prove_terminated
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.bootstrap.assembly import DevelopmentAssembly
from harness.bootstrap.checkpoints import saver
from harness.bootstrap.event_loop import run
from harness.bootstrap.releases import export_release, read_release
from harness.bootstrap.server import load_server_settings
from harness.contracts.runtime import Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


def operator(root: Path, auth: LiveDevelopmentAuthenticator) -> Principal:
    configured = dotenv_values(root / ".env").get("HARNESS_OPERATOR_TOKEN")
    token = (
        os.environ.get("HARNESS_OPERATOR_TOKEN")
        or configured
        or getpass.getpass("本地操作者Token（不回显）：")
    )
    return auth.authenticate("Bearer " + token)


async def control(
    root: Path, action: str, resource: str | None, release_manifest: str | None = None
) -> None:
    settings = load_server_settings(root)
    store = PostgresRecoveryStore(settings.database_dsn.get_secret_value(), settings.admission)
    auth = LiveDevelopmentAuthenticator(settings.identity_file, settings.environment)
    store.identity_source = auth
    expected = read_release(root, release_manifest) if release_manifest else None
    if action == "export-release":
        assembly = DevelopmentAssembly(
            root, owner="local-control", checkpointer=None, expected_release=expected
        )
        path = export_release(root, assembly.manifest)
        print("本地候选快照已生成：" + path.relative_to(root).as_posix())
        return
    actor = operator(root, auth)
    if action == "register-candidate":
        assembly = DevelopmentAssembly(
            root, owner="local-control", checkpointer=None, expected_release=expected
        )
        await assembly.store.register_release(actor, assembly.manifest)
        print("已注册候选：" + assembly.manifest.spec.release_id)
        return
    if resource is None:
        raise RuntimeFault("RESOURCE_ID_REQUIRED", 422)
    if action == "inspect-run":
        release = await store.release_for_run(actor, resource)
        try:
            assembly = DevelopmentAssembly(
                root, owner="local-control", checkpointer=None, expected_release=expected
            )
            assembly.assert_compatible(release)
            compatibility = "compatible"
        except RuntimeFault as exc:
            compatibility = exc.code
        except (ValueError, OSError):
            compatibility = "LOCAL_RELEASE_CONFIGURATION_INVALID"
        print(
            json.dumps(
                {
                    "run_id": resource,
                    "release_id": release.spec.release_id,
                    "manifest_hash": release.content_hash(),
                    "compatibility": compatibility,
                },
                ensure_ascii=False,
            )
        )
        return
    if action == "close-reconciliation":
        await store.close_reconciliation(actor, resource)
        print("只读核对尝试已收尾；unknown、业务guard和原计数保持不变。")
        return
    if action == "retire-worker":
        recorded = await store.registered_process(actor, resource)
        termination = prove_terminated(recorded)
        await store.retire_worker(actor, resource, termination)
        print("旧进程终止证明已落账；受影响运行保留recovery_required，未自动恢复。")
        return
    run, revision, _ = await store.recovery_run(actor, resource)
    if action == "abort-run":
        state = await store.recover_run(actor, run, revision, None, abort=True)
        print("收尾状态：" + state)
        return
    # 此处只读取实际安装图和官方checkpoint，不在业务事务中持锁查询。
    assembly = DevelopmentAssembly(
        root, owner="local-control", checkpointer=None, expected_release=expected
    )
    assembly.engine.observed_manifest_hash(run.manifest)
    async with saver(assembly.settings.database_dsn.get_secret_value()) as checkpoint:
        assembly.engine.checkpointer = checkpoint
        graph = assembly.engine.graph(run)
        state = await graph.aget_state(assembly.engine.config(run))
        proof: JsonSnapshot | None = None
        if state.values:
            if (
                state.values.get("harness_run_id") != run.run_id
                or state.values.get("harness_manifest_hash") != run.manifest.content_hash()
            ):
                raise RuntimeFault("RECOVERY_CHECKPOINT_BINDING_MISMATCH")
            interrupts = [item for task in state.tasks for item in task.interrupts]
            if interrupts:
                if len(interrupts) != 1:
                    raise RuntimeFault("AMBIGUOUS_RECOVERY_INTERRUPT")
                item = interrupts[0]
                actual = await assembly.engine.suspension_proof(run, item.value["operation_id"])
                proof = JsonSnapshot.capture(actual.model_dump(mode="json"))
            else:
                proof = JsonSnapshot.capture(
                    {
                        "kind": "continuation",
                        "run_id": run.run_id,
                        "manifest_hash": run.manifest.content_hash(),
                        "checkpoint_ref": state.config["configurable"]["checkpoint_id"],
                    }
                )
        assembly.assert_compatible(run.manifest)
        result = await assembly.store.recover_run(actor, run, revision, proof)
        print("受控恢复状态：" + result)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地候选与受控恢复")
    parser.add_argument(
        "action",
        choices=[
            "export-release",
            "register-candidate",
            "retire-worker",
            "resume-run",
            "abort-run",
            "close-reconciliation",
            "inspect-run",
        ],
    )
    parser.add_argument("resource", nargs="?")
    parser.add_argument("--release-manifest", help="本工作区内预期发布快照；只比对，不动态安装实现")
    args = parser.parse_args()
    try:
        run(control(Path.cwd().resolve(), args.action, args.resource, args.release_manifest))
    except RuntimeFault as exc:
        raise SystemExit(exc.code) from None


if __name__ == "__main__":
    main()
