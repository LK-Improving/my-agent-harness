"""实际Worker组合根；仅显式运行命令后启动，不随API执行图。"""

import argparse
import asyncio
import os
import signal
from pathlib import Path
from uuid import uuid4

from harness.adapters.local_process import current_process
from harness.bootstrap.assembly import DevelopmentAssembly
from harness.bootstrap.checkpoints import saver
from harness.bootstrap.event_loop import run
from harness.bootstrap.releases import read_release
from harness.bootstrap.server import load_server_settings
from harness.contracts.runtime import RuntimeFault
from harness.runtime.worker import Worker


async def serve(root: Path, release_manifest: str | None = None) -> None:
    # 可选追踪默认关闭，不能因宿主机全局环境开启原始内容外发。
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    settings = load_server_settings(root)
    if settings.environment != "development":
        raise RuntimeFault("DEVELOPMENT_WORKER_ONLY", 503)
    identity = uuid4().hex
    expected = read_release(root, release_manifest) if release_manifest else None
    assembly = DevelopmentAssembly(
        root, owner=identity, checkpointer=None, expected_release=expected
    )
    if not assembly.model_ready:
        raise RuntimeFault("MODEL_CAPABILITY_OR_CREDENTIAL_NOT_READY", 503)
    assembly.assert_compatible(assembly.manifest)
    await assembly.require_optional_schema()
    async with saver(settings.database_dsn.get_secret_value()) as checkpoint:
        assembly.engine.checkpointer = checkpoint
        assembly.store.process_identity = current_process()
        health = await assembly.store.health()
        if not health["schema"] or not health["checkpoint"]:
            raise RuntimeFault("EXPLICIT_MIGRATION_REQUIRED", 503)
        worker = Worker(
            assembly.store,
            assembly.engine,
            identity=identity,
            checkpoint_timeout_seconds=assembly.profile.checkpoint_verification_timeout_seconds,
        )
        loop = asyncio.get_running_loop()
        previous = {}
        for item in (signal.SIGINT, signal.SIGTERM):
            previous[item] = signal.getsignal(item)
            signal.signal(
                item, lambda signum, frame: loop.call_soon_threadsafe(worker.stop_claiming)
            )
        try:
            await worker.serve()
        finally:
            for item, handler in previous.items():
                signal.signal(item, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="单 Worker；可显式锁定预期发布快照")
    parser.add_argument("--release-manifest")
    args = parser.parse_args()
    run(serve(Path.cwd().resolve(), args.release_manifest))


if __name__ == "__main__":
    main()
