"""显式保留CLI；无定时扫描、不自动迁移、不启动运行Worker或模型。"""

import argparse
import getpass
import json
from pathlib import Path

from harness.adapters.cleanup_copies import ManagedTraceFiles, OfficialCheckpointCopies
from harness.adapters.development_auth import LiveDevelopmentAuthenticator
from harness.adapters.execution_postgres import PostgresExecutionStore
from harness.adapters.retention_postgres import PostgresRetention
from harness.bootstrap.checkpoints import saver
from harness.bootstrap.event_loop import run
from harness.bootstrap.providers import model_values
from harness.bootstrap.server import load_server_settings
from harness.contracts.retention import CleanupRequest
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


class ConfiguredCopies:
    def __init__(self, dsn: str, root: Path) -> None:
        self.dsn, self.root = dsn, root

    async def erase(self, target: JsonSnapshot) -> None:
        # trace清理无需连接checkpoint服务；可选模块不作为依赖探测。
        if target.as_object()["kind"] == "trace":
            await ManagedTraceFiles(self.root).erase(OfficialCheckpointCopies._threads(target))
            return
        async with saver(self.dsn) as checkpoint:
            await OfficialCheckpointCopies(checkpoint, self.dsn, ManagedTraceFiles(self.root)).erase(target)

    async def absent(self, target: JsonSnapshot) -> bool:
        if target.as_object()["kind"] == "trace":
            return await ManagedTraceFiles(self.root).absent(OfficialCheckpointCopies._threads(target))
        async with saver(self.dsn) as checkpoint:
            return await OfficialCheckpointCopies(checkpoint, self.dsn, ManagedTraceFiles(self.root)).absent(target)


async def execute(root: Path, args: argparse.Namespace) -> dict[str, object]:
    settings = load_server_settings(root)
    if settings.auth_mode != "development":
        raise RuntimeFault("RETENTION_AUTH_NOT_INSTALLED", 503)
    auth = LiveDevelopmentAuthenticator(settings.identity_file, settings.environment)
    values = model_values(root)
    token = values.get("HARNESS_OPERATOR_TOKEN") or getpass.getpass("本地保留管理Token（不回显）：")
    actor = auth.authenticate("Bearer " + token)
    store = PostgresExecutionStore(settings.database_dsn.get_secret_value(), settings.admission)
    store.identity_source = auth
    async with store._transaction() as conn:
        versions = await (await conn.execute("SELECT version FROM harness.schema_versions ORDER BY version")).fetchall()
        if [row["version"] for row in versions] != list(range(1, 10)):
            raise RuntimeFault("RETENTION_EXPLICIT_MIGRATION_REQUIRED", 503)
    manager = PostgresRetention(store, ConfiguredCopies(settings.database_dsn.get_secret_value(), root))
    if args.command == "request":
        request = CleanupRequest(request_id=args.request_id, kind=args.kind, reason=args.reason,
            expected_revision=args.revision, artifact_ref=args.artifact_ref, expected_content_hash=args.content_hash)
        result = await manager.request(actor, args.session_id, request)
    elif args.command == "inspect":
        result = await manager.inspect(actor, args.job_id)
    elif args.command == "reconcile":
        result = await manager.reconcile(actor, args.job_id)
    else:
        result = await manager.advance(actor, args.job_id)
    return result.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description="本地显式保留管理；不扫描、不自动删除")
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    request.add_argument("--session-id", required=True)
    request.add_argument("--request-id", required=True)
    request.add_argument("--revision", type=int, required=True)
    request.add_argument("--kind", choices=("session", "events", "artifact", "trace", "extensions"), required=True)
    request.add_argument("--reason", choices=("ttl", "delete"), default="ttl")
    request.add_argument("--artifact-ref")
    request.add_argument("--content-hash")
    for name in ("inspect", "advance", "reconcile"):
        command = commands.add_parser(name)
        command.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(execute(Path.cwd().resolve(), args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
