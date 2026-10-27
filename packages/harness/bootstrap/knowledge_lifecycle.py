"""显式知识生命周期CLI；不自动迁移、删库或启动清理任务。"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
from typing import TYPE_CHECKING

from harness.contracts.knowledge_lifecycle import DeleteDocument
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier

if TYPE_CHECKING:
    from harness.adapters.ingestion_sources import LocalSourceStore


class Resource(Contract):
    knowledge_base_id: Identifier
    document_id: Identifier


class JobResource(Contract):
    job_id: Identifier


def local_sources(root: Path, values: dict[str, str]) -> LocalSourceStore:
    from harness.adapters.ingestion_sources import LocalSourceStore
    from harness.bootstrap.ingestion import IngestionConfiguration

    config_ref = values.get("HARNESS_INGESTION_CONFIG_FILE", "")
    if not config_ref:
        raise RuntimeFault("INGESTION_CONFIGURATION_REQUIRED", 503)
    path = root / config_ref
    if path.stat().st_size > 65536:
        raise RuntimeFault("INGESTION_CONFIGURATION_TOO_LARGE", 503)
    config = IngestionConfiguration.model_validate_json(path.read_text("utf-8"))
    return LocalSourceStore(root / config.source_directory)


class ConfiguredSources:
    def __init__(self, root: Path, values: dict[str, str]) -> None:
        self.root, self.values = root, values

    async def delete(self, target: JsonSnapshot) -> None:
        await local_sources(self.root, self.values).delete(target)


class ConfiguredIndexes:
    def __init__(self, root: Path, values: dict[str, str]) -> None:
        self.root, self.values = root, values

    async def delete(self, target: JsonSnapshot) -> None:
        from pydantic import SecretStr

        from harness.adapters.milvus_cleanup import MilvusIndexCleanup
        from harness.adapters.milvus_index import MilvusDualIndex
        from harness.bootstrap.indexing import IndexConfiguration

        ref = self.values.get("HARNESS_INDEX_CONFIG_FILE", "")
        if not ref:
            raise RuntimeFault("INDEX_CONFIGURATION_REQUIRED", 503)
        path = self.root / ref
        if path.stat().st_size > 65536:
            raise RuntimeFault("INDEX_CONFIGURATION_TOO_LARGE", 503)
        config = IndexConfiguration.model_validate_json(path.read_text("utf-8"))
        backend = MilvusDualIndex(config.milvus_uri, SecretStr(self.values.get(config.token_env_var, "")),
                                  config.database, config.backend_hash(), rpc_timeout_seconds=config.rpc_timeout_seconds)
        await MilvusIndexCleanup(backend).delete(target)


async def execute(root: Path, args: argparse.Namespace) -> dict[str, object]:
    from harness.bootstrap.providers import model_values

    values = model_values(root)
    if values.get("HARNESS_RAG_ENABLED", "false").lower() != "true":
        raise RuntimeFault("RAG_DISABLED", 503)
    identity_ref, dsn = values.get("HARNESS_DEV_IDENTITY_FILE", ""), values.get("HARNESS_DATABASE_DSN", "")
    if not identity_ref or not dsn or values.get("HARNESS_AUTH_MODE", "development") != "development":
        raise RuntimeFault("CLEANUP_AUTH_CONFIGURATION_REQUIRED", 503)
    from harness.adapters.development_auth import LiveDevelopmentAuthenticator
    from harness.adapters.lifecycle_postgres import PostgresLifecycleRepository

    auth = LiveDevelopmentAuthenticator(root / identity_ref, values.get("HARNESS_ENVIRONMENT", "development"))
    token = values.get("HARNESS_OPERATOR_TOKEN") or getpass.getpass("知识管理Token（不回显）：")
    actor = auth.authenticate("Bearer " + token)
    repository = PostgresLifecycleRepository(dsn, auth.current)
    async with repository.uow.transaction() as conn:
        versions = await (await conn.execute("SELECT version FROM harness_rag.schema_versions ORDER BY version")).fetchall()
        if [row["version"] for row in versions] != [1, 2, 3, 4, 5]:
            raise RuntimeFault("CLEANUP_EXPLICIT_MIGRATION_REQUIRED", 503)
    if args.command in {"revoke", "delete"}:
        resource = Resource(knowledge_base_id=args.kb_id, document_id=args.document_id)
        if args.command == "revoke":
            await repository.revoke(actor, resource.knowledge_base_id, resource.document_id)
            return {"document_id": resource.document_id, "state": "revoked"}
        request = DeleteDocument(request_id=args.request_id, document_id=resource.document_id,
            retain_until=args.retain_until, source_store_ref=local_sources(root, values).storage_ref())
        return (await repository.request_delete(actor, resource.knowledge_base_id, request)).model_dump(mode="json")
    job_id = JobResource(job_id=args.job_id).job_id
    if args.command == "status":
        return (await repository.read(actor, job_id)).model_dump(mode="json")
    if args.command != "retry":
        raise RuntimeFault("CLEANUP_COMMAND_UNSUPPORTED", 422)
    from harness.ingestion.lifecycle import KnowledgeLifecycle

    service = KnowledgeLifecycle(repository, ConfiguredSources(root, values), ConfiguredIndexes(root, values))
    return (await service.advance(actor, job_id, max_steps=args.max_steps)).model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description="PG先撤权、显式申请清理及有界重试；不自动删除")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("revoke", "delete"):
        command = commands.add_parser(name)
        command.add_argument("--kb-id", required=True)
        command.add_argument("--document-id", required=True)
        if name == "delete":
            command.add_argument("--request-id", required=True)
            command.add_argument("--retain-until", required=True, help="带时区的ISO时间，明确回滚/保留窗口")
    status = commands.add_parser("status")
    status.add_argument("--job-id", required=True)
    retry = commands.add_parser("retry")
    retry.add_argument("--job-id", required=True)
    retry.add_argument("--max-steps", type=int, choices=range(1, 33), default=16)
    args = parser.parse_args()
    from harness.bootstrap.event_loop import run

    try:
        result = run(execute(Path.cwd().resolve(), args))
    except RuntimeFault as exc:
        print(json.dumps({"error_code": exc.code}))
        raise SystemExit(1) from None
    except (OSError, ValueError, ImportError):
        print(json.dumps({"error_code": "CLEANUP_LOCAL_CONFIGURATION_FAILED"}))
        raise SystemExit(1) from None
    except Exception:
        print(json.dumps({"error_code": "CLEANUP_COMMAND_FAILED"}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
