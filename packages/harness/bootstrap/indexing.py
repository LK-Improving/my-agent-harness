"""显式可选的代际构建/快照管理CLI；不在应用启动时加载或连接Milvus。"""

import argparse
import getpass
import json
from pathlib import Path

from pydantic import Field, SecretStr

from harness.contracts.hashing import canonical_hash
from harness.contracts.indexing import IndexBinding, IndexJob, IndexProfile, IndexRequest
from harness.contracts.providers import Digest
from harness.contracts.runtime import RuntimeFault
from harness.contracts.specs import Contract, Identifier


class IndexConfiguration(Contract):
    milvus_uri: str = Field(min_length=1, max_length=1024)
    database: Identifier = "default"
    credential_profile_ref: Identifier = "milvus-local@1"
    token_env_var: str = Field(default="HARNESS_MILVUS_TOKEN", pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    environment_profile_hash: Digest
    rpc_timeout_seconds: int = Field(default=20, ge=1, le=30, strict=True)
    profile: IndexProfile = IndexProfile()

    def backend_hash(self) -> str:
        # 固定端点、数据库、非秘密账户引用、期限与协议，不把Token写进manifest。
        return canonical_hash(self.model_dump(mode="json", exclude={"profile", "environment_profile_hash"}))


def job_summary(job: IndexJob) -> dict[str, object]:
    result = job.model_dump(mode="json", exclude={"binding", "scope"})
    result["binding_hash"] = job.binding.content_hash()
    return result


async def execute(root: Path, args: argparse.Namespace) -> dict[str, object]:
    from harness.bootstrap.providers import model_values

    values = model_values(root)
    if values.get("HARNESS_RAG_ENABLED", "false").lower() != "true":
        raise RuntimeFault("RAG_DISABLED", 503)
    dsn = values.get("HARNESS_DATABASE_DSN", "")
    identity_ref = values.get("HARNESS_DEV_IDENTITY_FILE", "")
    if not dsn or not identity_ref or values.get("HARNESS_AUTH_MODE", "development") != "development":
        raise RuntimeFault("INDEX_AUTH_CONFIGURATION_REQUIRED", 503)
    from harness.adapters.development_auth import LiveDevelopmentAuthenticator
    from harness.adapters.indexing_postgres import PostgresIndexRepository

    auth = LiveDevelopmentAuthenticator(root / identity_ref, values.get("HARNESS_ENVIRONMENT", "development"))
    token = values.get("HARNESS_OPERATOR_TOKEN") or getpass.getpass("知识管理Token（不回显）：")
    actor = auth.authenticate("Bearer " + token)
    repository = PostgresIndexRepository(dsn, auth.current)
    if args.command == "status":
        return job_summary(await repository.read(actor, args.job_id))
    if args.command == "publish":
        return (await repository.publish(actor, args.job_id)).model_dump(mode="json")
    if args.command == "activate":
        revision = await repository.activate(actor, args.snapshot_id, args.expected_snapshot, args.expected_revision)
        return {"snapshot_id": args.snapshot_id, "revision": revision}
    if args.command == "offline":
        await repository.offline(actor, args.snapshot_id)
        return {"snapshot_id": args.snapshot_id, "state": "offline"}
    if args.command == "active":
        return await repository.active(actor, args.kb_id)
    if args.command == "deactivate":
        revision = await repository.deactivate(actor, args.kb_id, args.expected_snapshot, args.expected_revision)
        return {"snapshot_id": None, "revision": revision}
    if args.command != "build":
        raise RuntimeFault("INDEX_COMMAND_UNSUPPORTED", 422)
    config_ref = values.get("HARNESS_INDEX_CONFIG_FILE", "")
    if not config_ref:
        raise RuntimeFault("INDEX_CONFIGURATION_REQUIRED", 503)
    path = root / config_ref
    if path.stat().st_size > 65536:
        raise RuntimeFault("INDEX_CONFIGURATION_TOO_LARGE", 503)
    config = IndexConfiguration.model_validate_json(path.read_text("utf-8"))
    from harness.adapters.milvus_index import MilvusDualIndex
    from harness.adapters.retrieval_models_postgres import PostgresRetrievalLedger
    from harness.bootstrap.retrieval_models import load_retrieval_runtime
    from harness.contracts.retrieval_models import RetrievalLimits
    from harness.ingestion.indexing import IndexBuildService

    models = load_retrieval_runtime(
        root, enabled=True, environment_profile_hash=config.environment_profile_hash,
        ledger=PostgresRetrievalLedger(dsn), authority=repository,
    )
    if models is None:
        raise RuntimeFault("INDEX_MODELS_REQUIRED", 503)
    binding = IndexBinding(
        profile=config.profile, model_snapshot=models.snapshot,
        embedding_binding_hash=models.embedding_binding.snapshot.content_hash,
        rerank_binding_hash=models.rerank_binding.snapshot.content_hash, backend_hash=config.backend_hash(),
    )
    backend = MilvusDualIndex(config.milvus_uri, SecretStr(values.get(config.token_env_var, "")),
                              config.database, config.backend_hash(), rpc_timeout_seconds=config.rpc_timeout_seconds)
    limits = RetrievalLimits.model_validate(models.snapshot.as_object()["limits"])
    service = IndexBuildService(repository, backend, models.gateway, binding, models.embedding_binding.profile, limits)
    request = IndexRequest(request_id=args.request_id, version_ids=tuple(sorted(args.version_id)))
    return job_summary(await service.build(actor, args.kb_id, request))


def nonnegative(value: str) -> int:
    revision = int(value)
    if revision < 0:
        raise argparse.ArgumentTypeError("revision不能为负数")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description="受限知识索引代际与快照管理")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--kb-id", required=True)
    build.add_argument("--request-id", required=True)
    build.add_argument("--version-id", action="append", required=True)
    for name in ("status", "publish"):
        command = commands.add_parser(name)
        command.add_argument("--job-id", required=True)
    activate = commands.add_parser("activate")
    activate.add_argument("--snapshot-id", required=True)
    activate.add_argument("--expected-snapshot")
    activate.add_argument("--expected-revision", type=nonnegative, required=True)
    offline = commands.add_parser("offline")
    offline.add_argument("--snapshot-id", required=True)
    active = commands.add_parser("active")
    active.add_argument("--kb-id", required=True)
    deactivate = commands.add_parser("deactivate")
    deactivate.add_argument("--kb-id", required=True)
    deactivate.add_argument("--expected-snapshot", required=True)
    deactivate.add_argument("--expected-revision", type=nonnegative, required=True)
    args = parser.parse_args()
    from harness.bootstrap.event_loop import run

    try:
        result = run(execute(Path.cwd().resolve(), args))
    except RuntimeFault as exc:
        print(json.dumps({"error_code": exc.code}))
        raise SystemExit(1) from None
    except (ValueError, OSError, ImportError):
        print(json.dumps({"error_code": "INDEX_LOCAL_CONFIGURATION_OR_DEPENDENCY_FAILED"}))
        raise SystemExit(1) from None
    except Exception:
        # CLI只交付稳定失败码，不把PG详情或SDK包含正文的异常打印到终端。
        print(json.dumps({"error_code": "INDEX_COMMAND_FAILED"}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
