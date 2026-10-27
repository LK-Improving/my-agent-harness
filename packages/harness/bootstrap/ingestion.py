"""受限本机管理CLI；身份沿用服务端Token映射，不接受租户/角色或任意上传路径。"""

import argparse
import asyncio
import getpass
import json
from pathlib import Path

from pydantic import Field

from harness.contracts.ingestion import IngestionProfile, Upload
from harness.contracts.runtime import RuntimeFault
from harness.contracts.specs import Contract, Identifier


class IngestionConfiguration(Contract):
    staging_directory: str = Field(min_length=1)
    source_directory: str = Field(min_length=1)
    profile: IngestionProfile = IngestionProfile()


def read_staged(directory: Path, filename: str, limit: int) -> bytes:
    if (not filename or filename in {".", ".."}
        or any(c in filename for c in '/\\:\x00') or any(ord(c) < 32 for c in filename)):
        raise RuntimeFault("INGESTION_UPLOAD_PATH_INVALID", 422)
    root = directory.resolve(strict=True)
    candidate = root / filename
    if candidate.is_symlink() or candidate.resolve(strict=True).parent != root:
        raise RuntimeFault("INGESTION_UPLOAD_PATH_INVALID", 422)
    if not candidate.is_file():
        raise RuntimeFault("INGESTION_UPLOAD_PATH_INVALID", 422)
    with candidate.open("rb") as stream:
        body = stream.read(limit + 1)
    if not body or len(body) > limit:
        raise RuntimeFault("INGESTION_UPLOAD_SIZE_EXCEEDED", 413)
    return body


class KnowledgeBaseInput(Contract):
    knowledge_base_id: Identifier


async def execute(root: Path, args: argparse.Namespace) -> dict[str, object]:
    # 未显式启用时不导入PG/解析适配器、不读取身份配置或连接服务。
    from harness.bootstrap.providers import model_values

    values = model_values(root)
    if values.get("HARNESS_RAG_ENABLED", "false").lower() != "true":
        raise RuntimeFault("RAG_DISABLED", 503)
    config_ref = values.get("HARNESS_INGESTION_CONFIG_FILE", "")
    dsn = values.get("HARNESS_DATABASE_DSN", "")
    if not config_ref or not dsn:
        raise RuntimeFault("INGESTION_CONFIGURATION_REQUIRED", 503)
    config_path = (root / config_ref).resolve()
    if config_path.stat().st_size > 65536:
        raise RuntimeFault("INGESTION_CONFIGURATION_TOO_LARGE", 503)
    config = IngestionConfiguration.model_validate_json(config_path.read_text("utf-8"))
    from harness.adapters.ingestion_postgres import PostgresIngestionRepository
    from harness.adapters.ingestion_sources import LocalSourceStore
    from harness.adapters.ingestion_text import IsolatedTextParser
    from harness.ingestion.pipeline import IngestionService

    from harness.adapters.development_auth import LiveDevelopmentAuthenticator

    identity_ref = values.get("HARNESS_DEV_IDENTITY_FILE", "")
    if not identity_ref or values.get("HARNESS_AUTH_MODE", "development") != "development":
        raise RuntimeFault("INGESTION_AUTH_CONFIGURATION_REQUIRED", 503)
    auth = LiveDevelopmentAuthenticator(
        root / identity_ref, values.get("HARNESS_ENVIRONMENT", "development"),
    )
    token = values.get("HARNESS_OPERATOR_TOKEN") or getpass.getpass("知识管理Token（不回显）：")
    principal = auth.authenticate("Bearer " + token)
    repository = PostgresIngestionRepository(dsn, auth.current)
    if args.command == "status":
        return (await repository.read(principal, args.job_id)).model_dump(mode="json")
    kb_id = KnowledgeBaseInput(knowledge_base_id=args.kb_id).knowledge_base_id
    if args.command == "create-kb":
        await repository.create_knowledge_base(principal, kb_id)
        return {"knowledge_base_id": kb_id}
    upload = Upload(request_id=args.request_id, document_id=args.document_id,
                    filename=args.filename, media_type=args.media_type)
    body = await asyncio.to_thread(
        read_staged, root / config.staging_directory, upload.filename, config.profile.max_upload_bytes,
    )
    service = IngestionService(repository, LocalSourceStore(root / config.source_directory),
                               IsolatedTextParser(), config.profile)
    return (await service.upload(principal, kb_id, upload, body)).model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description="本机受限知识入库管理")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-kb")
    create.add_argument("--kb-id", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--kb-id", required=True)
    upload.add_argument("--request-id", required=True)
    upload.add_argument("--document-id", required=True)
    upload.add_argument("--filename", required=True)
    upload.add_argument("--media-type", choices=("text/plain", "text/markdown"), required=True)
    status = commands.add_parser("status")
    status.add_argument("--job-id", required=True)
    args = parser.parse_args()
    from harness.bootstrap.event_loop import run

    try:
        result = run(execute(Path.cwd().resolve(), args))
    except RuntimeFault as exc:
        print(json.dumps({"error_code": exc.code}))
        raise SystemExit(1) from None
    except (OSError, ValueError):
        print(json.dumps({"error_code": "INGESTION_LOCAL_CONFIGURATION_OR_IO_FAILED"}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
