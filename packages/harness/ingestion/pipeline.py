"""入库编排；稳定请求去重，失败/中断不隐式重放，外部I/O不持数据库锁。"""

import asyncio
import hashlib
from pathlib import PurePosixPath

import anyio

from harness.contracts.hashing import canonical_hash
from harness.contracts.ingestion import (
    DocumentParser, IngestionJob, IngestionProfile, IngestionRepository,
    SourceRef, SourceStore, Upload,
)
from harness.contracts.runtime import Principal, RuntimeFault


def validate_upload(upload: Upload, body: bytes, profile: IngestionProfile) -> None:
    name = upload.filename
    suffix = PurePosixPath(name).suffix.lower()
    expected = {"text/plain": ".txt", "text/markdown": ".md"}[upload.media_type]
    if (name in {".", ".."} or any(c in name for c in '/\\:\x00')
        or any(ord(c) < 32 for c in name) or suffix != expected):
        raise RuntimeFault("INGESTION_UNSUPPORTED_FILENAME", 422)
    if not body or len(body) > profile.max_upload_bytes:
        raise RuntimeFault("INGESTION_UPLOAD_SIZE_EXCEEDED", 413)


class IngestionService:
    def __init__(self, repository: IngestionRepository, sources: SourceStore,
                 parser: DocumentParser, profile: IngestionProfile) -> None:
        self.repository, self.sources, self.parser, self.profile = repository, sources, parser, profile

    async def upload(self, principal: Principal, kb_id: str, upload: Upload,
                     body: bytes) -> IngestionJob:
        upload = Upload.model_validate_json(upload.model_dump_json())
        validate_upload(upload, body, self.profile)
        digest = hashlib.sha256(body).hexdigest()
        source = SourceRef(object_id=canonical_hash({
            "tenant": principal.tenant_id, "application": principal.application_id,
            "kb": kb_id, "sha256": digest,
        }), sha256=digest, size_bytes=len(body))
        job = await self.repository.begin(principal, kb_id, upload, source, self.profile)
        if job.state != "queued":
            return job
        # CAS在try外；并发败者不能把赢家任务改成failed。
        await self.repository.advance(principal, job.job_id, "queued", "storing")
        try:
            await self.sources.put(source, body)
            await self.repository.advance(principal, job.job_id, "storing", "parsing")
            chunks = await self.parser.parse(body, self.profile)
            await self.repository.advance(principal, job.job_id, "parsing", "persisting")
            return await self.repository.finish(principal, job.job_id, chunks)
        except BaseException as exc:
            code = (exc.code if isinstance(exc, RuntimeFault) else
                    "INGESTION_CANCELLED" if isinstance(exc, asyncio.CancelledError) else
                    "INGESTION_FAILED")
            # 保存失败不能覆盖原始取消；落库失败则保留中间态，禁止自动接管。
            with anyio.CancelScope(shield=True):
                try:
                    async with asyncio.timeout(5):
                        await self.repository.fail(principal, job.job_id, code)
                except Exception:
                    pass
            raise
