"""正式入库的纯契约；完成仅表示正文与切片持久化，不表示索引可查询。"""

from typing import Literal, Protocol

from pydantic import Field

from harness.contracts.providers import Digest
from harness.contracts.runtime import Principal
from harness.contracts.specs import Contract, Identifier


class IngestionProfile(Contract):
    version: Literal["text-ingestion@1"] = "text-ingestion@1"
    max_upload_bytes: int = Field(default=10485760, ge=1, le=10485760, strict=True)
    max_characters: int = Field(default=1000000, ge=1, le=1000000, strict=True)
    chunk_characters: int = Field(default=1600, ge=100, le=8000, strict=True)
    max_chunks: int = Field(default=1000, ge=1, le=1000, strict=True)
    parse_timeout_seconds: int = Field(default=60, ge=1, le=60, strict=True)


class Upload(Contract):
    request_id: Identifier
    document_id: Identifier
    filename: str = Field(min_length=1, max_length=240)
    media_type: Literal["text/plain", "text/markdown"]


class SourceRef(Contract):
    object_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    sha256: Digest
    size_bytes: int = Field(ge=1, le=10485760, strict=True)


class Chunk(Contract):
    ordinal: int = Field(ge=0, le=999, strict=True)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    body: str = Field(min_length=1, max_length=8000, repr=False)
    body_hash: Digest


class IngestionJob(Contract):
    job_id: Identifier
    knowledge_base_id: Identifier
    document_id: Identifier
    version_id: Identifier
    state: Literal["queued", "storing", "parsing", "persisting", "succeeded", "failed"]
    chunks_written: int = Field(default=0, ge=0)
    error_code: str | None = None


class SourceStore(Protocol):
    async def put(self, ref: SourceRef, body: bytes) -> None: ...


class DocumentParser(Protocol):
    async def parse(self, body: bytes, profile: IngestionProfile) -> tuple[Chunk, ...]: ...


class IngestionRepository(Protocol):
    async def create_knowledge_base(self, principal: Principal, kb_id: str) -> None: ...
    async def begin(self, principal: Principal, kb_id: str, upload: Upload,
                    source: SourceRef, profile: IngestionProfile) -> IngestionJob: ...
    async def read(self, principal: Principal, job_id: str) -> IngestionJob: ...
    async def advance(self, principal: Principal, job_id: str, expected: str,
                      target: str) -> None: ...
    async def finish(self, principal: Principal, job_id: str,
                     chunks: tuple[Chunk, ...]) -> IngestionJob: ...
    async def fail(self, principal: Principal, job_id: str, code: str) -> None: ...
