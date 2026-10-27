"""知识撤权与显式清理端口；状态不代表未处理的副本已物理删除。"""

from typing import Literal, Protocol

from pydantic import AwareDatetime, Field

from harness.contracts.runtime import Principal
from harness.contracts.providers import Digest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier


class DeleteDocument(Contract):
    request_id: Identifier
    document_id: Identifier
    retain_until: AwareDatetime
    source_store_ref: Digest


class CleanupJob(Contract):
    job_id: Identifier
    knowledge_base_id: Identifier
    document_id: Identifier
    state: Literal["pending", "blocked", "manual", "succeeded"]
    succeeded_steps: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    retained_sources: int = Field(default=0, ge=0)
    reason: str | None = None


class CleanupStep(Contract):
    step_id: Identifier
    job_id: Identifier
    kind: Literal["index", "source", "postgres"]
    attempt: int = Field(ge=1, le=3)
    target: JsonSnapshot


class CleanupRepository(Protocol):
    async def revoke(self, actor: Principal, kb_id: str, document_id: str) -> None: ...
    async def request_delete(self, actor: Principal, kb_id: str, request: DeleteDocument) -> CleanupJob: ...
    async def read(self, actor: Principal, job_id: str) -> CleanupJob: ...
    async def claim(self, actor: Principal, job_id: str) -> CleanupStep | None: ...
    async def finish(self, actor: Principal, step: CleanupStep, *, retained: bool = False) -> None: ...
    async def fail(self, actor: Principal, step: CleanupStep, code: str) -> None: ...


class SourceCleanup(Protocol):
    async def delete(self, target: JsonSnapshot) -> None: ...


class IndexCleanup(Protocol):
    async def delete(self, target: JsonSnapshot) -> None: ...
