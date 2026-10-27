"""显式、有界清理，不占运行Worker槽，不启动后台循环或重放模型调用。"""

import asyncio

import anyio

from harness.contracts.knowledge_lifecycle import CleanupJob, CleanupRepository, IndexCleanup, SourceCleanup
from harness.contracts.runtime import Principal, RuntimeFault


class KnowledgeLifecycle:
    def __init__(self, repository: CleanupRepository, sources: SourceCleanup, indexes: IndexCleanup) -> None:
        self.repository, self.sources, self.indexes = repository, sources, indexes

    async def advance(self, actor: Principal, job_id: str, *, max_steps: int = 16) -> CleanupJob:
        if not 1 <= max_steps <= 32:
            raise RuntimeFault("CLEANUP_STEP_LIMIT_INVALID", 422)
        for _ in range(max_steps):
            step = await self.repository.claim(actor, job_id)
            if step is None:
                break
            try:
                # PG步骤在finish事务完成；所有外部删除均在事务外执行。
                if step.kind == "index":
                    await self.indexes.delete(step.target)
                elif step.kind == "source":
                    await self.sources.delete(step.target)
                await self.repository.finish(actor, step)
            except BaseException as exc:
                code = exc.code if isinstance(exc, RuntimeFault) else (
                    "CLEANUP_CANCELLED" if isinstance(exc, asyncio.CancelledError) else "CLEANUP_IO_FAILED")
                with anyio.CancelScope(shield=True):
                    try:
                        async with asyncio.timeout(5):
                            await self.repository.fail(actor, step, code)
                    except Exception:
                        # 记录失败保留running；不会按租约过期自动接管。
                        pass
                raise
        return await self.repository.read(actor, job_id)
