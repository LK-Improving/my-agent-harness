"""确定线程的官方checkpoint与固定私有trace副本；不递归删除或接受客户端路径。"""

import asyncio
import re
import stat
from pathlib import Path
from typing import Protocol

from harness.adapters.postgres.connection import PostgresUnitOfWork
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot


class ThreadEraser(Protocol):
    async def adelete_thread(self, thread_id: str) -> None: ...


class ManagedTraceFiles:
    def __init__(self, project_root: Path) -> None:
        self.project = project_root.resolve()
        self.root = self.project / ".runtime" / "traces"

    def _path(self, thread_id: str) -> Path:
        if not re.fullmatch(r"harness-[a-f0-9]{64}", thread_id):
            raise RuntimeFault("CLEANUP_THREAD_INVALID", 422)
        if (self.project / ".runtime").is_symlink() or self.root.is_symlink():
            raise RuntimeFault("TRACE_CLEANUP_PATH_UNSAFE", 409)
        if self.root.resolve() != self.root or not self.root.is_relative_to(self.project):
            raise RuntimeFault("TRACE_CLEANUP_PATH_UNSAFE", 409)
        return self.root / (thread_id + ".jsonl")

    def _absent(self, threads: tuple[str, ...]) -> bool:
        absent = True
        for thread in threads:
            path = self._path(thread)
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeFault("TRACE_CLEANUP_FILE_UNSAFE", 409)
            absent = False
        return absent

    async def absent(self, threads: tuple[str, ...]) -> bool:
        return await asyncio.to_thread(self._absent, threads)

    def _erase(self, threads: tuple[str, ...]) -> None:
        for thread in threads:
            path = self._path(thread)
            try:
                before = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RuntimeFault("TRACE_CLEANUP_FILE_UNSAFE", 409)
            current = path.lstat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
                raise RuntimeFault("TRACE_CLEANUP_FILE_CHANGED", 409)
            path.unlink(missing_ok=True)

    async def erase(self, threads: tuple[str, ...]) -> None:
        await asyncio.to_thread(self._erase, threads)


class OfficialCheckpointCopies:
    def __init__(self, checkpoint: ThreadEraser, dsn: str, traces: ManagedTraceFiles) -> None:
        self.checkpoint, self.uow, self.traces = checkpoint, PostgresUnitOfWork(dsn), traces

    @staticmethod
    def _threads(target: JsonSnapshot) -> tuple[str, ...]:
        threads = target.as_object().get("threads")
        if not isinstance(threads, list) or len(threads) > 1000 or any(
            not isinstance(thread, str) or not re.fullmatch(r"harness-[a-f0-9]{64}", thread) for thread in threads):
            raise RuntimeFault("CLEANUP_TARGET_INVALID", 422)
        if target.as_object().get("kind") not in {"session", "trace"}:
            raise RuntimeFault("CLEANUP_TARGET_INVALID", 422)
        return tuple(threads)

    async def erase(self, target: JsonSnapshot) -> None:
        threads = self._threads(target)
        # 重试只能来自显式核对后的同一不可变目标；官方删除包含全部namespace与写入。
        if target.as_object()["kind"] == "session":
            for thread in threads:
                await self.checkpoint.adelete_thread(thread)
        await self.traces.erase(threads)

    async def absent(self, target: JsonSnapshot) -> bool:
        threads = self._threads(target)
        if not await self.traces.absent(threads):
            return False
        if target.as_object()["kind"] == "session":
            # alist空不等于没有孤立blob/write；只读核对锁定版本的官方三张表。
            async with self.uow.transaction() as conn:
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    if await (await conn.execute(f"SELECT 1 FROM harness_checkpoint.{table} WHERE thread_id=ANY(%s) LIMIT 1",
                                                (list(threads),))).fetchone():
                        return False
        return True
