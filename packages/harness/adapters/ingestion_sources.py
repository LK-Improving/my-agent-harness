"""本机私有原文对象存储；对象键不含用户文件名，原子安装且不覆盖既有对象。"""

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path

import anyio

from harness.contracts.ingestion import SourceRef
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.hashing import canonical_hash


class LocalSourceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def storage_ref(self) -> str:
        return canonical_hash({"kind": "private-local-source@1", "root": str(self.root)})

    async def put(self, ref: SourceRef, body: bytes) -> None:
        ref = SourceRef.model_validate_json(ref.model_dump_json())
        if hashlib.sha256(body).hexdigest() != ref.sha256 or len(body) != ref.size_bytes:
            raise RuntimeFault("INGESTION_SOURCE_HASH_MISMATCH", 409)
        task = asyncio.create_task(asyncio.to_thread(self._put, ref, body))
        try:
            await asyncio.shield(task)
        except BaseException:
            # 必须排空写文件线程，不能先标失败再由旧线程重新生成已清理的原文。
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            raise

    async def delete(self, target: JsonSnapshot) -> None:
        if target.as_object().get("source_store_ref") != self.storage_ref():
            raise RuntimeFault("CLEANUP_SOURCE_STORE_CHANGED", 409)
        ref = SourceRef.model_validate(target.as_object()["source"])
        task = asyncio.create_task(asyncio.to_thread(self._delete, ref))
        try:
            await asyncio.shield(task)
        except BaseException:
            # 取消时先排空文件线程，再允许持久失败或下一轮重试。
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            raise

    def _delete(self, ref: SourceRef) -> None:
        target = self.root / ref.object_id
        if target.is_symlink() or target.resolve().parent != self.root:
            raise RuntimeFault("CLEANUP_SOURCE_PATH_INVALID", 409)
        if not target.exists():
            return
        with target.open("rb") as stream:
            body = stream.read(ref.size_bytes + 1)
        if len(body) != ref.size_bytes or hashlib.sha256(body).hexdigest() != ref.sha256:
            raise RuntimeFault("CLEANUP_SOURCE_HASH_MISMATCH", 409)
        target.unlink(missing_ok=True)

    def _put(self, ref: SourceRef, body: bytes) -> None:
        # root由受信任运维配置提供，必须位于私有、无非信任写入者的目录。
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = self.root / ref.object_id
        if target.is_symlink():
            raise RuntimeFault("INGESTION_SOURCE_PATH_INVALID", 409)
        descriptor, temporary = tempfile.mkstemp(prefix="upload-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                with target.open("rb") as stream:
                    existing = stream.read(ref.size_bytes + 1)
                if existing != body:
                    raise RuntimeFault("INGESTION_SOURCE_CONFLICT", 409) from None
        finally:
            Path(temporary).unlink(missing_ok=True)
