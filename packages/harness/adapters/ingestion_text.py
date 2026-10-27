"""固定文本解析器通过受控线程启动，兼容Windows的PG Selector事件循环。"""

import asyncio
import json
import subprocess
import sys
import threading
import time

import anyio
from pydantic import TypeAdapter

from harness.contracts.ingestion import Chunk, IngestionProfile
from harness.contracts.runtime import RuntimeFault


class IsolatedTextParser:
    async def parse(self, body: bytes, profile: IngestionProfile) -> tuple[Chunk, ...]:
        if not body or len(body) > profile.max_upload_bytes:
            raise RuntimeFault("INGESTION_UPLOAD_SIZE_EXCEEDED", 413)
        cancelled = threading.Event()
        task = asyncio.create_task(asyncio.to_thread(self._parse, body, profile, cancelled))
        try:
            return await asyncio.shield(task)
        except BaseException:
            cancelled.set()
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            raise

    @staticmethod
    def _parse(body: bytes, profile: IngestionProfile,
               cancelled: threading.Event) -> tuple[Chunk, ...]:
        process = subprocess.Popen(
            [sys.executable, "-m", "harness.ingestion.parser_process", profile.model_dump_json()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        deadline = time.monotonic() + profile.parse_timeout_seconds
        pending_input: bytes | None = body
        try:
            while True:
                if cancelled.is_set():
                    raise RuntimeFault("INGESTION_CANCELLED", 409)
                if time.monotonic() >= deadline:
                    raise RuntimeFault("INGESTION_PARSE_TIMEOUT", 422)
                try:
                    output, _ = process.communicate(input=pending_input, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
            # 固定解析器总正文最多100万字符，JSON和切片元数据均有确定上界。
            if process.returncode != 0 or len(output) > 16000000:
                raise RuntimeFault("INGESTION_PARSER_FAILED", 422)
            data = json.loads(output)
            if "error_code" in data:
                raise RuntimeFault(data["error_code"], 422)
            return TypeAdapter(tuple[Chunk, ...]).validate_python(data["chunks"])
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
