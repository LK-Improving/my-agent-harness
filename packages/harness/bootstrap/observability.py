"""显式开启时才装配可选 Langfuse 导出。"""

import os
from logging import getLogger
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from harness.adapters.langfuse_observations import LangfuseObservationSink

logger = getLogger(__name__)


def langfuse_sink(root: Path, external_trace_copies: bool) -> LangfuseObservationSink | None:
    values = {**dotenv_values(root / ".env"), **os.environ}
    if values.get("HARNESS_LANGFUSE_ENABLED") != "true":
        return None
    origin = values.get("HARNESS_LANGFUSE_ORIGIN") or ""
    public = values.get("HARNESS_LANGFUSE_PUBLIC_KEY") or ""
    secret = values.get("HARNESS_LANGFUSE_SECRET_KEY") or ""
    try:
        url = urlsplit(origin)
        local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (not external_trace_copies or not public or not secret
                or not (url.scheme == "https" or local_http)
                or not url.hostname or url.username is not None or url.password is not None
                or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("追踪配置不完整")
        return LangfuseObservationSink(origin.rstrip("/") + "/api/public/otel/v1/traces", public, secret)
    except (ValueError, httpx.InvalidURL):
        logger.warning("Langfuse 追踪配置无效，继续使用无操作接收端")
        return None
