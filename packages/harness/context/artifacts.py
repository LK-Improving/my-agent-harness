"""固定JSON文本与Unicode字符分页；不把字节偏移误作字符偏移。"""

import hashlib

from pydantic import JsonValue

from harness.contracts.artifacts import ArtifactProfile
from harness.contracts.hashing import canonical_json
from harness.contracts.runtime import RuntimeFault


def artifact_text(data: JsonValue) -> tuple[str, str, str]:
    text = data if isinstance(data, str) else canonical_json(data)
    media_type = "text/plain" if isinstance(data, str) else "application/json"
    return text, media_type, hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_text(text: str, offset: int, limit: int, profile: ArtifactProfile) -> tuple[str, int | None]:
    if offset < 0 or offset > len(text) or limit < 1 or limit > profile.page_chars:
        raise RuntimeFault("ARTIFACT_PAGE_INVALID", 422)
    end = min(len(text), offset + limit)
    # UTF-8只在字符边界截取；极端多字节文本仍不超过页字节上限。
    page = text[offset:end].encode("utf-8")[:profile.page_bytes].decode("utf-8", errors="ignore")
    end = offset + len(page)
    return page, end if end < len(text) else None
