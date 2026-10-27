"""固定UTF-8解析算法；不执行Markdown、不读取网络或引用文件。"""

from harness.contracts.hashing import canonical_hash
from harness.contracts.ingestion import Chunk, IngestionProfile
from harness.contracts.runtime import RuntimeFault


def parse_text(body: bytes, profile: IngestionProfile) -> tuple[Chunk, ...]:
    if len(body) > profile.max_upload_bytes:
        raise RuntimeFault("INGESTION_UPLOAD_SIZE_EXCEEDED", 413)
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise RuntimeFault("INGESTION_INVALID_UTF8", 422) from None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if (not text.strip() or "\x00" in text
        or any(ord(c) < 32 and c not in "\n\t" for c in text)):
        raise RuntimeFault("INGESTION_INVALID_TEXT", 422)
    count = (len(text) + profile.chunk_characters - 1) // profile.chunk_characters
    if len(text) > profile.max_characters or count > profile.max_chunks:
        raise RuntimeFault("INGESTION_PARSE_LIMIT_EXCEEDED", 413)
    return tuple(Chunk(ordinal=i, start=start, end=min(start + profile.chunk_characters, len(text)),
                       body=text[start:start + profile.chunk_characters],
                       body_hash=canonical_hash(text[start:start + profile.chunk_characters]))
                 for i, start in enumerate(range(0, len(text), profile.chunk_characters)))
