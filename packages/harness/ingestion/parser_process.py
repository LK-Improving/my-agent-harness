"""固定纯文本子进程入口；只从stdin读有界字节，不接受文件/URL/插件。"""

import json
import sys

from harness.contracts.ingestion import IngestionProfile
from harness.contracts.runtime import RuntimeFault
from harness.ingestion.text import parse_text


def main() -> None:
    profile = IngestionProfile.model_validate_json(sys.argv[1])
    body = sys.stdin.buffer.read(profile.max_upload_bytes + 1)
    data: dict[str, object]
    try:
        chunks = parse_text(body, profile)
        data = {"chunks": [chunk.model_dump(mode="json") for chunk in chunks]}
    except RuntimeFault as exc:
        data = {"error_code": exc.code}
    sys.stdout.buffer.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
