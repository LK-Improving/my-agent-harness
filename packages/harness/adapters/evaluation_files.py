"""本地冻结包与追加式采集证据；不加载凭据。"""

import json
from pathlib import Path

from harness.contracts.evaluation import EvaluationFreeze
from harness.evaluation.assets import freeze_assets


def write_freeze(destination: Path, blobs: dict[str, bytes]) -> EvaluationFreeze:
    frozen = freeze_assets(blobs)
    destination.mkdir(parents=True, exist_ok=False)
    for item in frozen.files:
        (destination / item.name).write_bytes(blobs[item.name])
    # lock 最后落盘，部分写入不能视作有效冻结包。
    (destination / "freeze.json").write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
    return frozen


def bounded_read(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(16777217)
    if len(raw) > 16777216:
        raise ValueError("资产超过 16 MiB")
    return raw


def read_freeze(directory: Path) -> tuple[EvaluationFreeze, dict[str, bytes]]:
    frozen = EvaluationFreeze.model_validate_json(bounded_read(directory / "freeze.json"))
    if len(frozen.files) > 32:
        raise ValueError("冻结资产文件过多")
    root = directory.resolve()
    blobs: dict[str, bytes] = {}
    for item in frozen.files:
        path = (root / item.name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("冻结资产路径逃逸")
        blobs[item.name] = bounded_read(path)
    if freeze_assets(blobs) != frozen:
        raise ValueError("冻结资产发生漂移")
    return frozen, blobs


def append_record(path: Path, record: dict[str, object]) -> None:
    import os

    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
