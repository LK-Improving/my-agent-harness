"""纯字节资产校验；文件访问由适配器负责。"""

import hashlib
import json

from harness.contracts.evaluation import (
    EvaluationCase, EvaluationFreeze, FrozenFile, GatePolicy, JudgePolicy, MetricPolicy,
)
from harness.contracts.hashing import canonical_hash

REQUIRED = {"development.jsonl", "holdout.jsonl", "judge.json", "metrics.json", "gates.json"}


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("JSON 键重复")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("JSON 非有限数值")


def decode(raw: bytes) -> object:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    # 同时拒绝 1e999 溢出为 infinity 及规范化后重复键。
    canonical_hash(value)
    return value


def cases(raw: bytes) -> tuple[EvaluationCase, ...]:
    rows = tuple(EvaluationCase.model_validate(decode(line))
                 for line in raw.splitlines() if line.strip())
    if not rows or len(rows) > 1000:
        raise ValueError("每个 split 必须包含 1～1000 条用例")
    return rows


def freeze_assets(blobs: dict[str, bytes]) -> EvaluationFreeze:
    if not REQUIRED <= blobs.keys() or len(blobs) > 32 or "freeze.json" in blobs:
        raise ValueError("冻结包缺必需文件或文件过多")
    files = tuple(FrozenFile(name=name, sha256=hashlib.sha256(raw).hexdigest(),
                             size_bytes=len(raw)) for name, raw in sorted(blobs.items()))
    for name, raw in blobs.items():
        if name.endswith(".jsonl"):
            for line in raw.splitlines():
                if line.strip():
                    decode(line)
        else:
            decode(raw)
    development, holdout = cases(blobs["development.jsonl"]), cases(blobs["holdout.jsonl"])
    all_cases = development + holdout
    if len({row.input_hash() for row in all_cases}) != len(all_cases):
        raise ValueError("开发集与 holdout 不允许重复输入")
    JudgePolicy.model_validate(decode(blobs["judge.json"]))
    MetricPolicy.model_validate(decode(blobs["metrics.json"]))
    GatePolicy.model_validate(decode(blobs["gates.json"]))
    data = [item.model_dump(mode="json") for item in files
            if item.name not in {"judge.json", "metrics.json", "gates.json"}]
    return EvaluationFreeze(files=files, dataset_hash=canonical_hash(data),
                            development_ids=tuple(row.case_id for row in development),
                            holdout_ids=tuple(row.case_id for row in holdout))
