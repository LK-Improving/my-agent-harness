"""M5 冻结资产边界定义；本轮未执行。"""

import json
from pathlib import Path

import pytest

from harness.adapters.evaluation_files import read_freeze, write_freeze
from harness.evaluation.assets import cases, freeze_assets

ROOT = Path(__file__).resolve().parents[2]


def blobs():
    return {path.name: path.read_bytes() for path in (ROOT / "evals/m5").iterdir()
            if path.name != "freeze.json"}


def test_frozen_sources_and_holdout_preserve_all_original_queries():
    frozen, assets = read_freeze(ROOT / "evals/m5")
    assert frozen == freeze_assets(assets)
    assert len(frozen.development_ids) == 12 and len(frozen.holdout_ids) == 4
    assert frozen.holdout_for_tuning is False
    for split, source in (("development", "queries"), ("holdout", "holdout")):
        original = [json.loads(line) for line in (ROOT / f"evals/m0/{source}.jsonl").read_text("utf-8").splitlines()]
        rows = cases(assets[f"{split}.jsonl"])
        assert [row.message for row in rows] == [row["query"] for row in original]
        assert [row.expected["source_id"] for row in rows] == [row["id"] for row in original]


@pytest.mark.parametrize("name", ["development.jsonl", "holdout.jsonl", "judge.json", "metrics.json", "gates.json", "corpus.jsonl"])
def test_any_byte_drift_is_rejected(tmp_path, name):
    write_freeze(tmp_path / "frozen", blobs())
    path = tmp_path / "frozen" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="漂移"):
        read_freeze(tmp_path / "frozen")


def test_split_duplicate_input_and_identity_rejected():
    assets = blobs()
    assets["holdout.jsonl"] = assets["development.jsonl"]
    with pytest.raises(ValueError):
        freeze_assets(assets)


@pytest.mark.parametrize("raw", [b"", b'{"case_id":"x","case_id":"y"}', b'{"score":NaN}'])
def test_empty_invalid_or_nonfinite_cases_rejected(raw):
    with pytest.raises(ValueError):
        cases(raw)


def test_no_path_escape_or_overwrite(tmp_path):
    assets = blobs()
    assets["../escape.json"] = b"{}"
    with pytest.raises(ValueError):
        write_freeze(tmp_path / "invalid", assets)
    assert not (tmp_path / "invalid").exists()
    write_freeze(tmp_path / "valid", blobs())
    with pytest.raises(FileExistsError):
        write_freeze(tmp_path / "valid", blobs())


def test_unknown_judge_and_thresholds_remain_explicit():
    _, assets = read_freeze(ROOT / "evals/m5")
    judge = json.loads(assets["judge.json"])
    metrics = json.loads(assets["metrics.json"])
    assert judge["required"] and judge["implementation"] is None
    assert judge["missing_outcome"] == "incomplete"
    assert not metrics["thresholds_confirmed"] and not metrics["thresholds"]
