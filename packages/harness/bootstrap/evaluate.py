"""显式冻结/采集 CLI；不直接装配图或供应商。"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from harness.adapters.evaluation_files import append_record, bounded_read, read_freeze, write_freeze
from harness.adapters.evaluation_http import HTTPEvaluationAPI
from harness.contracts.evaluation import GatePolicy
from harness.contracts.evaluation_contexts import EvaluationLimits
from harness.contracts.evaluation_gates import JudgeInput
from harness.contracts.evaluation_observations import ObservationBundle
from harness.contracts.hashing import canonical_json
from harness.contracts.release import ReleaseManifest
from harness.contracts.specs import Identifier
from harness.evaluation.assets import REQUIRED, cases, decode
from harness.evaluation.collector import collect
from harness.evaluation.judge import prepare_judge_input
from harness.evaluation.judge_collector import collect_judges
from harness.evaluation.observations import merge_observations
from harness.evaluation.reports import BaselineInput, build_report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="冻结评测资产与真实 API/SSE 采集")
    commands = result.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--resource", action="append", default=[], metavar="NAME=PATH")
    run = commands.add_parser("collect")
    run.add_argument("--freeze", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--token-env", default="HARNESS_EVALUATION_TOKEN")
    run.add_argument("--split", choices=("development", "holdout"), required=True)
    run.add_argument("--batch-id", required=True)
    run.add_argument("--scope", action="append", default=[])
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout-seconds", type=float, default=180)
    run.add_argument("--max-sessions", type=int, default=64)
    run.add_argument("--max-runs", type=int, default=128)
    run.add_argument("--context-lifetime-seconds", type=int, default=3600)
    gate = commands.add_parser("gate")
    gate.add_argument("--freeze", type=Path, required=True)
    gate.add_argument("--collection", type=Path, required=True)
    gate.add_argument("--split", choices=("development", "holdout"), required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--candidate", type=Path)
    merge = commands.add_parser("merge-observations")
    merge.add_argument("--freeze", type=Path, required=True)
    merge.add_argument("--collection", type=Path, required=True)
    merge.add_argument("--candidate", type=Path, required=True)
    merge.add_argument("--observations", type=Path, required=True)
    merge.add_argument("--split", choices=("development", "holdout"), required=True)
    merge.add_argument("--output", type=Path, required=True)
    judge = commands.add_parser("prepare-judge")
    judge.add_argument("--freeze", type=Path, required=True)
    judge.add_argument("--collection", type=Path, required=True)
    judge.add_argument("--candidate", type=Path, required=True)
    judge.add_argument("--split", choices=("development", "holdout"), required=True)
    judge.add_argument("--case-id", required=True)
    judge.add_argument("--material-source-ref", required=True)
    judge.add_argument("--material", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("collect-judge")
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--collection", type=Path, required=True)
    score.add_argument("--candidate", type=Path, required=True)
    score.add_argument("--judge-candidate", type=Path, required=True)
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--split", choices=("development", "holdout"), required=True)
    score.add_argument("--batch-id", required=True)
    score.add_argument("--base-url", required=True)
    score.add_argument("--token-env", default="HARNESS_EVALUATION_TOKEN")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--timeout-seconds", type=float, default=180)
    score.add_argument("--max-sessions", type=int, default=64)
    score.add_argument("--max-runs", type=int, default=128)
    score.add_argument("--context-lifetime-seconds", type=int, default=3600)
    material = commands.add_parser("fetch-judge-material")
    material.add_argument("--run-id", required=True)
    material.add_argument("--base-url", required=True)
    material.add_argument("--token-env", default="HARNESS_EVALUATION_TOKEN")
    material.add_argument("--output", type=Path, required=True)
    for command in (run, gate):
        command.add_argument("--baseline-collection", type=Path)
        command.add_argument("--baseline-freeze", type=Path)
        command.add_argument("--baseline-report", type=Path)
    return result


def http_token(args: argparse.Namespace) -> str:
    token = os.environ.get(args.token_env)
    if not token or "\r" in token or "\n" in token:
        raise ValueError("缺少有效评测身份令牌")
    url = urlsplit(args.base_url)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username or url.password or url.query or url.fragment
            or url.path not in {"", "/"}):
        raise ValueError("API 地址必须为无身份信息的 HTTP(S) origin")
    return token


def local_output(path: Path) -> Path:
    output = path.resolve()
    root = (Path.cwd() / ".agents" / "Documents").resolve()
    if not output.is_relative_to(root):
        raise ValueError("评测材料必须保存于本项目 .agents/Documents 内")
    return output


def baseline_input(args: argparse.Namespace) -> BaselineInput | None:
    paths = (args.baseline_collection, args.baseline_freeze, args.baseline_report)
    if not any(paths):
        return None
    if not all(paths):
        raise ValueError("基线必须同时提供collection、freeze及report")
    frozen, blobs = read_freeze(args.baseline_freeze)
    collection = decode(bounded_read(args.baseline_collection))
    report = decode(bounded_read(args.baseline_report))
    if not isinstance(collection, dict) or not isinstance(report, dict):
        raise ValueError("基线采集与报告必须为JSON对象")
    return BaselineInput(collection, frozen, blobs, report)


async def run_collection(args: argparse.Namespace) -> int:
    from pydantic import TypeAdapter

    batch_id = TypeAdapter(Identifier).validate_python(args.batch_id)
    baseline = await asyncio.to_thread(baseline_input, args)
    frozen, blobs = await asyncio.to_thread(read_freeze, args.freeze)
    candidate = ReleaseManifest.model_validate_json(await asyncio.to_thread(bounded_read, args.candidate))
    gate = GatePolicy.model_validate(decode(blobs["gates.json"]))
    rows = cases(blobs[f"{args.split}.jsonl"])
    limits = EvaluationLimits(max_sessions=args.max_sessions, max_runs=args.max_runs)
    if (len(rows) > limits.max_sessions or len(rows) > limits.max_runs
            or not 1 <= args.context_lifetime_seconds <= 86400):
        raise ValueError("选定用例数量超出评测额度或有效期非法")
    token = http_token(args)
    if args.timeout_seconds <= 0 or args.timeout_seconds > 3600:
        raise ValueError("采集超时必须在 0～3600 秒内")
    # 证据只放本地文档区；新目录防止覆盖旧批次或自动重跑未知创建。
    output = local_output(args.output)
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    async with httpx.AsyncClient(base_url=args.base_url,
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=httpx.Timeout(30, read=min(args.timeout_seconds, 30)),
                                follow_redirects=False, trust_env=False) as client:
        report = await collect(
            HTTPEvaluationAPI(client, timeout_seconds=args.timeout_seconds), frozen, rows,
            candidate, gate.version, batch_id, frozenset(args.scope),
            lambda record: append_record(output / "journal.jsonl", record),
            limits=limits, lifetime_seconds=args.context_lifetime_seconds,
        )
    report["split"] = args.split
    report["frozen_files"] = [item.model_dump(mode="json") for item in frozen.files]
    await asyncio.to_thread((output / "collection.json").write_text,
                            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                            encoding="utf-8")
    report = build_report(report, frozen, blobs, args.split, candidate, baseline)
    await asyncio.to_thread((output / "report.json").write_text,
                            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                            encoding="utf-8")
    if report["outcome"] == "passed":
        return 0
    if report["outcome"] == "failed":
        return 1
    return 2


def run_gates(args: argparse.Namespace) -> int:
    frozen, blobs = read_freeze(args.freeze)
    collection = decode(bounded_read(args.collection))
    if not isinstance(collection, dict):
        raise ValueError("采集报告必须为JSON对象")
    # 门禁报告的hash绑定原采集文件内容；不回写输入或覆盖旧门禁报告。
    candidate = (ReleaseManifest.model_validate_json(bounded_read(args.candidate)) if args.candidate
                 else ReleaseManifest.model_validate(collection.get("candidate_manifest")))
    report = build_report(collection, frozen, blobs, args.split, candidate, baseline_input(args))
    output = args.output.resolve()
    root = (Path.cwd() / ".agents" / "Documents").resolve()
    if not output.is_relative_to(root):
        raise ValueError("门禁证据必须保存于本项目.agents/Documents内")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if report["outcome"] == "passed":
        return 0
    if report["outcome"] == "failed":
        return 1
    return 2


def run_merge(args: argparse.Namespace) -> None:
    frozen, blobs = read_freeze(args.freeze)
    collection = decode(bounded_read(args.collection))
    if not isinstance(collection, dict):
        raise ValueError("采集报告必须为JSON对象")
    candidate = ReleaseManifest.model_validate_json(bounded_read(args.candidate))
    bundle = ObservationBundle.model_validate(decode(bounded_read(args.observations)))
    merged = merge_observations(collection, frozen, blobs, args.split, candidate, bundle)
    output = args.output.resolve()
    root = (Path.cwd() / ".agents" / "Documents").resolve()
    if not output.is_relative_to(root):
        raise ValueError("合并证据必须保存于本项目.agents/Documents内")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False))


def run_prepare_judge(args: argparse.Namespace) -> None:
    frozen, blobs = read_freeze(args.freeze)
    collection = decode(bounded_read(args.collection))
    if not isinstance(collection, dict):
        raise ValueError("采集报告必须为JSON对象")
    candidate = ReleaseManifest.model_validate_json(bounded_read(args.candidate))
    request = prepare_judge_input(collection, frozen, blobs, args.split, candidate, args.case_id,
                                  args.material_source_ref, decode(bounded_read(args.material)))
    output = args.output.resolve()
    root = (Path.cwd() / ".agents" / "Documents").resolve()
    if not output.is_relative_to(root):
        raise ValueError("Judge 输入必须保存于本项目.agents/Documents内")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(request.model_dump(mode="json")))


async def run_judge_collection(args: argparse.Namespace) -> int:
    from pydantic import TypeAdapter

    batch_id = TypeAdapter(Identifier).validate_python(args.batch_id)
    if not 1 <= len(args.input) <= 1000 or not 0 < args.timeout_seconds <= 3600:
        raise ValueError("Judge 输入数量或采集超时无效")
    frozen, blobs = await asyncio.to_thread(read_freeze, args.freeze)
    collection = decode(await asyncio.to_thread(bounded_read, args.collection))
    if not isinstance(collection, dict):
        raise ValueError("目标采集必须为 JSON 对象")
    target = ReleaseManifest.model_validate_json(await asyncio.to_thread(bounded_read, args.candidate))
    judge = ReleaseManifest.model_validate_json(await asyncio.to_thread(bounded_read, args.judge_candidate))
    inputs = []
    total = 0
    for path in args.input:
        raw = await asyncio.to_thread(bounded_read, path)
        total += len(raw)
        if len(raw) > 65536 or total > 16777216:
            raise ValueError("Judge 输入超过批次大小上限")
        inputs.append(JudgeInput.model_validate(decode(raw)))
    limits = EvaluationLimits(max_sessions=args.max_sessions, max_runs=args.max_runs)
    token = http_token(args)
    output = local_output(args.output)
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    async with httpx.AsyncClient(base_url=args.base_url,
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=httpx.Timeout(30, read=min(args.timeout_seconds, 30)),
                                follow_redirects=False, trust_env=False) as client:
        result, bundle = await collect_judges(
            HTTPEvaluationAPI(client, timeout_seconds=args.timeout_seconds), frozen, blobs, args.split,
            collection, target, judge, tuple(inputs), batch_id,
            lambda record: append_record(output / "journal.jsonl", record),
            limits=limits, lifetime_seconds=args.context_lifetime_seconds,
        )
    await asyncio.to_thread((output / "judge-batch.json").write_text,
                            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if bundle is not None:
        await asyncio.to_thread((output / "observations.json").write_text,
                                bundle.model_dump_json(indent=2), encoding="utf-8")
    return 0 if result["outcome"] == "passed" else 1 if result["outcome"] == "failed" else 2


async def fetch_judge_material(args: argparse.Namespace) -> None:
    token = http_token(args)
    output = local_output(args.output)
    async with httpx.AsyncClient(base_url=args.base_url,
                                headers={"Authorization": f"Bearer {token}"}, timeout=30,
                                follow_redirects=False, trust_env=False) as client:
        material = await HTTPEvaluationAPI(client).judge_source_material(args.run_id)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(material.model_dump_json(indent=2))


def main() -> None:
    args = parser().parse_args()
    if args.command == "freeze":
        blobs = {name: bounded_read(args.source / name) for name in REQUIRED}
        for resource in args.resource:
            name, separator, path = resource.partition("=")
            if not separator or name in blobs or len(blobs) >= 32:
                raise ValueError("资源必须为不重复的 NAME=PATH")
            blobs[name] = bounded_read(Path(path))
        frozen = write_freeze(args.output, blobs)
        print(json.dumps({"freeze_hash": frozen.content_hash(),
                          "dataset_hash": frozen.dataset_hash}))
    elif args.command == "gate":
        raise SystemExit(run_gates(args))
    elif args.command == "merge-observations":
        run_merge(args)
    elif args.command == "prepare-judge":
        run_prepare_judge(args)
    elif args.command == "collect-judge":
        raise SystemExit(asyncio.run(run_judge_collection(args)))
    elif args.command == "fetch-judge-material":
        asyncio.run(fetch_judge_material(args))
    else:
        raise SystemExit(asyncio.run(run_collection(args)))


if __name__ == "__main__":
    main()
