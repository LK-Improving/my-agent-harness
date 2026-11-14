"""按冻结用例重算描述性分项，保留不适用和缺测。"""

from collections import Counter, defaultdict
from decimal import Decimal
from math import ceil, log2
from statistics import mean
from typing import Any, cast

from harness.contracts.evaluation import EvaluationCase, EvaluationFreeze, JudgePolicy, MetricPolicy
from harness.contracts.evaluation_gates import JudgeReceipt, RunGateEvidence
from harness.contracts.evaluation_metrics import MetricObservation
from harness.contracts.events import RuntimeEvent
from harness.contracts.release import ReleaseManifest
from harness.evaluation.gates import TERMINALS

COST_CATEGORIES = ("model", "tool", "retrieval", "judge")
TIMINGS = ("queue_ms", "total_ms", "first_public_answer_delta_ms", "approval_wait_ms", "execution_used_ms")


def ranking_scores(gold: set[str], ranked: tuple[str, ...]) -> dict[str, float]:
    """按切片别名去重后截断，使用二值相关性和固定五槽 Precision。"""
    hits = [i for i, alias in enumerate(tuple(dict.fromkeys(ranked))[:5], 1) if alias in gold]
    ideal = sum(1 / log2(i + 1) for i in range(1, min(5, len(gold)) + 1))
    return {"precision_at_5": len(hits) / 5, "recall_at_5": len(hits) / len(gold),
            "mrr_at_5": 1 / hits[0] if hits else 0.0,
            "ndcg_at_5": sum(1 / log2(i + 1) for i in hits) / ideal}


def distribution(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    n = len(ordered)
    return {"n": n, **{f"p{p}": ordered[ceil(n * p / 100) - 1] if n else None for p in (50, 95, 99)},
            "small_sample": 0 < n < 100}


def timing_samples(events: list[RuntimeEvent], ledger: RunGateEvidence) -> dict[str, float]:
    accepted = events[0].timestamp
    timings = {"execution_used_ms": float(ledger.execution_used_ms)}

    def record(name: str, event: RuntimeEvent | None) -> None:
        if event is not None and event.timestamp >= accepted:
            timings[name] = (event.timestamp - accepted).total_seconds() * 1000

    record("total_ms", events[-1] if events[-1].type in TERMINALS else None)
    record("queue_ms", next((e for e in events if e.type == "run.started"), None))
    record("first_public_answer_delta_ms", next((e for e in events if e.type == "answer.delta"), None))
    pending = {}
    waits = []
    valid = True
    for event in events:
        approval = event.payload.get("approval_id")
        if not isinstance(approval, str):
            continue
        if event.type == "approval.required":
            valid = valid and approval not in pending
            pending[approval] = event.timestamp
        elif event.type == "approval.decided":
            start = pending.pop(approval, None)
            if start is None or start > event.timestamp:
                valid = False
            else:
                waits.append((event.timestamp - start).total_seconds() * 1000)
    if valid and not pending and waits:
        timings["approval_wait_ms"] = sum(waits)
    return timings


def summarize(
    collection: dict[str, object], rows: tuple[EvaluationCase, ...], frozen: EvaluationFreeze,
    candidate: ReleaseManifest, policy: MetricPolicy, judge: JudgePolicy, gates: dict[str, object],
) -> dict[str, object]:
    gate_cases = {item["case_id"]: item for item in cast(list[dict[str, Any]], gates["cases"])}
    versions = cast(dict[str, Any], gates["version_comparison"])
    version_cases = {item["case_id"]: item["outcome"] for item in versions["cases"]}
    raw = collection.get("cases")
    indexed: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("case_id"), str):
            indexed[item["case_id"]].append(item)
    results = []
    metric_values: defaultdict[str, list[float]] = defaultdict(list)
    latencies: defaultdict[str, list[float]] = defaultdict(list)
    amounts: defaultdict[str, Decimal] = defaultdict(Decimal)
    cost_states: dict[str, Counter[str]] = {name: Counter() for name in COST_CATEGORIES}
    price_versions: dict[str, set[str]] = {name: set() for name in COST_CATEGORIES}
    eligible_runs = completed = full_cost = 0
    expected_retrieval = expected_memory = expected_no_evidence = 0

    for row in rows:
        gold = row.expected.get("relevant_aliases")
        retrieval = isinstance(gold, list) and bool(gold) and all(isinstance(x, str) for x in gold)
        no_evidence = gold == []
        memory = row.expected.get("memory_evaluation") is True
        expected_retrieval += retrieval
        expected_no_evidence += no_evidence
        expected_memory += memory
        case: dict[str, object] = {"case_id": row.case_id, "quality": None,
            "retrieval": None, "memory": None, "timings_ms": None,
            "retrieval_applicable": retrieval, "memory_applicable": memory,
            "no_evidence_applicable": no_evidence, "no_evidence_decision": None,
            "costs": dict.fromkeys(COST_CATEGORIES), "cost_total_by_currency": None,
            "evidence": "missing_or_unbound"}
        results.append(case)
        records = indexed[row.case_id]
        gate_checks = {item["gate"]: item["outcome"] for item in gate_cases[row.case_id]["checks"]}
        if (len(records) != 1 or version_cases.get(row.case_id) != "passed"
                or any(gate_checks.get(name) != "passed" for name in ("case_execution", "persisted_sse"))
                or any(name in gate_checks for name in ("case_input", "persisted_audit", "terminal_result"))
                or versions["outcome"] == "failed"
                or gates["checks"]):
            continue
        item = records[0]
        ledger = RunGateEvidence.model_validate(item["ledger_evidence"])
        events = [RuntimeEvent.model_validate(event) for event in item["events"]]
        timings = timing_samples(events, ledger)
        case.update(evidence="persisted_run_bound", timings_ms=timings)
        eligible_runs += 1
        completed += ledger.status == "succeeded" and events[-1].type == "run.completed"
        for name, value in timings.items():
            latencies[name].append(value)

        if judge.required and gate_checks.get("judge") == "passed":
            scores = JudgeReceipt.model_validate(item["judge_receipt"]).scores
            case["quality"] = scores
            for name, value in scores.items():
                metric_values[f"judge.{name}"].append(value)
            if no_evidence and "no_evidence_decision" in scores:
                case["no_evidence_decision"] = scores["no_evidence_decision"]
                metric_values["no_evidence_decision"].append(scores["no_evidence_decision"])
        try:
            observed = MetricObservation.model_validate(item.get("metric_observation"))
        except ValueError:
            continue
        if (observed.case_id, observed.run_id, observed.freeze_hash, observed.manifest_hash,
                observed.result_hash, observed.event_seq) != (
                row.case_id, ledger.run_id, frozen.content_hash(), candidate.content_hash(),
                ledger.result_hash, ledger.event_seq):
            continue
        case["metric_observation_source"] = observed.source_ref
        if retrieval and isinstance(gold, list) and observed.ranked_chunk_aliases is not None:
            ranking = ranking_scores(set(gold), observed.ranked_chunk_aliases)
            case["retrieval"] = ranking
            for name, value in ranking.items():
                metric_values[name].append(value)
        if memory and observed.memory is not None:
            evidence = observed.memory
            memory_scores: dict[str, float] = {name: int(value) for name, value in (
                ("key_fact_retained", evidence.key_fact_retained), ("erroneous_deletion", evidence.erroneous_deletion),
                ("correction_followed", evidence.correction_followed)) if value is not None}
            if evidence.before_tokens and evidence.after_tokens is not None:
                memory_scores["compression_savings"] = 1 - evidence.after_tokens / evidence.before_tokens
            case["memory"] = memory_scores
            for name, value in memory_scores.items():
                metric_values[f"memory.{name}"].append(value)
        case_amounts: defaultdict[str, Decimal] = defaultdict(Decimal)
        for name, cost in observed.costs.items():
            cost_states[name][cost.state] += 1
            if cost.amount is not None and cost.currency is not None and cost.price_version is not None:
                case_amounts[cost.currency] += cost.amount
                amounts[cost.currency] += cost.amount
                price_versions[name].add(cost.price_version)
        case["costs"] = {name: observed.costs[name].model_dump(mode="json") if name in observed.costs else None
                         for name in COST_CATEGORIES}
        if all(name in observed.costs and observed.costs[name].state != "unknown" for name in COST_CATEGORIES):
            full_cost += 1
            case["cost_total_by_currency"] = {key: str(value) for key, value in case_amounts.items()}

    n = len(rows)

    def score(name: str, applicable: int) -> dict[str, object]:
        values = metric_values[name]
        return {"mean": mean(values) if values else None, "n": len(values),
                "applicable": applicable, "missing": applicable - len(values), "not_applicable": n - applicable}

    environment = candidate.profiles.as_object().get("environment")
    environment = environment if isinstance(environment, dict) else {}
    return {
        "version": "evaluation-metrics-report@1", "metric_policy_version": policy.version,
        "configured_metrics": list(policy.metrics),
        "thresholds_confirmed": policy.thresholds_confirmed, "threshold_outcome": "not_evaluated",
        "expected_cases": n, "bound_runs": eligible_runs, "cases": results,
        "quality": {"judge_version": judge.version,
                    "dimensions": {name: score(f"judge.{name}", n if judge.required else 0)
                                   for name in judge.dimensions},
                    "no_evidence_decision": score("no_evidence_decision", expected_no_evidence if judge.required else 0)},
        "retrieval": {"level": "chunk_alias", "k": 5, "relevance": "binary",
                      **{f"{name}_at_5": score(f"{name}_at_5", expected_retrieval)
                         for name in ("recall", "precision", "mrr", "ndcg")}},
        "memory": {name: score(f"memory.{name}", expected_memory) for name in
                   ("key_fact_retained", "erroneous_deletion", "correction_followed", "compression_savings")},
        "cost": {"fully_covered_cases": full_cost, "expected_cases": n, "coverage": full_cost / n if n else None,
                 "known_subtotal_by_currency": {key: str(value) for key, value in amounts.items()} or None,
                 "batch_total_by_currency": {key: str(value) for key, value in amounts.items()} if n and full_cost == n else None,
                 "categories": {name: {**{state: cost_states[name][state] for state in ("known", "unknown", "not_billable")},
                                       "missing": n - sum(cost_states[name].values()),
                                       "price_versions": sorted(price_versions[name])} for name in COST_CATEGORIES}},
        "capacity": {"collection_mode": "sequential" if collection.get("version") == "evaluation-collection@1" else None,
                     "client_concurrency": 1 if collection.get("version") == "evaluation-collection@1" else None,
                     "configured_worker_replicas": environment.get("execution_worker_replicas"),
                     "configured_slots_per_worker": environment.get("max_active_runs_per_worker"),
                     "completed": completed, "observed_run_count": eligible_runs,
                     "completion_rate": completed / eligible_runs if eligible_runs else None,
                     "percentiles_ms": {name: distribution(latencies[name]) for name in TIMINGS},
                     "scope": "descriptive_only"},
    }
