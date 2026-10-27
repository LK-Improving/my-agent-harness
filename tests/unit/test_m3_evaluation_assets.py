"""M3交接资产完整性定义；不导入业务模块，不执行评测或替代实际API/SSE验收。"""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def document(path: str):
    return json.loads((ROOT / path).read_text("utf-8"))


def rows(path: str):
    return [json.loads(line) for line in (ROOT / path).read_text("utf-8").splitlines() if line.strip()]


def test_original_sources_are_byte_frozen_and_holdout_lineage_is_preserved():
    suite = document("evals/m3/suite.json")
    assert {s["role"] for s in suite["sources"]} == {"corpus", "development", "holdout", "operations", "probe_policy"}
    for source in suite["sources"]:
        path = (ROOT / source["path"]).resolve()
        assert path.is_relative_to(ROOT) and path.parent == ROOT / "evals/m0"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256_bytes"]
    sources = {source["role"]: source["path"] for source in suite["sources"]}
    development, holdout = rows(sources["development"]), rows(sources["holdout"])
    assert len(development) == 12 and len(holdout) == 4
    assert not {r["id"] for r in development} & {r["id"] for r in holdout}
    expectations = rows(suite["answer_expectations"])
    assert len({r["case_id"] for r in expectations}) == len(expectations) == 16
    assert {r["source_id"] for r in expectations} == {r["id"] for r in development + holdout}
    assert {r["source_id"] for r in expectations if r["split"] == "holdout"} == {r["id"] for r in holdout}
    assert suite["execution"]["holdout_may_be_used_for_tuning"] is False


def test_answer_labels_distinguish_stale_claim_and_access_from_empty_recall():
    suite = document("evals/m3/suite.json")
    corpus = {r["id"]: r for r in rows("evals/m0/corpus.jsonl")}
    expectations = rows(suite["answer_expectations"])
    for case in expectations:
        assert case["judge_required"] and case["acceptable_points"]
        assert set(case["allowed_support_aliases"]) <= corpus.keys()
        assert set(case["forbidden_support_aliases"]) <= corpus.keys()
        assert not set(case["allowed_support_aliases"]) & set(case["forbidden_support_aliases"])
        assert all(corpus[alias]["tenant"] == "tenant-a" and corpus[alias]["snapshot"] == "s1"
                   and not corpus[alias]["revoked"] for alias in case["allowed_support_aliases"])
    by_source = {r["source_id"]: r for r in expectations}
    assert by_source["q11"]["behavior"] == "reject_stale_claim_or_correct_with_current_evidence"
    assert by_source["q12"]["behavior"] == by_source["h04"]["behavior"] == "withhold_unauthorized_content"
    assert by_source["q10"]["behavior"] == "clarify_or_abstain"
    policy = document(suite["gate_policy"])
    assert policy["retrieval_metrics"]["negative_cases_require_separate_answer_and_isolation_evidence"]
    assert policy["answer_judge"]["missing_or_failed_outcome"] == "incomplete"


def test_scenarios_reference_existing_tests_without_importing_or_claiming_end_to_end():
    suite = document("evals/m3/suite.json")
    scenarios = rows(suite["scenarios"])
    assert len({r["id"] for r in scenarios}) == len(scenarios)
    assert {r["id"] for r in scenarios} == set(suite["required_scenario_ids"])
    assert {task for r in scenarios for task in r["tasks"]} == set(suite["required_tasks"])
    for scenario in scenarios:
        assert scenario["required_evidence"]
        if scenario["layer"] == "manual_e2e":
            assert not scenario["test_refs"] and scenario["status"] == "procedure_defined_not_run"
            assert (ROOT / scenario["procedure_ref"].split("#", 1)[0]).is_file()
        else:
            assert scenario["test_refs"] and scenario["status"] == "defined_not_run"
            for reference in scenario["test_refs"]:
                filename, function = reference.split("::")
                path = (ROOT / filename).resolve()
                assert path.is_relative_to(ROOT / "tests")
                tree = ast.parse(path.read_text("utf-8"))
                assert any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and node.name == function for node in tree.body), reference
    dialogues = rows(suite["dialogues"])
    assert len({row["id"] for row in dialogues}) == len(dialogues)
    assert all(row["scenario_id"] in suite["required_scenario_ids"] and row["requires_judge"] for row in dialogues)


def test_handoff_is_not_a_runner_or_an_acceptance_report():
    suite = document("evals/m3/suite.json")
    policy = document(suite["gate_policy"])
    assert suite["status"] == "defined_not_run" and not suite["execution"]["automatic_run"]
    assert not suite["execution"]["acceptance_runner_implemented"]
    assert not suite["execution"]["skip_or_missing_is_pass"]
    assert policy["status"] == "draft_not_executed"
    assert policy["answer_judge"]["implementation"] is None
    assert policy["retrieval_metrics"]["formal_business_quality_thresholds"] is None
    assert "credential_disclosures" in policy["zero_tolerance_counts"]
