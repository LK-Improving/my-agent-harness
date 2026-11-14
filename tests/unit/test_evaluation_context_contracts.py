"""M5-02 准入/撤销边界测试定义；本轮未运行。"""

import pytest
from pydantic import ValidationError

from harness.contracts.evaluation_contexts import (
    EvaluationAdmission, EvaluationLimits, EvaluationRevocationRequest,
)
from harness.contracts.release import EvaluationContext


@pytest.mark.parametrize("values", [{"max_sessions": 0}, {"max_runs": True},
                                    {"max_runs": 1001}, {"max_sessions": "2"}])
def test_client_cannot_remove_resource_bounds(values):
    with pytest.raises(ValidationError):
        EvaluationLimits(**values)


def test_test_resources_are_exact_server_allowlist_without_prefix_inference():
    profile = EvaluationAdmission()
    assert profile.test_resource_scopes == frozenset({"synthetic"})
    assert "kb:production" not in profile.test_resource_scopes
    assert profile.max_valid_contexts_per_subject > 0
    assert EvaluationAdmission(candidate_release_ids=frozenset()).candidate_release_ids == frozenset()


@pytest.mark.parametrize("changes", [{"expected_manifest_hash": "bad"}, {"reason": "arbitrary"},
                                    {"expires_at": "2099-01-01T00:00:00Z"}])
def test_revocation_has_no_reactivation_or_free_text_channel(changes):
    with pytest.raises(ValidationError):
        EvaluationRevocationRequest.model_validate({"request_id": "revoke-1",
            "expected_manifest_hash": "a" * 64, **changes})


def test_legacy_context_does_not_gain_extra_resource_scopes():
    context = EvaluationContext(context_id="legacy", tenant_id="t", subject_id="u", agent_id="a",
        candidate_release_id="r", manifest_hash="a" * 64, dataset_hash="b" * 64,
        gate_policy_version="gate@1", environment_profile_hash="c" * 64,
        resource_scopes=frozenset({"synthetic"}), expires_at="2026-10-28T00:00:00Z")
    assert context.admission_version == "legacy-evaluation@1"
    assert context.resource_scopes == frozenset({"synthetic"})
    assert context.limits.max_runs == 128
