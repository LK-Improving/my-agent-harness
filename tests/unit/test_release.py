"""候选版本身份、能力隔离和评测范围的反例。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harness.contracts.release import CapabilityProfile, EvaluationContext, ReleaseManifest
from harness.contracts.specs import AgentSpec, ToolSpec
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["operations", "travel"])
def test_example_agent_specs_are_valid(name: str) -> None:
    spec = AgentSpec.model_validate_json((ROOT / "agents" / f"{name}.json").read_text("utf-8"))
    if name == "operations":
        assert not spec.knowledge_bindings
    else:
        assert spec.graph_template == "rag_chat"


def test_operations_write_tool_has_explicit_sandbox_guarantees() -> None:
    tool = ToolSpec.model_validate_json(
        (ROOT / "examples/operations_assistant/tool.json").read_text("utf-8")
    )
    assert tool.idempotency is not None
    assert tool.retry.max_attempts_per_operation == 2


def test_manifest_hash_changes_when_executable_inputs_change() -> None:
    spec = AgentSpec.model_validate_json((ROOT / "agents/operations.json").read_text("utf-8"))
    manifest = ReleaseManifest(
        spec=spec,
        source_revision=None,
        source_tree_hash="1" * 64,
        dependency_lock_hash="2" * 64,
        prompt_hash="3" * 64,
        tool_schema_hashes={f"{tool.name}@{tool.version}": "4" * 64 for tool in spec.allowed_tools},
        model_profile={"provider": "controlled-stub", "max_retries": 0},
        profiles={},
        environment_profile_hash="5" * 64,
    )
    assert manifest.source_revision is None
    assert (
        manifest.content_hash()
        != manifest.model_copy(update={"prompt_hash": "a" * 64}).content_hash()
    )
    assert (
        manifest.content_hash()
        == ReleaseManifest.model_validate_json(manifest.model_dump_json()).content_hash()
    )
    with pytest.raises(ValidationError, match="工具目录"):
        ReleaseManifest.model_validate({**manifest.model_dump(), "tool_schema_hashes": {}})
    assert json.loads(manifest.model_dump_json())["model_profile"]["max_retries"] == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"environment": "production"},
        {"execution_worker_replicas": 2},
        {"max_active_runs_per_worker": 2},
        {"enabled_capabilities": ["core", "queue"]},
        {"answer_stream_retry_policy": "replace_draft"},
    ],
)
def test_base_profile_rejects_unimplemented_modes(updates) -> None:
    with pytest.raises(ValidationError):
        CapabilityProfile.model_validate(
            {"environment": "development", "auth_mode": "development", **updates}
        )


def test_evaluation_binding_does_not_grant_cross_subject_or_expired_access() -> None:
    now = datetime.now(UTC)
    context = EvaluationContext(
        context_id="ctx1",
        tenant_id="a",
        subject_id="eval",
        agent_id="ops",
        candidate_release_id="candidate",
        manifest_hash="a" * 64,
        dataset_hash="b" * 64,
        gate_policy_version="gate@1",
        environment_profile_hash="c" * 64,
        resource_scopes=frozenset({"sandbox"}),
        expires_at=now + timedelta(seconds=1),
    )
    context.require_access("a", "eval", now)
    for tenant, subject, when in [
        ("b", "eval", now),
        ("a", "other", now),
        ("a", "eval", now + timedelta(seconds=1)),
    ]:
        with pytest.raises(PermissionError):
            context.require_access(tenant, subject, when)
