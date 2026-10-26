"""验证输入身份、状态边界与配置不能静默放宽。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from harness.contracts.events import ArtifactPage, RuntimeEvent
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.specs import CostSpec, ToolSpec
from harness.contracts.state import RunDeadlines, RunState, validate_transition
from pydantic import ValidationError


def test_request_identity_excludes_transport_and_revision() -> None:
    first = RunRequest(request_id="r1", message="订单 1", expected_revision=0)
    replay = first.model_copy(update={"request_id": "r2", "expected_revision": 99})
    assert first.business_hash() == replay.business_hash()
    assert first.business_hash() != first.model_copy(update={"message": "订单 2"}).business_hash()
    assert first.business_hash() != first.model_copy(update={"message": "订单 1 "}).business_hash()


def test_canonical_unicode_and_key_order_but_not_attachment_version() -> None:
    assert canonical_hash({"b": "e\u0301", "a": 1}) == canonical_hash({"a": 1, "b": "é"})
    first = RunRequest(request_id="r1", message="原文", expected_revision=0)
    assert (
        first.business_hash()
        != first.model_copy(update={"attachment_versions": ("attachment@2",)}).business_hash()
    )
    with pytest.raises(ValueError, match="重复"):
        canonical_hash({"é": 1, "e\u0301": 2})
    with pytest.raises(ValueError):
        canonical_hash(float("nan"))


def test_request_does_not_accept_client_tenant_or_boolean_revision() -> None:
    with pytest.raises(ValidationError):
        RunRequest.model_validate(
            {"request_id": "r1", "message": "输入", "expected_revision": 0, "tenant_id": "other"}
        )
    with pytest.raises(ValidationError):
        RunRequest(request_id="r1", message="输入", expected_revision=True)


@pytest.mark.parametrize("target", [RunState.CANCELLED, RunState.FAILED, RunState.COMMITTING])
def test_unknown_blocks_terminal_or_commit(target: RunState) -> None:
    with pytest.raises(ValueError, match="核对"):
        validate_transition(RunState.RUNNING, target, has_unknown=True)
    validate_transition(RunState.RUNNING, RunState.RECONCILING, has_unknown=True)


def test_terminal_replay_and_approval_cannot_skip_dispatch() -> None:
    validate_transition(RunState.SUCCEEDED, RunState.SUCCEEDED)
    with pytest.raises(ValueError):
        validate_transition(RunState.SUCCEEDED, RunState.RUNNING)
    with pytest.raises(ValueError):
        validate_transition(RunState.WAITING_APPROVAL, RunState.SUCCEEDED)


def test_unknown_cost_is_preserved_and_free_is_explicit() -> None:
    assert CostSpec(billable=True).upper_bound is None
    assert CostSpec(billable=False, upper_bound=Decimal("0")).upper_bound == 0
    with pytest.raises(ValidationError):
        CostSpec(billable=False)
    with pytest.raises(ValidationError):
        CostSpec(billable=True, upper_bound=Decimal("1"))


def test_write_tool_cannot_omit_governance() -> None:
    with pytest.raises(ValidationError, match="审批"):
        ToolSpec(
            name="create",
            version="1",
            handler_ref="domain.create",
            effect="write",
            input_schema={},
            output_schema={},
            authorization_policy_ref="auth@1",
            timeout_ms=100,
            cost=CostSpec(billable=False, upper_bound=Decimal("0")),
        )


def test_unicode_artifact_offsets() -> None:
    page = ArtifactPage(
        ref="artifact-1",
        content_hash="a" * 64,
        source_ref="operation-1",
        offset=2,
        next_offset=4,
        has_more=True,
        content="中😀",
        truncated=True,
    )
    assert len(page.content) == 2
    with pytest.raises(ValidationError, match="Unicode"):
        ArtifactPage.model_validate({**page.model_dump(), "next_offset": 7})


def test_model_draft_must_record_attempt() -> None:
    with pytest.raises(ValidationError, match="尝试身份"):
        RuntimeEvent(
            run_id="run1",
            seq=1,
            type="answer.started",
            timestamp=datetime.now(UTC),
            payload={"draft_id": "d1", "producer": "model"},
        )


def test_resume_does_not_get_fresh_execution_budget() -> None:
    now = datetime.now(UTC)
    deadlines = RunDeadlines(
        queue_deadline_at=now + timedelta(seconds=10),
        execution_budget_ms=120000,
        execution_used_ms=119000,
        run_expires_at=now + timedelta(seconds=20),
    )
    assert deadlines.execution_remaining_ms(now) == 1000
    assert deadlines.execution_remaining_ms(now + timedelta(seconds=21)) == 0
    with pytest.raises(ValueError, match="时区"):
        deadlines.execution_remaining_ms(now.replace(tzinfo=None))
