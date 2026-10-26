"""M2 纯边界负例；不冒充外部写入或恢复证明。"""

import pytest
from harness.contracts.writes import ReconciliationResult
from harness.tools.writes import WriteRegistry
from pydantic import SecretStr, ValidationError

from domain_tools.operations.work_orders import (
    CreateWorkOrderInput,
    WorkOrderSandboxHandler,
    registered_work_order,
)


def test_404_or_missing_data_is_not_proof_of_no_write() -> None:
    with pytest.raises(ValidationError, match="权威证明"):
        ReconciliationResult(
            operation_id="operation",
            args_hash="a" * 64,
            conclusion="confirmed_not_executed",
            evidence_ref="http-404",
        )
    with pytest.raises(ValidationError, match="外部回执"):
        ReconciliationResult(
            operation_id="operation",
            args_hash="a" * 64,
            conclusion="confirmed_succeeded",
            evidence_ref="model-claim",
        )


def test_sandbox_registration_and_scope_are_explicit() -> None:
    handler = WorkOrderSandboxHandler("http://127.0.0.1:18081", SecretStr("a" * 40))
    entry = registered_work_order(handler)
    registry = WriteRegistry((entry,))
    assert registry.get("create_work_order").spec.content_hash == entry.spec.content_hash
    with pytest.raises(ValueError, match="重名"):
        WriteRegistry((entry, entry))
    for origin in (
        "http://example.com",
        "http://localhost@evil.test",
        "http://127.0.0.1/path",
        "http://localhost?secret=value",
    ):
        with pytest.raises(ValueError):
            WorkOrderSandboxHandler(origin, SecretStr("a" * 40))
    with pytest.raises(ValidationError):
        CreateWorkOrderInput(order_id="production-order-123")
