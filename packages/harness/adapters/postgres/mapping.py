"""数据库行到契约的纯映射；不连接数据库，不执行授权或提交。"""

from harness.adapters.postgres.connection import Row
from harness.contracts.runtime import Principal, RunView
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.state import ToolReceipt
from harness.contracts.writes import WriteOperation


def run_view(row: Row, session: Row) -> RunView:
    return RunView.model_validate(
        {
            **{name: row[name] for name in RunView.model_fields if name in row},
            "release_id": session["release_id"],
        }
    )


def write_operation(row: Row) -> WriteOperation:
    return WriteOperation(
        operation_id=row["operation_id"],
        run_id=row["run_id"],
        logical_call_id=row["logical_call_id"],
        principal=Principal.model_validate(row["principal"]),
        tool=JsonSnapshot.capture(row["tool"]),
        arguments=JsonSnapshot.capture(row["arguments"]),
        action_key=row["action_key"],
        plan_hash=row["plan_hash"],
        binding_hash=row["binding_hash"],
        approval_id=row["approval_id"],
        state=row["state"],
        receipt=ToolReceipt.model_validate(row["receipt"]) if row["receipt"] else None,
        output_error=row["output_error"],
    )
