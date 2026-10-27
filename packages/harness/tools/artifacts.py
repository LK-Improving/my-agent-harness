"""内置读回工具；使用可信session，复用客户端的授权字符分页端口。"""

from typing import Any

from harness.contracts.artifacts import ArtifactPage, ArtifactRead, ArtifactReader
from harness.contracts.runtime import ClaimedRun, Principal, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ToolSpec
from harness.contracts.state import ToolReceipt
from harness.tools.reads import RegisteredRead


def registered_artifact_reader(reader: ArtifactReader) -> RegisteredRead:
    async def forbidden(actor: Principal, arguments: dict[str, Any]) -> Any:
        raise RuntimeFault("ARTIFACT_RUN_CONTEXT_REQUIRED", 403)

    async def read(run: ClaimedRun, owner: str, logical: str, arguments: dict[str, Any]) -> dict[str, Any]:
        page = await reader.read(run.principal, run.session_id, ArtifactRead.model_validate(arguments))
        return page.model_dump(mode="json")

    async def revalidate(run: ClaimedRun, owner: str, receipt: ToolReceipt) -> None:
        page = ArtifactPage.model_validate(receipt.data)
        current = await reader.read(run.principal, run.session_id, ArtifactRead(
            artifact_ref=page.artifact_ref, offset=page.offset, limit=max(1, len(page.text))))
        if current != page:
            raise RuntimeFault("ARTIFACT_PAGE_CHANGED", 409)

    spec = ToolSpec(name="read_artifact", version="1", handler_ref="harness.artifact.read@1", effect="read",
                    input_schema=ArtifactRead.model_json_schema(), output_schema=ArtifactPage.model_json_schema(),
                    authorization_policy_ref="artifact-session-source@1", timeout_ms=10000,
                    cost={"billable": False, "upper_bound": "0"})
    return RegisteredRead(spec=JsonSnapshot.capture(spec.model_dump(mode="json")), input_type=ArtifactRead,
                          output_type=ArtifactPage, handler=forbidden, authorize=lambda actor, args: True,
                          contextual_handler=read, revalidate=revalidate,
                          task_guard=lambda run, spec, args: None)
