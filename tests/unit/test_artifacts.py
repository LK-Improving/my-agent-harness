"""产物字符边界、同端口分页与统一执行计数的测试定义；本批不运行。"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from domain_tools.knowledge.completion import knowledge_receipts
from harness.bootstrap.api import create_app
from harness.context.artifacts import artifact_text, page_text
from harness.context.requests import envelope_tokens
from harness.contracts.artifacts import ArtifactPage, ArtifactProfile, ArtifactRead
from harness.contracts.execution import ReadReservation
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ToolBinding
from harness.contracts.state import ToolReceipt
from harness.tools.artifacts import registered_artifact_reader
from harness.tools.reads import ReadCoordinator
from tests.unit.test_knowledge_runtime import values


def test_unicode_pages_reassemble_exact_text_and_json() -> None:
    profile = ArtifactProfile(page_bytes=7)
    for data in ("中😀e\u0301结尾", {"z": "中😀" * 20, "a": [1, True, None]}):
        text, media, digest = artifact_text(data)
        offset, pages = 0, []
        while True:
            page, next_offset = page_text(text, offset, 4, profile)
            assert len(page.encode("utf-8")) <= 7
            pages.append(page)
            if next_offset is None:
                break
            assert next_offset == offset + len(page) and next_offset > offset
            offset = next_offset
        assert "".join(pages) == text and len(digest) == 64
        assert media == ("text/plain" if isinstance(data, str) else "application/json")
        assert page_text(text, len(text), 4, profile) == ("", None)
    with pytest.raises(RuntimeFault, match="ARTIFACT_PAGE_INVALID"):
        page_text("文字", 3, 1, profile)
    for update in ({"offset": -1}, {"limit": 4001}, {"offset": True}, {"artifact_ref": "file:///private"}):
        with pytest.raises(ValidationError):
            ArtifactRead.model_validate({"artifact_ref": "artifact-" + "a" * 32, **update})
    assert artifact_text({"b": 1, "a": 2}) == artifact_text({"a": 2, "b": 1})


class PageReader:
    def __init__(self) -> None:
        self.calls = []
        self.revoked = False
        self.expires = datetime.now(UTC) + timedelta(days=1)

    async def read(self, actor, session_id, request):
        if self.revoked:
            raise RuntimeFault("REVOKED", 403)
        self.calls.append((actor, session_id, request))
        text = "中文😀分页" * 5
        body, next_offset = page_text(text, request.offset, request.limit, ArtifactProfile())
        return ArtifactPage(artifact_ref=request.artifact_ref, media_type="text/plain", content_hash=artifact_text(text)[2],
                            source_run_id="source-run", source_operation_id="source-op", total_chars=len(text),
                            total_bytes=len(text.encode("utf-8")), expires_at=self.expires,
                            offset=request.offset, next_offset=next_offset, text=body)


class ReadLedger:
    def __init__(self) -> None:
        self.count = 0
        self.receipt = None
        self.args = None

    async def reserve_read(self, run, owner, logical, spec, arguments):
        if self.receipt is None:
            self.count += 1
        self.args = arguments
        return ReadReservation(operation_id="page-op", attempt_id="page-attempt", receipt=self.receipt)

    async def save_read(self, reservation, data, error):
        self.receipt = ToolReceipt(operation_id=reservation.operation_id, attempt_id=reservation.attempt_id,
                                   status="failed" if error else "succeeded", args_hash=self.args.content_hash,
                                   data=data.materialize() if data else None, cost_status="not_billable")
        return self.receipt


@pytest.mark.asyncio
async def test_builtin_uses_trusted_session_dispatch_cache_and_rechecks_authority() -> None:
    run, _, knowledge, _ = values()
    reader, ledger = PageReader(), ReadLedger()
    entry = registered_artifact_reader(reader)
    spec = run.manifest.spec.model_copy(update={"allowed_tools": (ToolBinding(name="read_artifact", version="1", max_operations_per_run=4),)})
    run = run.model_copy(update={"manifest": run.manifest.model_copy(update={"spec": spec})})

    def domain_guard(*args):
        raise AssertionError("内置工具不得接受领域业务参数守卫")

    coordinator = ReadCoordinator(ledger, (entry,), task_guard=domain_guard)
    args = {"artifact_ref": "artifact-" + "a" * 32, "offset": 2, "limit": 5}
    receipt = await coordinator.execute(run, "owner", "read_artifact", "logical", args)
    assert await coordinator.execute(run, "owner", "read_artifact", "logical", args) == receipt
    assert ledger.count == 1 and all(session == run.session_id for _, session, _ in reader.calls)
    assert knowledge_receipts((knowledge, receipt)) == (knowledge,)
    reader.revoked = True
    with pytest.raises(RuntimeFault, match="REVOKED"):
        await coordinator.execute(run, "owner", "read_artifact", "logical", args)


@pytest.mark.asyncio
async def test_client_and_tool_share_page_schema_and_invalid_requests_are_rejected() -> None:
    run, _, _, _ = values()
    reader = PageReader()

    class Store:
        async def admit_read(self, actor):
            pass

    app = create_app(Store(), lambda token: run.principal, artifacts=reader)
    url = f"/api/v1/sessions/{run.session_id}/artifacts/artifact-{'a' * 32}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(url, params={"offset": 3, "limit": 7})
        expected = await reader.read(run.principal, run.session_id, ArtifactRead(artifact_ref="artifact-" + "a" * 32, offset=3, limit=7))
        assert response.status_code == 200 and response.json() == expected.model_dump(mode="json")
        assert (await client.get(url, params={"limit": 4001})).status_code == 422
        assert (await client.get(url, params={"offset": -1})).status_code == 422
        reader.revoked = True
        assert (await client.get(url)).status_code == 403
    # 正文页连同来源/分页元数据进入完整模型封装计数，不把读回算免费上下文。
    base = [{"role": "user", "content": "读取"}]
    with_page = [*base, {"role": "assistant", "tool_calls": [{"id": "page"}]},
                 {"role": "tool", "tool_call_id": "page", "content": JsonSnapshot.capture(expected.model_dump(mode="json")).canonical_text}]
    assert envelope_tokens("test", with_page, [], 128) > envelope_tokens("test", base, [], 128)
