"""上下文流水线、完整工具组和摘要故障测试定义；本批不运行。"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from harness.context.pipeline import ContextPipeline
from harness.context.projection import project_model_messages
from harness.context.requests import require_complete_tool_groups
from harness.contracts.context import ContextProfile, ContextSnapshot, ContextTurn, source_hash
from harness.contracts.execution import ExecutionProfile
from harness.contracts.models import ModelResult, ModelUsage
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from tests.unit.test_knowledge_runtime import values
from tests.unit.test_m0_boundaries import prepared

ROOT = Path(__file__).resolve().parents[2]
PROMPT = (ROOT / "prompts/context/summary.txt").read_text("utf-8")


def context_values(*, history_mode="success_history", soft=1):
    run, _, _, _ = values()
    execution = ExecutionProfile.model_validate({
        "route": {"provider": "controlled", "model": "controlled", "adapter_ref": "test@1",
            "configuration_hash": "f" * 64, "capabilities": ["tool_calling"], "capabilities_verified": True,
            "context_window_tokens": 100000, "max_output_tokens": 128},
        "budget": {"version": "test@1", "max_input_tokens": 12000, "max_output_tokens": 128,
            "max_total_tokens": 50000, "execution_budget_ms": 120000}})
    profile = ContextProfile(history_mode=history_mode, soft_limit_percent=soft,
        summary_prompt_hash=hashlib.sha256(PROMPT.encode("utf-8")).hexdigest())
    turns = tuple(ContextTurn(run_id=f"old-{i}", revision=i + 1, messages=JsonSnapshot.capture([
        {"role": "user", "content": f"旧任务{i}"}, {"role": "assistant", "content": "用户已确认的事实" * 20}])) for i in range(3))
    snapshot = ContextSnapshot(source_revision=3, turns=turns if history_mode == "success_history" else (),
        protected_state=JsonSnapshot.capture({"operations": [{"operation_id": "pending", "state": "unknown", "args_hash": "a" * 64}]}))
    history = [m for turn in turns for m in turn.messages.as_object_array()]
    manifest = run.manifest.model_copy(update={"profiles": JsonSnapshot.capture({"context": profile.model_dump(mode="json"),
        "execution": execution.model_dump(mode="json")}), "model_profile": JsonSnapshot.capture(execution.route.model_dump(mode="json"))})
    run = run.model_copy(update={"manifest": manifest, "history": JsonSnapshot.capture(history),
        "request": run.request.model_copy(update={"expected_revision": 3})})
    current = [{"role": "user", "content": run.request.message},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "分页乙"},
        {"role": "tool", "tool_call_id": "a", "content": "分页甲"}]
    messages = JsonSnapshot.capture([{"role": "system", "content": "固定规则"},
        *(history if history_mode == "success_history" else []), *current])
    return run, execution, profile, snapshot, messages, current


class Source:
    def __init__(self, snapshot):
        self.snapshot, self.revoked = snapshot, False

    async def load(self, run, owner, profile):
        if self.revoked:
            raise RuntimeFault("HISTORY_REVOKED", 403)
        return self.snapshot


class Projection:
    async def project_messages(self, run, owner, messages, *, source_run_id=None):
        return project_model_messages(messages.as_object_array())


class Summarizer:
    def __init__(self, fault=None):
        self.calls, self.fault = [], fault

    async def __call__(self, run, owner, messages, purpose):
        self.calls.append((messages, purpose))
        if self.fault == "cancel":
            raise asyncio.CancelledError()
        if self.fault == "fail":
            raise RuntimeFault("SUMMARY_FAILED", 503)
        content = {"invalid": True} if self.fault == "schema" else {"summary": "保留事实与纠错"}
        return ModelResult(model="controlled", content=json.dumps(content, ensure_ascii=False),
            finish_reason="length" if self.fault == "length" else "stop", request_id="test", usage=ModelUsage(),
            model_call_id="summary-call", model_attempt_id="summary-attempt", request_fingerprint="a" * 64)


@pytest.mark.asyncio
async def test_summary_covers_only_older_turns_and_preserves_current_pending_and_tool_groups():
    run, execution, profile, snapshot, messages, current = context_values()
    source, summary = Source(snapshot), Summarizer()
    pipeline = ContextPipeline(source, Projection(), summary, execution, profile, PROMPT)
    built = await pipeline.build(run, "owner", messages, JsonSnapshot.capture([]))
    assert len(summary.calls) == 1 and built.candidate.purpose.covered_turn_ids == ("old-0",)
    assert built.candidate.purpose.source_hash == source_hash(3, snapshot.turns[:1])
    output = built.messages.as_object_array()
    assert output[-len(current):] == current
    require_complete_tool_groups(output)
    assert "pending" in built.messages.canonical_text and "unknown" in built.messages.canonical_text
    assert all(turn.messages.as_object_array()[0] in output for turn in snapshot.turns[1:])
    assert output[1]["role"] == "user" and "context_summary_data" in output[1]["content"]
    assert built.input_tokens == pipeline.count(built.messages, JsonSnapshot.capture([]))
    source.revoked = True
    with pytest.raises(RuntimeFault, match="HISTORY_REVOKED"):
        await pipeline.authorize(run, "owner")
    assert snapshot.source_revision == 3  # 只产生本轮候选，不推进任何来源游标。


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["cancel", "fail", "schema", "length"])
async def test_summary_failure_does_not_fall_back_to_truncated_or_mutated_history(fault):
    run, execution, profile, snapshot, messages, _ = context_values()
    summary = Summarizer(fault)
    pipeline = ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT)
    expected = asyncio.CancelledError if fault == "cancel" else RuntimeFault
    with pytest.raises(expected):
        await pipeline.build(run, "owner", messages, JsonSnapshot.capture([]))
    assert len(summary.calls) == 1 and snapshot.source_revision == 3


@pytest.mark.asyncio
async def test_current_only_never_sends_old_knowledge_to_summary():
    run, execution, profile, snapshot, messages, current = context_values(history_mode="current_only")
    summary = Summarizer()
    built = await ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT).build(
        run, "owner", messages, JsonSnapshot.capture([]))
    assert not summary.calls and built.candidate is None
    assert "旧任务" not in built.messages.canonical_text and built.messages.as_object_array()[-len(current):] == current


@pytest.mark.asyncio
async def test_final_count_includes_tools_and_hard_failure_never_recursively_summarizes():
    run, execution, profile, snapshot, messages, _ = context_values()
    summary = Summarizer()
    pipeline = ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT)
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED"):
        await pipeline.build(run, "owner", messages, JsonSnapshot.capture([{"schema": "中文" * 5000}]))
    assert len(summary.calls) == 0
    changed = messages.as_object_array()
    changed[-1]["content"] = "x" * 20000
    summary.calls.clear()
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED"):
        await pipeline.build(run, "owner", JsonSnapshot.capture(changed), JsonSnapshot.capture([]))
    assert len(summary.calls) == 0


@pytest.mark.asyncio
async def test_history_mutation_and_incomplete_tool_group_are_rejected():
    run, execution, profile, snapshot, messages, _ = context_values()
    pipeline = ContextPipeline(Source(snapshot), Projection(), Summarizer(), execution, profile, PROMPT)
    changed = messages.as_object_array()
    changed[1]["content"] = "改写历史"
    with pytest.raises(RuntimeFault, match="CONTEXT_HISTORY_PREFIX_CHANGED"):
        await pipeline.build(run, "owner", JsonSnapshot.capture(changed), JsonSnapshot.capture([]))
    with pytest.raises(RuntimeFault, match="TOOL_TRANSACTION_GROUP_INCOMPLETE"):
        await pipeline.build(run, "owner", JsonSnapshot.capture(messages.as_object_array()[:-1]), JsonSnapshot.capture([]))


def test_v1_prepared_request_fingerprint_does_not_acquire_purpose_fields():
    request = prepared()
    assert not {"purpose", "purpose_ref", "purpose_binding"} & request.model_dump(mode="json").keys()
    assert request.fingerprint() == type(request).model_validate_json(request.model_dump_json()).fingerprint()
