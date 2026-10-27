"""首期记忆能力关闭与不可压缩输入拒绝定义；本批不运行。"""

import json

import pytest

from harness.context.boundary import context_input_limit, require_context_capacity, require_memory_boundary
from harness.context.pipeline import ContextPipeline
from harness.contracts.context import MemoryBoundaryProfile
from harness.contracts.models import ModelResult, ModelUsage
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.runtime.models import GovernedModelGateway
from tests.unit.test_context_pipeline import PROMPT, Projection, Source, Summarizer, context_values


def with_boundary(run, **changes):
    boundary = {**MemoryBoundaryProfile().model_dump(mode="json"), **changes}
    return run.model_copy(update={"manifest": run.manifest.model_copy(update={"profiles": JsonSnapshot.capture({
        **run.manifest.profiles.as_object(), "context_boundary": boundary})})})


def test_defaults_are_session_only_and_each_uninstalled_memory_capability_is_rejected():
    run, execution, *_ = context_values()
    boundary = require_memory_boundary(run.manifest)
    assert boundary.memory_profile_ref == "session-only@1"
    assert not boundary.automatic_long_term_profile and not boundary.semantic_history_recall and not boundary.complex_fact_extraction
    for key in ("automatic_long_term_profile", "semantic_history_recall", "complex_fact_extraction"):
        with pytest.raises(RuntimeFault, match="MEMORY_CAPABILITY_NOT_INSTALLED"):
            require_memory_boundary(with_boundary(run, **{key: True}).manifest)
    unknown = run.manifest.spec.model_copy(update={"memory_profile_ref": "semantic-profile@1"})
    with pytest.raises(RuntimeFault, match="MEMORY_PROFILE_NOT_INSTALLED"):
        require_memory_boundary(run.manifest.model_copy(update={"spec": unknown}))
    limit = context_input_limit(execution)
    require_context_capacity(limit, execution)
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED"):
        require_context_capacity(limit + 1, execution)
    small_window = execution.model_copy(update={"route": execution.route.model_copy(update={"context_window_tokens": 4096})})
    assert context_input_limit(small_window) == 4096 - execution.budget.max_output_tokens - execution.safety_margin_tokens
    unknown_window = execution.model_copy(update={"route": execution.route.model_copy(update={"context_window_tokens": None})})
    with pytest.raises(RuntimeFault, match="MODEL_CONTEXT_WINDOW_UNVERIFIED"):
        require_context_capacity(1, unknown_window)


@pytest.mark.asyncio
async def test_unsupported_memory_is_rejected_before_source_or_model_calls():
    run, execution, profile, snapshot, messages, _ = context_values()
    calls = []

    class NoCalls:
        async def load(self, *args):
            calls.append("source")
            raise AssertionError("不应读取来源")

        async def complete(self, *args):
            calls.append("transport")
            raise AssertionError("不应派发模型")

    bad = with_boundary(run, semantic_history_recall=True)
    with pytest.raises(RuntimeFault, match="MEMORY_CAPABILITY_NOT_INSTALLED"):
        await ContextPipeline(NoCalls(), Projection(), Summarizer(), execution, profile, PROMPT).build(
            bad, "owner", messages, JsonSnapshot.capture([]))
    gateway = GovernedModelGateway(NoCalls(), NoCalls(), execution)
    with pytest.raises(RuntimeFault, match="MEMORY_CAPABILITY_NOT_INSTALLED"):
        await gateway.complete(bad, "owner", messages, JsonSnapshot.capture([]))
    assert not calls


@pytest.mark.asyncio
async def test_required_current_task_and_tail_are_never_dropped_to_make_room():
    run, execution, profile, snapshot, messages, _ = context_values()
    changed = messages.as_object_array()
    changed[-1]["content"] = "不可丢弃的工具证据" * 2000
    summary = Summarizer()
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED") as error:
        await ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT).build(
            run, "owner", JsonSnapshot.capture(changed), JsonSnapshot.capture([]))
    assert error.value.status == 429 and not summary.calls
    assert snapshot.source_revision == 3 and changed[-1]["tool_call_id"] == "a"


@pytest.mark.asyncio
async def test_summary_input_itself_too_large_has_the_same_error_without_dispatch():
    run, execution, profile, snapshot, _, current = context_values()
    first = snapshot.turns[0].model_copy(update={"messages": JsonSnapshot.capture([
        {"role": "user", "content": "较早原始任务"}, {"role": "assistant", "content": "很长的成功历史" * 5000}])})
    snapshot = snapshot.model_copy(update={"turns": (first, *snapshot.turns[1:])})
    history = [m for turn in snapshot.turns for m in turn.messages.as_object_array()]
    run = run.model_copy(update={"history": JsonSnapshot.capture(history)})
    messages = JsonSnapshot.capture([{"role": "system", "content": "固定规则"}, *history, *current])
    summary = Summarizer()
    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED"):
        await ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT).build(
            run, "owner", messages, JsonSnapshot.capture([]))
    assert not summary.calls


@pytest.mark.asyncio
async def test_summary_result_that_still_cannot_fit_is_not_staged_or_recursively_summarized():
    run, execution, profile, snapshot, messages, _ = context_values()
    execution = execution.model_copy(update={"budget": execution.budget.model_copy(update={"max_input_tokens": 4500,
        "max_output_tokens": 5000}), "route": execution.route.model_copy(update={"max_output_tokens": 5000})})
    calls = []

    async def long_summary(*args):
        calls.append("summary")
        return ModelResult(model="controlled", content=json.dumps({"summary": "s" * 3900}), finish_reason="stop",
            request_id="test", usage=ModelUsage(), model_call_id="summary", model_attempt_id="attempt", request_fingerprint="a" * 64)

    class Staging:
        async def load_committed(self, *args):
            return None

        async def stage(self, *args):
            calls.append("stage")

    with pytest.raises(RuntimeFault, match="CONTEXT_BUDGET_EXCEEDED"):
        await ContextPipeline(Source(snapshot), Projection(), long_summary, execution, profile, PROMPT, Staging()).build(
            run, "owner", messages, JsonSnapshot.capture([]))
    assert calls == ["summary"]
