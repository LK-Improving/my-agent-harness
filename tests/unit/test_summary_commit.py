"""跨轮摘要复用、纠错和尾部工具组保护的单元定义；未运行。"""

import json

import pytest

from harness.context.pipeline import ContextPipeline
from harness.context.requests import require_complete_tool_groups
from harness.contracts.context import (CommittedSummary, ContextSnapshot, ContextTurn, ModelCallPurpose,
    SummaryCandidate, source_hash, summary_cursor)
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from tests.unit.test_context_pipeline import PROMPT, Projection, Source, Summarizer, context_values


class Summaries:
    def __init__(self, committed=None, fail=False):
        self.committed, self.fail, self.staged = committed, fail, []

    async def load_committed(self, run, owner):
        return self.committed

    async def stage(self, run, owner, candidate):
        if self.fail:
            raise RuntimeFault("SUMMARY_STAGE_UNAVAILABLE", 503)
        self.staged.append(candidate)


def committed_values(*, soft=100):
    run, execution, profile, snapshot, _, current = context_values(soft=soft)
    purpose = ModelCallPurpose(source_revision=3, covered_turn_ids=(snapshot.turns[0].run_id,),
        source_hash=source_hash(3, snapshot.turns[:1]), summary_prompt_hash=profile.summary_prompt_hash)
    text = "旧条件为周末；如用户纠正须以当前输入为准。"
    candidate = SummaryCandidate(purpose=purpose, summary=text,
        summary_hash=canonical_hash({"purpose": purpose.model_dump(mode="json"), "summary": text}),
        model_call_id="old-summary", model_attempt_id="old-attempt")
    committed = CommittedSummary(candidate=candidate, covered_revision=1, committed_revision=4,
        cursor_hash=summary_cursor(candidate, 1, 4, "old-3", "commit-4"), commit_run_id="old-3", commit_id="commit-4")
    fourth = ContextTurn(run_id="old-3", revision=4, messages=JsonSnapshot.capture([
        {"role": "user", "content": "切换主题"}, {"role": "assistant", "content": "请补充日期"}]))
    snapshot = ContextSnapshot(source_revision=4, turns=(*snapshot.turns, fourth), protected_state=snapshot.protected_state)
    history = [m for turn in snapshot.turns for m in turn.messages.as_object_array()]
    current[0] = {"role": "user", "content": "纠正：改为工作日，保留未完成事项。"}
    run = run.model_copy(update={"history": JsonSnapshot.capture(history), "request": run.request.model_copy(
        update={"message": current[0]["content"], "expected_revision": 4})})
    messages = JsonSnapshot.capture([{"role": "system", "content": "固定规则"}, *history, *current])
    return run, execution, profile, snapshot, messages, current, committed


@pytest.mark.asyncio
async def test_committed_summary_reuse_keeps_current_correction_pending_and_complete_tail():
    run, execution, profile, snapshot, messages, current, committed = committed_values()
    summary, store = Summarizer(), Summaries(committed)
    built = await ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT, store).build(
        run, "owner", messages, JsonSnapshot.capture([]))
    output = built.messages.as_object_array()
    assert not summary.calls and not store.staged and built.candidate is None
    assert output[-len(current):] == current and "unknown" in built.messages.canonical_text
    assert {"role": "user", "content": "旧任务0"} not in output
    assert all(turn.messages.as_object_array()[0] in output for turn in snapshot.turns[-2:])
    require_complete_tool_groups(output)
    assert output[1]["role"] == "user" and "context_summary_data" in output[1]["content"]


@pytest.mark.asyncio
async def test_new_summary_uses_original_turns_not_the_previous_summary():
    run, execution, profile, snapshot, messages, _, committed = committed_values(soft=1)
    summary, store = Summarizer(), Summaries(committed)
    built = await ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT, store).build(
        run, "owner", messages, JsonSnapshot.capture([]))
    body = json.loads(summary.calls[0][0].as_object_array()[1]["content"])
    assert len(summary.calls) == 1 and len(store.staged) == 1
    assert "context_summary_data" not in summary.calls[0][0].canonical_text
    assert [turn["run_id"] for turn in body["turns"]] == ["old-0", "old-1"]
    assert built.candidate.purpose.covered_turn_ids == ("old-0", "old-1")


@pytest.mark.asyncio
async def test_stage_failure_never_returns_a_publishable_built_context():
    run, execution, profile, snapshot, messages, _ = context_values()
    summary = Summarizer()
    with pytest.raises(RuntimeFault, match="SUMMARY_STAGE_UNAVAILABLE"):
        await ContextPipeline(Source(snapshot), Projection(), summary, execution, profile, PROMPT,
            Summaries(fail=True)).build(run, "owner", messages, JsonSnapshot.capture([]))
    assert len(summary.calls) == 1 and snapshot.source_revision == 3
