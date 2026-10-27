"""独立上下文流水线；候选经受限端口暂存，游标只由成功事务推进。"""

import hashlib
from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from harness.context.projection import project_model_messages
from harness.context.boundary import context_input_limit, require_context_capacity, require_memory_boundary
from harness.context.requests import envelope_tokens
from harness.contracts.context import (
    BuiltContext, ContextArtifactProjection, ContextProfile, ContextSnapshot, ContextSource,
    ContextTurn, ModelCallPurpose, SummaryCandidate, SummaryStore, SummaryText, source_hash,
)
from harness.contracts.execution import ExecutionProfile
from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.models import ModelResult
from harness.contracts.runtime import ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot

Summarize = Callable[[ClaimedRun, str, JsonSnapshot, ModelCallPurpose], Awaitable[ModelResult]]


def summary_messages(prompt: str, revision: int, turns: tuple[ContextTurn, ...], identity_hash: str | None = None) -> JsonSnapshot:
    return JsonSnapshot.capture([
        {"role": "system", "content": prompt},
        {"role": "user", "content": canonical_json({
            "source_revision": revision, "source_hash": identity_hash or source_hash(revision, turns),
            "turns": [{"run_id": t.run_id, "revision": t.revision, "messages": t.messages.materialize()} for t in turns],
        })},
    ])


class ContextPipeline:
    def __init__(self, source: ContextSource, artifacts: ContextArtifactProjection,
                 summarize: Summarize, execution: ExecutionProfile, profile: ContextProfile,
                 summary_prompt: str, summaries: SummaryStore | None = None) -> None:
        if hashlib.sha256(summary_prompt.encode("utf-8")).hexdigest() != profile.summary_prompt_hash:
            raise ValueError("摘要提示词与固定profile不一致")
        self.source, self.artifacts, self.summarize = source, artifacts, summarize
        self.execution, self.profile, self.summary_prompt = execution, profile, summary_prompt
        self.summaries = summaries

    def count(self, messages: JsonSnapshot, tools: JsonSnapshot) -> int:
        return envelope_tokens(self.execution.route.model, messages.as_object_array(), tools.as_object_array(),
                               self.execution.budget.max_output_tokens)

    @property
    def hard_limit(self) -> int:
        return context_input_limit(self.execution)

    async def authorize(self, run: ClaimedRun, owner: str) -> None:
        require_memory_boundary(run.manifest)
        snapshot = await self.source.load(run, owner, self.profile)
        for turn in snapshot.turns:
            if any(m["role"] == "tool" for m in turn.messages.as_object_array()):
                await self.artifacts.project_messages(run, owner, turn.messages, source_run_id=turn.run_id)
        if self.summaries is not None:
            await self.summaries.load_committed(run, owner)

    async def build(self, run: ClaimedRun, owner: str, messages: JsonSnapshot, tools: JsonSnapshot) -> BuiltContext:
        require_memory_boundary(run.manifest)
        if run.manifest.profiles.as_object().get("context") != self.profile.model_dump(mode="json"):
            raise RuntimeFault("CONTEXT_PROFILE_MISMATCH", 409)
        snapshot = await self.source.load(run, owner, self.profile)
        if run.manifest.profiles.as_object().get("summary_commit") is not None and self.summaries is None:
            raise RuntimeFault("SUMMARY_STORE_NOT_INSTALLED", 503)
        committed = await self.summaries.load_committed(run, owner) if self.summaries is not None else None
        raw = project_model_messages(messages.as_object_array()).as_object_array()
        system_count = 0
        while system_count < len(raw) and raw[system_count]["role"] == "system":
            system_count += 1
        system = raw[:system_count]
        expected = run.history.as_object_array() if self.profile.history_mode == "success_history" else []
        history = [m for turn in snapshot.turns for m in turn.messages.as_object_array()]
        boundary = system_count + len(expected)
        if raw[system_count:boundary] != expected or history != expected:
            raise RuntimeFault("CONTEXT_HISTORY_PREFIX_CHANGED", 409)
        current = raw[boundary:]
        task = JsonSnapshot.capture({"role": "user", "content": run.request.message}).as_object()
        if not current or current[0] != task:
            raise RuntimeFault("CONTEXT_CURRENT_TASK_CHANGED", 409)
        if any(m["role"] == "system" for m in current):
            raise RuntimeFault("CONTEXT_SYSTEM_MESSAGE_IN_TASK", 422)
        current = (await self.artifacts.project_messages(run, owner, JsonSnapshot.capture(current))).as_object_array()
        projected = []
        for turn in snapshot.turns:
            projected.append(turn.model_copy(update={"messages": await self.artifacts.project_messages(run, owner, turn.messages,
                source_run_id=turn.run_id)}) if any(m["role"] == "tool" for m in turn.messages.as_object_array()) else turn)
        projected_turns = tuple(projected)
        history = [m for turn in projected_turns for m in turn.messages.as_object_array()]
        covered = 0
        if committed is not None:
            covered = committed.covered_revision
            if (covered > len(projected_turns) - self.profile.keep_tail_turns
                or tuple(t.run_id for t in snapshot.turns[:covered]) != committed.candidate.purpose.covered_turn_ids):
                raise RuntimeFault("SUMMARY_CURSOR_CORRUPT", 409)
            history = [{"role": "user", "content": canonical_json({"kind": "context_summary_data",
                **committed.candidate.model_dump(mode="json")})},
                *[m for turn in projected_turns[covered:] for m in turn.messages.as_object_array()]]
        protected = []
        if snapshot.protected_state.as_object():
            protected = [{"role": "user", "content": canonical_json({"kind": "runtime_pending_state", "data": snapshot.protected_state.materialize()})}]
        full = project_model_messages([*system, *history, *protected, *current])
        required_tail = [m for turn in projected_turns[-self.profile.keep_tail_turns:] for m in turn.messages.as_object_array()]
        minimum = project_model_messages([*system, *required_tail, *protected, *current])
        # 必需内容已不可容纳时不再付费摘要，也不丢弃当前任务或半个工具事务组。
        require_context_capacity(self.count(minimum, tools), self.execution)
        count = self.count(full, tools)
        soft = self.hard_limit * self.profile.soft_limit_percent // 100
        older = snapshot.turns[:-self.profile.keep_tail_turns]
        if count <= soft or len(older) <= covered:
            require_context_capacity(count, self.execution)
            return BuiltContext(messages=full, input_tokens=count, source_revision=snapshot.source_revision)
        # 摘要来源绑定原始授权turn；供应商仅收到先经投影的同一前缀。
        projected_older = projected_turns[:len(older)]
        purpose = ModelCallPurpose(source_revision=snapshot.source_revision,
            covered_turn_ids=tuple(turn.run_id for turn in older), source_hash=source_hash(snapshot.source_revision, older),
            summary_prompt_hash=self.profile.summary_prompt_hash)
        summary_input = summary_messages(self.summary_prompt, snapshot.source_revision, projected_older, purpose.source_hash)
        require_context_capacity(self.count(summary_input, JsonSnapshot.capture([])), self.execution)
        result = await self.summarize(run, owner, summary_input, purpose)
        if result.tool_calls.as_object_array() or result.finish_reason not in {"stop", "end_turn"} or result.content is None:
            raise RuntimeFault("CONTEXT_SUMMARY_RESULT_INVALID", 422)
        try:
            text = SummaryText.model_validate_json(result.content, strict=True).summary
        except ValidationError:
            raise RuntimeFault("CONTEXT_SUMMARY_RESULT_INVALID", 422) from None
        if len(text) > self.profile.max_summary_chars:
            raise RuntimeFault("CONTEXT_SUMMARY_RESULT_TOO_LARGE", 422)
        candidate = SummaryCandidate(purpose=purpose, summary=text,
            summary_hash=canonical_hash({"purpose": purpose.model_dump(mode="json"), "summary": text}),
            model_call_id=result.model_call_id, model_attempt_id=result.model_attempt_id)
        summary = {"role": "user", "content": canonical_json({"kind": "context_summary_data", **candidate.model_dump(mode="json")})}
        tail = [m for turn in projected_turns[len(older):] for m in turn.messages.as_object_array()]
        built = project_model_messages([*system, summary, *tail, *protected, *current])
        count = self.count(built, tools)
        require_context_capacity(count, self.execution)
        if self.summaries is not None:
            await self.summaries.stage(run, owner, candidate)
        return BuiltContext(messages=built, input_tokens=count, source_revision=snapshot.source_revision, candidate=candidate)
