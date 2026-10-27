"""授权成功历史、完整上下文构建与有来源摘要的纯契约。"""

from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import ClaimedRun
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import Contract, Identifier


class ContextProfile(Contract):
    version: Literal["success-context@1"] = "success-context@1"
    history_mode: Literal["success_history", "current_only"] = "success_history"
    soft_limit_percent: int = Field(default=75, ge=1, le=100, strict=True)
    keep_tail_turns: int = Field(default=2, ge=1, le=20, strict=True)
    max_history_turns: int = Field(default=256, ge=1, le=1000, strict=True)
    max_pending_operations: int = Field(default=12, ge=1, le=128, strict=True)
    max_summary_chars: int = Field(default=4000, ge=1, le=8000, strict=True)
    summary_prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class MemoryBoundaryProfile(Contract):
    version: Literal["session-memory-boundary@1"] = "session-memory-boundary@1"
    memory_profile_ref: Literal["session-only@1"] = "session-only@1"
    automatic_long_term_profile: Literal[False] = False
    semantic_history_recall: Literal[False] = False
    complex_fact_extraction: Literal[False] = False


class ContextTurn(Contract):
    run_id: Identifier
    revision: int = Field(ge=1, strict=True)
    messages: JsonSnapshot


def source_hash(revision: int, turns: tuple[ContextTurn, ...]) -> str:
    return canonical_hash({"source_revision": revision, "turns": [
        {"run_id": turn.run_id, "revision": turn.revision, "messages_hash": turn.messages.content_hash}
        for turn in turns
    ]})


class ContextSnapshot(Contract):
    source_revision: int = Field(ge=0, strict=True)
    turns: tuple[ContextTurn, ...]
    protected_state: JsonSnapshot


class ModelCallPurpose(Contract):
    kind: Literal["context_summary"] = "context_summary"
    source_revision: int = Field(ge=0, strict=True)
    covered_turn_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1000)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    summary_prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def unique_turns(self) -> Self:
        if len(set(self.covered_turn_ids)) != len(self.covered_turn_ids):
            raise ValueError("摘要来源turn不能重复")
        return self

    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class SummaryText(Contract):
    summary: str = Field(min_length=1, max_length=8000)


class SummaryCandidate(Contract):
    purpose: ModelCallPurpose
    summary: str = Field(min_length=1, max_length=8000)
    summary_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_call_id: Identifier
    model_attempt_id: Identifier

    @model_validator(mode="after")
    def bind_content(self) -> Self:
        if self.summary_hash != canonical_hash({"purpose": self.purpose.model_dump(mode="json"), "summary": self.summary}):
            raise ValueError("摘要候选内容与来源hash不一致")
        return self


class BuiltContext(Contract):
    messages: JsonSnapshot
    input_tokens: int = Field(ge=0)
    source_revision: int = Field(ge=0)
    candidate: SummaryCandidate | None = None


class ContextSource(Protocol):
    async def load(self, run: ClaimedRun, owner: str, profile: ContextProfile) -> ContextSnapshot: ...


class ContextBuilder(Protocol):
    async def build(self, run: ClaimedRun, owner: str, messages: JsonSnapshot, tools: JsonSnapshot) -> BuiltContext: ...


class ContextArtifactProjection(Protocol):
    async def project_messages(self, run: ClaimedRun, owner: str, messages: JsonSnapshot,
                               *, source_run_id: str | None = None) -> JsonSnapshot: ...


class SummaryCommitProfile(Contract):
    version: Literal["summary-commit@1"] = "summary-commit@1"
    stage_ttl_seconds: int = Field(default=900, ge=1, le=3600, strict=True)
    max_stage_bytes: int = Field(default=32768, ge=1024, le=65536, strict=True)


class CommittedSummary(Contract):
    candidate: SummaryCandidate
    covered_revision: int = Field(ge=1, strict=True)
    committed_revision: int = Field(ge=1, strict=True)
    cursor_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    commit_run_id: Identifier
    commit_id: Identifier

    @model_validator(mode="after")
    def bind_cursor(self) -> Self:
        if (self.covered_revision != len(self.candidate.purpose.covered_turn_ids)
            or self.committed_revision != self.candidate.purpose.source_revision + 1
            or self.covered_revision >= self.committed_revision
            or self.cursor_hash != summary_cursor(self.candidate, self.covered_revision,
                self.committed_revision, self.commit_run_id, self.commit_id)):
            raise ValueError("已提交摘要的来源和游标不一致")
        return self


def summary_cursor(candidate: SummaryCandidate, covered_revision: int, committed_revision: int,
                   commit_run_id: str, commit_id: str) -> str:
    return canonical_hash({"summary_hash": candidate.summary_hash, "covered_revision": covered_revision,
        "committed_revision": committed_revision, "commit_run_id": commit_run_id, "commit_id": commit_id})


class SummaryStore(Protocol):
    async def load_committed(self, run: ClaimedRun, owner: str) -> CommittedSummary | None: ...

    async def stage(self, run: ClaimedRun, owner: str, candidate: SummaryCandidate) -> None: ...
