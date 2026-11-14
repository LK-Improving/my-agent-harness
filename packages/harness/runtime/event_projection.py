"""从一个 run 的连续持久事件派生公开草稿视图。"""

from collections.abc import Iterable
from dataclasses import dataclass

from harness.contracts.events import (
    AnswerDeltaPayload, AnswerInvalidatedPayload, AnswerStartedPayload, RunAcceptedPayload,
    RunCompletedPayload, RuntimeEvent, RunTerminatedPayload,
)
from harness.contracts.hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class EventProjection:
    cursor: int
    terminal: str | None
    draft_id: str | None
    answer: str | None
    completion: RunCompletedPayload | None


def project_events(run_id: str, events: Iterable[RuntimeEvent]) -> EventProjection:
    """相同序号及内容的重放幂等，缺口与冲突要求调用方重新回放。"""
    digests: list[str] = []
    invalidated: set[str] = set()
    chunks: list[str] = []
    draft: AnswerStartedPayload | None = None
    completion = None
    terminal = None
    attempt: str | None = None
    for event in events:
        if event.run_id != run_id:
            raise ValueError("EVENT_RUN_MISMATCH")
        digest = canonical_hash(event.model_dump(mode="json"))
        if event.seq <= len(digests):
            if digest != digests[event.seq - 1]:
                raise ValueError("EVENT_SEQUENCE_CONFLICT")
            continue
        if event.seq != len(digests) + 1:
            raise ValueError("EVENT_SEQUENCE_GAP")
        payload = event.typed_payload()
        if (event.seq == 1) != isinstance(payload, RunAcceptedPayload):
            raise ValueError("EVENT_ACCEPTED_ORDER_INVALID")
        digests.append(digest)
        if isinstance(payload, (RunCompletedPayload, RunTerminatedPayload)):
            if terminal is not None:
                raise ValueError("EVENT_DUPLICATE_TERMINAL")
            terminal = event.type
            if isinstance(payload, RunCompletedPayload):
                if draft is None or draft.draft_id != payload.committed_draft_id or not chunks:
                    raise ValueError("EVENT_COMPLETION_DRAFT_MISMATCH")
                completion = payload
            else:
                draft = None
                chunks.clear()
        elif isinstance(payload, (AnswerStartedPayload, AnswerDeltaPayload, AnswerInvalidatedPayload)):
            if terminal is not None:
                raise ValueError("EVENT_DRAFT_AFTER_TERMINAL")
            if isinstance(payload, AnswerStartedPayload):
                if draft is not None or payload.draft_id in invalidated:
                    raise ValueError("EVENT_DRAFT_REUSED")
                draft = payload
                attempt = payload.model_attempt_id
            elif draft is None or draft.draft_id != payload.draft_id:
                raise ValueError("EVENT_DRAFT_MISMATCH")
            elif isinstance(payload, AnswerInvalidatedPayload):
                invalidated.add(payload.draft_id)
                draft = None
                attempt = None
                chunks.clear()
            else:
                if attempt is not None and attempt != payload.model_attempt_id:
                    raise ValueError("EVENT_ATTEMPT_MISMATCH")
                attempt = payload.model_attempt_id
                chunks.append(payload.delta)
    return EventProjection(len(digests), terminal, draft.draft_id if draft else None,
                           "".join(chunks) if draft else None, completion)
