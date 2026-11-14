"""纯事件投影与原始线格式兼容用例定义。"""

from datetime import UTC, datetime

import pytest

from harness.contracts.events import RunAcceptedPayload, RuntimeEvent
from harness.runtime.event_projection import project_events


def event(seq, kind, **payload):
    return RuntimeEvent(run_id="run", seq=seq, type=kind, timestamp=datetime(2026, 11, 7, tzinfo=UTC),
                        payload=payload)


def history():
    return [
        event(1, "run.accepted", release_id="release", manifest_hash="old",
              environment_profile_hash="env", future_field="保留"),
        event(2, "answer.started", draft_id="draft", producer="model", model_call_id="call",
              model_attempt_id="attempt"),
        event(3, "answer.delta", draft_id="draft", model_attempt_id="attempt", delta="甲"),
        event(4, "answer.delta", draft_id="draft", model_attempt_id="attempt", delta="乙"),
    ]


def test_replay_duplicate_unknown_event_and_single_completion():
    initial = history()
    assert isinstance(initial[0].typed_payload(), RunAcceptedPayload)
    assert initial[0].model_dump()["payload"]["future_field"] == "保留"
    complete = event(6, "run.completed", committed_draft_id="draft", commit_id="commit", result_hash="hash")
    state = project_events("run", [*initial, initial[2], event(5, "future.extra"), complete, complete])
    assert state.cursor == 6 and state.answer == "甲乙"
    assert state.completion is not None and state.completion.commit_id == "commit"


@pytest.mark.parametrize("scenario,code", [
    ("gap", "EVENT_SEQUENCE_GAP"),
    ("conflict", "EVENT_SEQUENCE_CONFLICT"),
    ("late_delta", "EVENT_DRAFT_MISMATCH"),
    ("wrong_attempt", "EVENT_ATTEMPT_MISMATCH"),
    ("second_completion", "EVENT_DUPLICATE_TERMINAL"),
])
def test_conflicts_and_invalidated_draft(scenario, code):
    initial = history()
    invalidated = event(5, "answer.invalidated", draft_id="draft", reason="failed")
    complete = event(5, "run.completed", committed_draft_id="draft", commit_id="commit", result_hash="hash")
    suffixes = {
        "gap": [event(6, "future.extra")],
        "conflict": [event(4, "answer.delta", draft_id="draft", model_attempt_id="attempt", delta="改")],
        "late_delta": [invalidated, event(6, "answer.delta", draft_id="draft", model_attempt_id="attempt", delta="迟")],
        "wrong_attempt": [event(5, "answer.delta", draft_id="draft", model_attempt_id="other", delta="迟")],
        "second_completion": [complete, complete.model_copy(update={"seq": 6})],
    }
    assert project_events("run", [*initial, invalidated]).answer is None
    with pytest.raises(ValueError, match=code):
        project_events("run", [*initial, *suffixes[scenario]])
