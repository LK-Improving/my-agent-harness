"""完整工具事务组、固定预算与本机终止证明的反例定义。"""

import json
from unittest.mock import patch

import pytest
from harness.adapters.development_auth import LiveDevelopmentAuthenticator
from harness.adapters.local_process import prove_terminated
from harness.context.requests import envelope_tokens, require_complete_tool_groups
from harness.contracts.execution import ProcessIdentity
from harness.contracts.runtime import Principal, RuntimeFault


def test_tool_groups_require_exact_call_ids_and_all_results() -> None:
    head = {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]}
    require_complete_tool_groups(
        [head, {"role": "tool", "tool_call_id": "b"}, {"role": "tool", "tool_call_id": "a"}]
    )
    for messages in (
        [head, {"role": "tool", "tool_call_id": "wrong"}],
        [head, {"role": "tool", "tool_call_id": "a"}],
        [head, {"role": "tool", "tool_call_id": "a"}, {"role": "user", "content": "next"}],
    ):
        with pytest.raises(RuntimeFault):
            require_complete_tool_groups(messages)
    small = envelope_tokens("model", [{"role": "user", "content": "查询"}], [], 128)
    large = envelope_tokens(
        "model", [{"role": "user", "content": "查询"}], [{"schema": "中文" * 100}], 128
    )
    assert large > small


def test_live_identity_file_revocation_and_no_scope_expansion(tmp_path) -> None:
    path = tmp_path / "identities.json"
    actor = Principal(
        tenant_id="t",
        subject_id="u",
        application_id="app",
        roles=frozenset({"user"}),
        resource_scopes=frozenset({"read"}),
    )
    path.write_text(json.dumps({"a" * 40: actor.model_dump(mode="json")}), "utf-8")
    auth = LiveDevelopmentAuthenticator(path, "development")
    assert auth.authenticate("Bearer " + "a" * 40) == actor
    path.write_text("{}", "utf-8")
    with pytest.raises(RuntimeFault):
        auth.current(actor)


def test_live_pid_is_not_termination_proof_even_without_heartbeat() -> None:
    identity = ProcessIdentity(
        host_id="local",
        boot_time=1000,
        pid=42,
        started_at=2000,
        namespace_id="host",
        execution_class="local-no-subprocess@1",
        boot_id="boot",
        start_marker="old-start",
    )
    with (
        patch("harness.adapters.local_process.current_process", return_value=identity),
        patch("harness.adapters.local_process.process_marker") as marker,
    ):
        marker.return_value = "old-start"
        with pytest.raises(RuntimeFault, match="OLD_WORKER_STILL_PRESENT"):
            prove_terminated(identity)
        marker.return_value = "new-start"
        assert prove_terminated(identity).reason == "pid_reused_with_different_start_time"


def test_wall_clock_change_is_not_reboot_proof() -> None:
    identity = ProcessIdentity(
        host_id="local",
        boot_time=1000,
        pid=42,
        started_at=2000,
        namespace_id="host",
        execution_class="local-no-subprocess@1",
        boot_id="same-boot",
        start_marker="same-process",
    )
    changed_clock = identity.model_copy(update={"boot_time": 5000})
    with (
        patch("harness.adapters.local_process.current_process", return_value=changed_clock),
        patch("harness.adapters.local_process.process_marker", return_value="same-process"),
    ):
        with pytest.raises(RuntimeFault, match="OLD_WORKER_STILL_PRESENT"):
            prove_terminated(identity)
