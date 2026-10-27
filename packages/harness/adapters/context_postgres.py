"""授权成功历史的PG来源；锁内只读与审计，摘要模型调用在事务外。"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

from harness.adapters.artifacts_postgres import PostgresArtifacts
from harness.adapters.postgres.connection import Connection
from harness.context.pipeline import summary_messages
from harness.context.boundary import require_memory_boundary
from harness.context.projection import project_model_messages
from harness.contracts.artifacts import ArtifactProfile
from harness.contracts.context import ContextProfile, ContextSnapshot, ContextTurn, ModelCallPurpose, source_hash
from harness.contracts.runtime import ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.state import TaskState

if TYPE_CHECKING:
    from harness.adapters.execution_postgres import PostgresExecutionStore


class PostgresContextSource:
    def __init__(self, execution: PostgresExecutionStore) -> None:
        self.execution = execution

    async def load_in_transaction(self, conn: Connection, run: ClaimedRun, owner: str,
                                   profile: ContextProfile) -> ContextSnapshot:
        require_memory_boundary(run.manifest)
        current, manifest = await self.execution.authorize_extension_in_transaction(conn, run.principal, owner, run.run_id)
        if (current["session_id"] != run.session_id or manifest.content_hash() != run.manifest.content_hash()
            or manifest.profiles.as_object().get("context") != profile.model_dump(mode="json")):
            raise RuntimeFault("CONTEXT_MANIFEST_MISMATCH", 409)
        session = await self.execution._session(conn, run.principal, run.session_id, lock=True)
        revision = session["revision"]
        if revision != run.request.expected_revision:
            raise RuntimeFault("CONTEXT_SOURCE_REVISION_CHANGED", 409)
        turns = []
        protected_task = None
        if profile.history_mode == "success_history":
            rows = await (await conn.execute("""SELECT t.run_id,t.revision,t.messages,r.result->'task' AS task_state FROM harness.turns t
                JOIN harness.runs r USING(run_id) JOIN harness.commits c USING(run_id)
                WHERE t.session_id=%s AND r.status='succeeded' ORDER BY t.revision LIMIT %s""",
                (run.session_id, profile.max_history_turns + 1))).fetchall()
            if len(rows) > profile.max_history_turns:
                raise RuntimeFault("CONTEXT_HISTORY_LIMIT", 429)
            if len(rows) != revision or any(r["revision"] != i + 1 for i, r in enumerate(rows)):
                raise RuntimeFault("CONTEXT_HISTORY_INCOMPLETE", 409)
            for row in rows:
                source, source_session = await self.execution._authorized_run(conn, run.principal, row["run_id"])
                await self.execution._authorize_public_resource(conn, run.principal, source, source_session)
                messages = project_model_messages(row["messages"])
                if any(m["role"] == "system" for m in messages.as_object_array()):
                    raise RuntimeFault("CONTEXT_HISTORY_ROLE_INVALID", 409)
                turns.append(ContextTurn(run_id=row["run_id"], revision=row["revision"], messages=messages))
            if rows and rows[-1]["task_state"] is not None:
                task = TaskState.model_validate(rows[-1]["task_state"])
                if task.source_turn_id != rows[-1]["run_id"]:
                    raise RuntimeFault("CONTEXT_TASK_SOURCE_MISMATCH", 409)
                if task.pending_items or task.blockers:
                    protected_task = task.model_dump(mode="json", include={"objective", "source_turn_id", "constraints", "pending_items", "blockers"})
        pending = await (await conn.execute("""SELECT o.operation_id,o.state,o.tool->>'name' AS tool_name,
            o.arguments,o.args_hash,a.state AS approval_state FROM harness.operations o
            JOIN harness.approvals a USING(approval_id) WHERE o.run_id=%s
            AND o.state IN ('prepared','dispatched','unknown') ORDER BY o.operation_id LIMIT %s""",
            (run.run_id, profile.max_pending_operations + 1))).fetchall()
        if len(pending) > profile.max_pending_operations:
            raise RuntimeFault("CONTEXT_PENDING_LIMIT", 429)
        protected: dict[str, object] = {}
        if pending:
            protected["operations"] = pending
        if protected_task is not None:
            protected["task"] = protected_task
        snapshot = ContextSnapshot(source_revision=revision, turns=tuple(turns), protected_state=JsonSnapshot.capture(protected))
        await self.execution._audit(conn, run.principal, "context.history_authorized", run.run_id)
        return snapshot

    async def load(self, run: ClaimedRun, owner: str, profile: ContextProfile) -> ContextSnapshot:
        async with self.execution._transaction() as conn:
            return await self.load_in_transaction(conn, run, owner, profile)

    async def verify_summary(self, conn: Connection, run: ClaimedRun, owner: str,
                              messages: JsonSnapshot, purpose: ModelCallPurpose) -> None:
        configuration = run.manifest.profiles.as_object()
        profile = ContextProfile.model_validate(configuration.get("context"))
        snapshot = await self.load_in_transaction(conn, run, owner, profile)
        older = snapshot.turns[:-profile.keep_tail_turns]
        if (profile.history_mode != "success_history" or not older
            or purpose.source_revision != snapshot.source_revision
            or purpose.covered_turn_ids != tuple(t.run_id for t in older)
            or purpose.source_hash != source_hash(snapshot.source_revision, older)
            or purpose.summary_prompt_hash != profile.summary_prompt_hash):
            raise RuntimeFault("CONTEXT_SUMMARY_SOURCE_MISMATCH", 409)
        supplied = messages.as_object_array()
        if len(supplied) != 2 or supplied[0].get("role") != "system" or not isinstance(supplied[0].get("content"), str):
            raise RuntimeFault("CONTEXT_SUMMARY_INPUT_MISMATCH", 409)
        prompt = cast(str, supplied[0]["content"])
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != purpose.summary_prompt_hash:
            raise RuntimeFault("CONTEXT_SUMMARY_PROMPT_MISMATCH", 409)
        artifacts = PostgresArtifacts(self.execution, ArtifactProfile.model_validate(configuration.get("artifacts")))
        projected = []
        for turn in older:
            projected.append(turn.model_copy(update={"messages": await artifacts.project_stored_messages(
                conn, run.principal, run.session_id, turn.messages, source_run_id=turn.run_id)}))
        expected = summary_messages(prompt, snapshot.source_revision, tuple(projected), purpose.source_hash)
        if expected != messages:
            raise RuntimeFault("CONTEXT_SUMMARY_INPUT_MISMATCH", 409)
