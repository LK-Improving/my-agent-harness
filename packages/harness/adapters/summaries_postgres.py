"""摘要受限暂存与成功事务发布；不创建模型调用或修改官方checkpoint。"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from psycopg.types.json import Jsonb
from pydantic import ValidationError

from harness.adapters.context_postgres import PostgresContextSource
from harness.adapters.postgres.connection import Connection
from harness.contracts.context import (CommittedSummary, ContextProfile, ModelCallPurpose, SummaryCandidate,
    SummaryCommitProfile, SummaryText, source_hash, summary_cursor)
from harness.contracts.hashing import canonical_hash, canonical_json
from harness.contracts.models import ModelResult, PreparedModelRequest
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, RuntimeFault

if TYPE_CHECKING:
    from harness.adapters.execution_postgres import PostgresExecutionStore


class PostgresSummaryStore:
    def __init__(self, execution: PostgresExecutionStore) -> None:
        self.execution = execution

    @staticmethod
    def policy(run: ClaimedRun) -> SummaryCommitProfile:
        configuration = run.manifest.profiles.as_object().get("summary_commit")
        if configuration is None:
            raise RuntimeFault("SUMMARY_COMMIT_PROFILE_REQUIRED", 503)
        return SummaryCommitProfile.model_validate(configuration)

    async def _model_proof(self, conn: Connection, run: ClaimedRun, candidate: SummaryCandidate,
                           creator_run_id: str) -> PreparedModelRequest:
        call = await self.execution._one(conn, "SELECT * FROM harness.model_calls WHERE run_id=%s AND model_call_id=%s",
                                         (creator_run_id, candidate.model_call_id))
        try:
            binding = ModelCallPurpose.model_validate(call["purpose_binding"])
            request = PreparedModelRequest.model_validate(call["request"])
            result = ModelResult.model_validate(call["result"])
        except ValidationError:
            raise RuntimeFault("SUMMARY_MODEL_PROOF_INVALID", 409) from None
        if (call["purpose"] != "context_summary" or call["state"] != "succeeded"
            or call["purpose_ref"] != candidate.purpose.fingerprint() or binding != candidate.purpose):
            raise RuntimeFault("SUMMARY_MODEL_PROOF_INVALID", 409)
        if (request.fingerprint() != call["request_fingerprint"] or request.context.run_id != creator_run_id
            or request.manifest_hash != run.manifest.content_hash() or request.tools.as_object_array()
            or result.request_fingerprint != request.fingerprint() or result.model_call_id != candidate.model_call_id
            or result.model_attempt_id != candidate.model_attempt_id or request.model_attempt_id != candidate.model_attempt_id
            or result.tool_calls.as_object_array() or result.finish_reason not in {"stop", "end_turn"}
            or result.content is None):
            raise RuntimeFault("SUMMARY_MODEL_PROOF_INVALID", 409)
        profile = ContextProfile.model_validate(run.manifest.profiles.as_object().get("context"))
        try:
            text = SummaryText.model_validate_json(result.content, strict=True).summary
        except ValidationError:
            raise RuntimeFault("SUMMARY_MODEL_CONTENT_MISMATCH", 409) from None
        if (text != candidate.summary or len(text) > profile.max_summary_chars
            or (result.usage.input_tokens or 0) > request.budget.input_tokens
            or (result.usage.output_tokens or 0) > request.max_output_tokens):
            raise RuntimeFault("SUMMARY_MODEL_CONTENT_MISMATCH", 409)
        return request

    async def _load_committed(self, conn: Connection, run: ClaimedRun, owner: str) -> CommittedSummary | None:
        self.policy(run)
        source = PostgresContextSource(self.execution)
        profile = ContextProfile.model_validate(run.manifest.profiles.as_object().get("context"))
        snapshot = await source.load_in_transaction(conn, run, owner, profile)
        session = await self.execution._session(conn, run.principal, run.session_id, lock=True)
        row = await (await conn.execute("""SELECT s.*,r.status AS creator_status,t.revision AS turn_revision,g.candidate_hash AS staged_hash
            FROM harness.session_summaries s JOIN harness.runs r ON r.run_id=s.commit_run_id
            JOIN harness.commits c ON c.commit_id=s.commit_id AND c.run_id=s.commit_run_id
            JOIN harness.turns t ON t.run_id=s.commit_run_id AND t.session_id=s.session_id
            JOIN harness.summary_candidates g ON g.run_id=s.commit_run_id AND g.session_id=s.session_id
                AND g.commit_id=s.commit_id AND g.state='committed'
            WHERE s.session_id=%s""", (run.session_id,))).fetchone()
        if row is None:
            orphan = await (await conn.execute("SELECT 1 FROM harness.session_summaries WHERE session_id=%s", (run.session_id,))).fetchone()
            if session["summary_cursor_hash"] is not None or orphan is not None:
                raise RuntimeFault("SUMMARY_CURSOR_CORRUPT", 409)
            return None
        try:
            committed = CommittedSummary.model_validate({name: row[name] for name in CommittedSummary.model_fields})
        except ValidationError:
            raise RuntimeFault("SUMMARY_CURSOR_CORRUPT", 409) from None
        candidate = committed.candidate
        covered = snapshot.turns[:committed.covered_revision]
        if (row["creator_status"] != "succeeded" or row["turn_revision"] != committed.committed_revision
            or row["staged_hash"] != canonical_hash(candidate.model_dump(mode="json"))
            or committed.committed_revision != candidate.purpose.source_revision + 1
            or committed.committed_revision > snapshot.source_revision
            or candidate.purpose.summary_prompt_hash != profile.summary_prompt_hash
            or committed.covered_revision != len(candidate.purpose.covered_turn_ids)
            or tuple(t.run_id for t in covered) != candidate.purpose.covered_turn_ids
            or candidate.purpose.source_hash != source_hash(candidate.purpose.source_revision, covered)
            or committed.covered_revision > candidate.purpose.source_revision - profile.keep_tail_turns
            or committed.cursor_hash != summary_cursor(candidate, committed.covered_revision,
                committed.committed_revision, committed.commit_run_id, committed.commit_id)
            or session["summary_cursor_hash"] != committed.cursor_hash):
            raise RuntimeFault("SUMMARY_CURSOR_CORRUPT", 409)
        await self._model_proof(conn, run, candidate, committed.commit_run_id)
        return committed

    async def load_committed(self, run: ClaimedRun, owner: str) -> CommittedSummary | None:
        async with self.execution._transaction() as conn:
            return await self._load_committed(conn, run, owner)

    async def _candidate_proof(self, conn: Connection, run: ClaimedRun, owner: str,
                               candidate: SummaryCandidate) -> None:
        try:
            candidate = SummaryCandidate.model_validate(candidate.model_dump(mode="python"))
        except ValidationError:
            raise RuntimeFault("SUMMARY_CANDIDATE_INVALID", 422) from None
        if len(canonical_json(candidate.model_dump(mode="json")).encode("utf-8")) > self.policy(run).max_stage_bytes:
            raise RuntimeFault("SUMMARY_STAGE_CAPACITY", 429)
        request = await self._model_proof(conn, run, candidate, run.run_id)
        await PostgresContextSource(self.execution).verify_summary(conn, run, owner, request.messages, candidate.purpose)

    async def stage(self, run: ClaimedRun, owner: str, candidate: SummaryCandidate) -> None:
        async with self.execution._transaction() as conn:
            await self._candidate_proof(conn, run, owner, candidate)
            previous = await self._load_committed(conn, run, owner)
            if previous is not None and len(candidate.purpose.covered_turn_ids) <= previous.covered_revision:
                raise RuntimeFault("SUMMARY_CURSOR_NOT_ADVANCED", 409)
            session = await self.execution._session(conn, run.principal, run.session_id, lock=True)
            current = await self.execution._one(conn, "SELECT expires_at,clock_timestamp() AS now FROM harness.runs WHERE run_id=%s", (run.run_id,))
            digest = canonical_hash(candidate.model_dump(mode="json"))
            old = await (await conn.execute("SELECT * FROM harness.summary_candidates WHERE run_id=%s FOR UPDATE", (run.run_id,))).fetchone()
            if old is not None:
                if (old["candidate_hash"] != digest or old["candidate_hash"] != canonical_hash(old["candidate"])
                    or old["expected_cursor_hash"] != session["summary_cursor_hash"]):
                    raise RuntimeFault("SUMMARY_STAGE_CONFLICT", 409)
                if old["state"] != "staged" or current["now"] >= old["expires_at"]:
                    raise RuntimeFault("SUMMARY_STAGE_UNAVAILABLE", 409)
                return
            expires = min(current["expires_at"], current["now"] + timedelta(seconds=self.policy(run).stage_ttl_seconds))
            await conn.execute("""INSERT INTO harness.summary_candidates
                (run_id,session_id,source_revision,expected_cursor_hash,candidate,candidate_hash,expires_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s)""", (run.run_id, run.session_id, candidate.purpose.source_revision,
                session["summary_cursor_hash"], Jsonb(candidate.model_dump(mode="json")), digest, expires))
            await self.execution._audit(conn, run.principal, "summary.staged", digest)

    async def commit_in_transaction(self, conn: Connection, owner: str, run: ClaimedRun,
                                     answer: AnswerCandidate, commit_id: str) -> None:
        self.policy(run)
        await self.execution.authorize_extension_in_transaction(conn, run.principal, owner, run.run_id)
        session = await self.execution._session(conn, run.principal, run.session_id, lock=True)
        if session["revision"] != run.request.expected_revision:
            raise RuntimeFault("SUMMARY_SOURCE_CAS_CONFLICT", 409)
        staged = await (await conn.execute("SELECT *,clock_timestamp() AS now FROM harness.summary_candidates WHERE run_id=%s FOR UPDATE", (run.run_id,))).fetchone()
        if staged is None:
            if session["summary_cursor_hash"] is not None:
                await self._load_committed(conn, run, owner)
            return
        if (staged["session_id"] != run.session_id or staged["state"] != "staged"
            or staged["now"] >= staged["expires_at"] or staged["source_revision"] != session["revision"]
            or staged["expected_cursor_hash"] != session["summary_cursor_hash"]):
            raise RuntimeFault("SUMMARY_SOURCE_CAS_CONFLICT", 409)
        try:
            candidate = SummaryCandidate.model_validate(staged["candidate"])
        except ValidationError:
            raise RuntimeFault("SUMMARY_STAGE_CORRUPT", 409) from None
        if canonical_hash(candidate.model_dump(mode="json")) != staged["candidate_hash"]:
            raise RuntimeFault("SUMMARY_STAGE_CORRUPT", 409)
        await self._candidate_proof(conn, run, owner, candidate)
        previous = await self._load_committed(conn, run, owner)
        covered = len(candidate.purpose.covered_turn_ids)
        if previous is not None and covered <= previous.covered_revision:
            raise RuntimeFault("SUMMARY_CURSOR_NOT_ADVANCED", 409)
        call = await self.execution._one(conn, "SELECT request,purpose,state FROM harness.model_calls WHERE run_id=%s AND model_call_id=%s", (run.run_id, answer.model_call_id))
        frame = canonical_json({"kind": "context_summary_data", **candidate.model_dump(mode="json")})
        request = PreparedModelRequest.model_validate(call["request"])
        if (call["purpose"] != "answer" or call["state"] != "succeeded"
            or not any(m.get("role") == "user" and m.get("content") == frame for m in request.messages.as_object_array())):
            raise RuntimeFault("SUMMARY_NOT_USED_BY_FINAL_ANSWER", 409)
        cursor = summary_cursor(candidate, covered, session["revision"] + 1, run.run_id, commit_id)
        cas = await (await conn.execute("""UPDATE harness.sessions SET summary_cursor_hash=%s
            WHERE session_id=%s AND revision=%s AND summary_cursor_hash IS NOT DISTINCT FROM %s RETURNING session_id""",
            (cursor, run.session_id, staged["source_revision"], staged["expected_cursor_hash"]))).fetchone()
        if cas is None:
            raise RuntimeFault("SUMMARY_SOURCE_CAS_CONFLICT", 409)
        await conn.execute("""INSERT INTO harness.session_summaries
            (session_id,candidate,covered_revision,committed_revision,cursor_hash,commit_run_id,commit_id)
            VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(session_id) DO UPDATE SET
            candidate=EXCLUDED.candidate,covered_revision=EXCLUDED.covered_revision,
            committed_revision=EXCLUDED.committed_revision,cursor_hash=EXCLUDED.cursor_hash,
            commit_run_id=EXCLUDED.commit_run_id,commit_id=EXCLUDED.commit_id""",
            (run.session_id, Jsonb(candidate.model_dump(mode="json")), covered, session["revision"] + 1, cursor, run.run_id, commit_id))
        await conn.execute("UPDATE harness.summary_candidates SET state='committed',commit_id=%s WHERE run_id=%s", (commit_id, run.run_id))
        await self.execution._audit(conn, run.principal, "summary.committed", cursor)
