"""在同一事务内比较并切换 active 指针、保存精确发布审计。"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

from harness.adapters.evaluation_files import bounded_read, read_freeze
from harness.adapters.postgres.evaluations import PostgresEvaluationContexts
from harness.adapters.runtime_postgres import PostgresRuntimeStore
from harness.contracts.activation import ActivationPolicy, ActivationRequest
from harness.contracts.evaluation_versions import ExecutionVersion
from harness.contracts.release import EvaluationContext, ReleaseManifest
from harness.contracts.runtime import Principal, RuntimeFault
from harness.evaluation.activation import verify_report
from harness.evaluation.assets import decode


def require_publisher(actor: Principal, agent_id: str) -> None:
    if "publisher" not in actor.roles or f"agent:{agent_id}:publish" not in actor.resource_scopes:
        raise RuntimeFault("FORBIDDEN", 403)


class PostgresReleaseActivation:
    def __init__(
        self, store: PostgresRuntimeStore, directory: Path,
        observe: Callable[[ReleaseManifest], ExecutionVersion],
        current_identity: Callable[[Principal], Principal],
    ) -> None:
        self.store = store
        self.directory = directory
        self.observe = observe
        self.current_identity = current_identity

    def _verify(
        self, actor: Principal, agent_id: str, request: ActivationRequest, candidate: ReleaseManifest,
    ) -> tuple[ActivationPolicy, EvaluationContext]:
        try:
            policy = ActivationPolicy.model_validate_json(bounded_read(self.directory / "policy.json"))
            if (policy.tenant_id, policy.application_id, policy.agent_id, policy.report_hash) != (
                actor.tenant_id, actor.application_id, agent_id, request.report_hash
            ):
                raise RuntimeFault("ACTIVATION_REPORT_NOT_APPROVED", 403)
            frozen, blobs = read_freeze(self.directory / "freeze")
            report = decode(bounded_read(self.directory / "report.json"))
            collection = decode(bounded_read(self.directory / "collection.json"))
            if not isinstance(collection, dict) or not isinstance(report, dict):
                raise ValueError("发布材料必须是 JSON 对象")
            context = verify_report(policy, candidate, frozen, blobs, collection, report)
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            raise RuntimeFault("ACTIVATION_EVIDENCE_UNAVAILABLE", 503) from None
        if self.observe(candidate).differences(candidate):
            raise RuntimeFault("ACTIVATION_INSTALLED_VERSION_MISMATCH")
        return policy, context

    async def activate(self, actor: Principal, agent_id: str, request: ActivationRequest) -> dict[str, str]:
        require_publisher(actor, agent_id)
        async with self.store._transaction() as conn:
            await self.store._admit(conn, actor)
            row = await self.store._release(conn, actor, request.release_id)
            candidate = ReleaseManifest.model_validate(row["manifest"])
            if (
                candidate.spec.agent_id != agent_id or row["agent_id"] != agent_id
                or candidate.spec.release_id != request.release_id
                or row["manifest_hash"] != request.manifest_hash
                or candidate.content_hash() != request.manifest_hash
            ):
                raise RuntimeFault("ACTIVATION_CANDIDATE_MISMATCH")

        # 回放和工作树取证均在数据库事务外完成。
        policy, context = await asyncio.to_thread(self._verify, actor, agent_id, request, candidate)
        scope = (actor.tenant_id, actor.application_id, agent_id)
        async with self.store._transaction() as conn:
            await self.store._lock_tenant(conn, actor)
            require_publisher(await asyncio.to_thread(self.current_identity, actor), agent_id)
            row = await self.store._one(
                conn,
                """SELECT manifest_hash, manifest, agent_id FROM harness.releases
                   WHERE tenant_id=%s AND application_id=%s AND release_id=%s FOR SHARE""",
                (*scope[:2], request.release_id),
            )
            if (row["manifest_hash"] != request.manifest_hash or row["agent_id"] != agent_id
                    or row["manifest"] != candidate.model_dump(mode="json")):
                raise RuntimeFault("ACTIVATION_CANDIDATE_MISMATCH")
            evidence = await self.store._one(
                conn,
                """SELECT *, expires_at > clock_timestamp() AS live
                   FROM harness.evaluation_contexts
                   WHERE tenant_id=%s AND application_id=%s AND release_id=%s AND context_id=%s FOR SHARE""",
                (*scope[:2], request.release_id, context.context_id),
            )
            if (evidence["revoked_at"] is not None or not evidence["live"]
                    or PostgresEvaluationContexts.binding(evidence) != context):
                raise RuntimeFault("ACTIVATION_EVALUATION_CONTEXT_INVALID")
            current = await (await conn.execute(
                """SELECT release_id FROM harness.active_releases
                   WHERE tenant_id=%s AND application_id=%s AND agent_id=%s FOR UPDATE""", scope,
            )).fetchone()
            previous = current["release_id"] if current else None
            if previous != request.expected_active_release_id:
                raise RuntimeFault("ACTIVATION_ACTIVE_RELEASE_CONFLICT")
            await conn.execute(
                """INSERT INTO harness.active_releases (tenant_id, application_id, agent_id, release_id)
                   VALUES (%s,%s,%s,%s) ON CONFLICT (tenant_id, application_id, agent_id)
                   DO UPDATE SET release_id=EXCLUDED.release_id""", (*scope, request.release_id),
            )
            await self.store._audit(conn, actor, "release.activated", json.dumps({
                **policy.model_dump(mode="json"), "context_id": context.context_id,
                "release_id": request.release_id, "manifest_hash": request.manifest_hash,
                "previous_release_id": previous,
            }, sort_keys=True))
        return {"release_id": request.release_id, "manifest_hash": request.manifest_hash,
                "report_hash": request.report_hash}
