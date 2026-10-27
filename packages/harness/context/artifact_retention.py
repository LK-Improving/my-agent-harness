"""产物保留纯判定：TTL和显式撤权不会释放未决证据。"""

from datetime import datetime, timedelta

from harness.contracts.artifacts import ArtifactRetentionFacts, ArtifactRetentionPolicy, ArtifactRetentionView


def retention_view(*, artifact_ref: str, facts: ArtifactRetentionFacts,
                   policy: ArtifactRetentionPolicy, checked_at: datetime,
                   expires_at: datetime, revoked_at: datetime | None,
                   deletion_requested: bool, read_denial_code: str | None,
                   executor_installed: bool = False, purged_at: datetime | None = None) -> ArtifactRetentionView:
    reasons = tuple(name for name, present in facts.model_dump().items() if present)
    return ArtifactRetentionView(
        artifact_ref=artifact_ref, policy_version=policy.version,
        state="purged" if purged_at is not None else ("retained" if reasons else "cleanup_pending"),
        purged_at=purged_at, deletion_requested=deletion_requested,
        revoked_at=revoked_at, expires_at=expires_at, readable=read_denial_code is None,
        read_denial_code=read_denial_code,
        retention_reasons=() if purged_at is not None else (*reasons, *(() if executor_installed else ("cleanup_executor_not_installed",))),
        checked_at=checked_at,
        next_check_at=checked_at + timedelta(seconds=policy.recheck_seconds),
    )
