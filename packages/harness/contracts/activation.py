"""发布激活请求与部署侧受信材料索引。"""

from typing import Annotated, Literal

from pydantic import Field

from harness.contracts.specs import Contract, Identifier

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ActivationRequest(Contract):
    release_id: Identifier
    manifest_hash: Digest
    report_hash: Digest
    expected_active_release_id: Identifier | None


class ActivationPolicy(Contract):
    version: Literal["activation-policy@1"] = "activation-policy@1"
    tenant_id: Identifier
    application_id: Identifier
    agent_id: Identifier
    environment_profile_hash: Digest
    freeze_hash: Digest
    dataset_hash: Digest
    gate_policy_version: Identifier
    report_hash: Digest
