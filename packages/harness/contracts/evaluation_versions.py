"""实际执行版本观察与报告身份；不把候选声明当取证。"""

from typing import Literal

from pydantic import AwareDatetime, Field

from harness.contracts.hashing import canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.specs import Contract


class ExecutionVersion(Contract):
    version: Literal["workspace-observation@1"] = "workspace-observation@1"
    observed_at: AwareDatetime
    observed_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_tree_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    dependency_lock_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool_schema_hashes_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    profiles_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    environment_profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    configuration_consistent: bool = Field(strict=True)

    def differences(self, candidate: ReleaseManifest) -> tuple[str, ...]:
        expected = {
            "observed_manifest_hash": candidate.content_hash(),
            "source_tree_hash": candidate.source_tree_hash,
            "dependency_lock_hash": candidate.dependency_lock_hash,
            "prompt_hash": candidate.prompt_hash,
            "tool_schema_hashes_hash": canonical_hash(candidate.tool_schema_hashes.as_object()),
            "model_profile_hash": canonical_hash(candidate.model_profile.as_object()),
            "profiles_hash": canonical_hash(candidate.profiles.as_object()),
            "environment_profile_hash": candidate.environment_profile_hash,
        }
        changed = [name for name, value in expected.items() if getattr(self, name) != value]
        if not self.configuration_consistent:
            changed.append("configuration_consistent")
        return tuple(changed)
