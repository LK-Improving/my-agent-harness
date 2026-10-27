"""纯发布构建与已安装清单；快照不能代替实际实现注册。"""

import hashlib
from dataclasses import dataclass

from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec


def build_development_release(
    *,
    spec: AgentSpec,
    source_hash: str,
    lock_hash: str,
    prompt: str,
    tool_hashes: dict[str, str],
    model_profile: JsonSnapshot,
    profiles: JsonSnapshot,
    environment_hash: str,
    release_prefix: str = "operations-development",
) -> ReleaseManifest:
    # 完整候选内容参与身份；不把调用方给的旧 release_id 当作新构建身份。
    candidate = ReleaseManifest(
        spec=spec.model_copy(update={"release_id": "development-candidate"}),
        source_revision=None,
        source_tree_hash=source_hash,
        dependency_lock_hash=lock_hash,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        tool_schema_hashes=JsonSnapshot.capture(tool_hashes),
        model_profile=model_profile,
        profiles=profiles,
        environment_profile_hash=environment_hash,
    )
    release_id = release_prefix + "-" + candidate.content_hash()[:24]
    return ReleaseManifest.model_validate(
        {
            **candidate.model_dump(mode="python"),
            "spec": candidate.spec.model_copy(update={"release_id": release_id}),
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class InstalledReleases:
    manifests: tuple[ReleaseManifest, ...]

    def __init__(self, manifests: tuple[ReleaseManifest, ...]) -> None:
        copies = tuple(
            ReleaseManifest.model_validate_json(item.model_dump_json()) for item in manifests
        )
        if len({item.spec.release_id for item in copies}) != len(copies):
            raise ValueError("已安装发布ID不能重复")
        object.__setattr__(self, "manifests", copies)

    @property
    def hashes(self) -> frozenset[str]:
        return frozenset(item.content_hash() for item in self.manifests)

    def require(self, release: ReleaseManifest) -> ReleaseManifest:
        installed = next(
            (item for item in self.manifests if item.spec.release_id == release.spec.release_id),
            None,
        )
        if installed is None or installed.content_hash() != release.content_hash():
            raise RuntimeFault("EXECUTABLE_RELEASE_NOT_INSTALLED", 503)
        return installed
