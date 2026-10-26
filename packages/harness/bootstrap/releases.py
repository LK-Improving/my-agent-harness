"""本地候选快照读写；读入快照仅用于比对，绝不据此安装旧代码。"""

from pathlib import Path

from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import RuntimeFault


def read_release(root: Path, ref: str) -> ReleaseManifest:
    try:
        path = (root / ref).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("快照路径必须位于本工作区")
        with path.open("rb") as stream:
            raw = stream.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            raise ValueError("快照过大")
        return ReleaseManifest.model_validate_json(raw)
    except (OSError, ValueError):
        raise RuntimeFault("RELEASE_SNAPSHOT_INVALID", 422) from None


def export_release(root: Path, manifest: ReleaseManifest) -> Path:
    folder = root / ".runtime" / "releases"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (manifest.content_hash() + ".json")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(manifest.model_dump_json(indent=2))
    except FileExistsError:
        existing = read_release(root, path.relative_to(root).as_posix())
        if existing != manifest:
            raise RuntimeFault("RELEASE_SNAPSHOT_CONFLICT") from None
    return path
