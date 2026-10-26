"""发布固定身份与缺实现阻断定义；只用本地合成配置，不调用模型或数据库。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from harness.bootstrap.assembly import DevelopmentAssembly, file_hash, source_hash
from harness.bootstrap.releases import export_release, read_release
from harness.contracts.runtime import RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec
from harness.runtime.releases import InstalledReleases, build_development_release

ROOT = Path(__file__).resolve().parents[2]


def release(spec=None, **changes):
    spec = spec or AgentSpec.model_validate_json(
        (ROOT / "agents/operations.json").read_text("utf-8")
    )
    values = {
        "spec": spec,
        "source_hash": "1" * 64,
        "lock_hash": "2" * 64,
        "prompt": "受控提示词",
        "tool_hashes": {f"{item.name}@{item.version}": "3" * 64 for item in spec.allowed_tools},
        "model_profile": JsonSnapshot.capture({}),
        "profiles": JsonSnapshot.capture({}),
        "environment_hash": "4" * 64,
    }
    values.update(changes)
    return build_development_release(**values)


@pytest.mark.parametrize(
    "field",
    ["policy_profile_ref", "model_profile_ref", "completion_validator_ref", "memory_profile_ref"],
)
def test_full_agent_spec_changes_release_identity(field):
    original = release()
    changed = release(original.spec.model_copy(update={field: "changed@2"}))
    assert changed.spec.release_id != original.spec.release_id
    assert release(original.spec).content_hash() == original.content_hash()
    with pytest.raises(RuntimeFault, match="EXECUTABLE_RELEASE_NOT_INSTALLED"):
        InstalledReleases((changed,)).require(original)


def test_tool_limits_and_nonsecret_binding_change_release_identity():
    original = release()
    binding = original.spec.allowed_tools[0]
    changed = release(
        original.spec.model_copy(
            update={
                "allowed_tools": (
                    binding.model_copy(
                        update={"max_operations_per_run": binding.max_operations_per_run + 1}
                    ),
                    *original.spec.allowed_tools[1:],
                )
            }
        )
    )
    account = release(
        profiles=JsonSnapshot.capture({"model_binding": {"credential_binding": "account-2@1"}})
    )
    assert len({original.spec.release_id, changed.spec.release_id, account.spec.release_id}) == 3
    same_id = changed.model_copy(
        update={"spec": changed.spec.model_copy(update={"release_id": original.spec.release_id})}
    )
    with pytest.raises(RuntimeFault):
        InstalledReleases((original,)).require(same_id)


def test_export_is_immutable_and_snapshot_does_not_install_implementation(tmp_path):
    original = release()
    path = export_release(tmp_path, original)
    assert export_release(tmp_path, original) == path
    loaded = read_release(tmp_path, path.relative_to(tmp_path).as_posix())
    assert loaded == original
    with pytest.raises(RuntimeFault, match="EXECUTABLE_RELEASE_NOT_INSTALLED"):
        InstalledReleases(()).require(loaded)
    path.write_text("{}", "utf-8")
    with pytest.raises(RuntimeFault, match="RELEASE_SNAPSHOT_INVALID"):
        export_release(tmp_path, original)
    with pytest.raises(RuntimeFault, match="RELEASE_SNAPSHOT_INVALID"):
        read_release(tmp_path, "../outside.json")


@pytest.mark.parametrize("changed", ["source", "lock", "agent", "prompt", "admission", "model"])
def test_installed_workspace_drift_blocks_execution(tmp_path, monkeypatch, changed):
    (tmp_path / "packages").mkdir()
    (tmp_path / "agents").mkdir()
    for name, value in {
        "pyproject.toml": "name='controlled'",
        "uv.lock": "controlled-lock",
        "agents/operations.json": "{}",
        "prompt.txt": "原提示词",
        "packages/core.py": "# 受控源码",
    }.items():
        (tmp_path / name).write_text(value, "utf-8")
    manifest = release(
        prompt="原提示词",
        source_hash=source_hash(tmp_path),
        lock_hash=file_hash(tmp_path / "uv.lock"),
    )
    # 只构造守卫所需本地状态，不启动完整组合根或依赖服务。
    assembly = DevelopmentAssembly.__new__(DevelopmentAssembly)
    assembly.root, assembly.manifest = tmp_path, manifest
    assembly.releases = InstalledReleases((manifest,))
    assembly._agent_file_hash = file_hash(tmp_path / "agents/operations.json")
    assembly._prompt_files = {"prompt.txt": file_hash(tmp_path / "prompt.txt")}
    assembly._installed_versions = {}
    assembly.settings = "fixed-settings"
    assembly._dependencies_match = True
    assembly.model_runtime = SimpleNamespace(ready=lambda: changed != "model")
    monkeypatch.setattr(
        "harness.bootstrap.assembly.load_server_settings",
        lambda root: "changed" if changed == "admission" else "fixed-settings",
    )
    files = {
        "source": "packages/core.py",
        "lock": "uv.lock",
        "agent": "agents/operations.json",
        "prompt": "prompt.txt",
    }
    if changed in files:
        (tmp_path / files[changed]).write_text("changed", "utf-8")
    with pytest.raises(
        RuntimeFault, match="INSTALLED_RELEASE_DRIFT|RELEASE_MODEL_CONFIGURATION_UNAVAILABLE"
    ):
        assembly.assert_compatible(manifest)
