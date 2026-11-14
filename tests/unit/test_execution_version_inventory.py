"""工作区版本观察独立读文件的定义；不装配供应商、PG或图。"""

import hashlib
import sys
from types import SimpleNamespace

from harness.bootstrap.assembly import DevelopmentAssembly, file_hash, source_hash
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import AdmissionProfile
from harness.contracts.snapshots import JsonSnapshot
from harness.runtime.releases import InstalledReleases
from tests.unit.test_evaluation_collector import candidate


def dump(value):
    return SimpleNamespace(model_dump=lambda **kwargs: value)


def test_observer_rehashes_workspace_instead_of_returning_candidate(tmp_path):
    (tmp_path / "packages").mkdir()
    source = tmp_path / "packages/core.py"
    source.write_text("# 合成代码v1", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("name='controlled'", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("controlled-lock", encoding="utf-8")
    base = candidate()
    admission = AdmissionProfile()
    entries = {binding.name: SimpleNamespace(spec=JsonSnapshot.capture({"name": binding.name, "version": binding.version}))
               for binding in base.spec.allowed_tools}
    tool_hashes = {f"{binding.name}@{binding.version}": entries[binding.name].spec.content_hash
                   for binding in base.spec.allowed_tools}
    env = {"python": sys.version.split()[0], "installed_runtime_versions": {}, "admission": admission.model_dump(mode="json")}
    profiles = {"environment": env, "execution": {"version": "controlled"}, "extensions": {},
        "artifacts": {}, "context": {}, "summary_commit": {}, "artifact_retention": {}, "retention": {}}
    release = base.model_copy(update={"source_tree_hash": source_hash(tmp_path),
        "dependency_lock_hash": file_hash(tmp_path / "uv.lock"),
        "prompt_hash": hashlib.sha256(b"controlled").hexdigest(), "tool_schema_hashes": JsonSnapshot.capture(tool_hashes),
        "profiles": JsonSnapshot.capture(profiles), "environment_profile_hash": canonical_hash(env)})
    assembly = DevelopmentAssembly.__new__(DevelopmentAssembly)
    assembly.root = tmp_path
    assembly.releases = InstalledReleases((release,))
    assembly.assert_compatible = lambda current: None
    assembly.reads, assembly.writes = SimpleNamespace(entries=entries), None
    assembly._installed_versions = {}
    assembly.store = SimpleNamespace(profile=admission, retention_policy=dump({}))
    assembly.profile = dump({"version": "controlled"})
    assembly.profile.route = dump({})
    assembly.extensions = SimpleNamespace(snapshot=JsonSnapshot.capture({}))
    assembly.artifacts = SimpleNamespace(profile=dump({}))
    assembly.artifact_retention = SimpleNamespace(policy=dump({}))
    assembly.context_profile, assembly.summary_commit_profile = dump({}), dump({})
    assembly.model_runtime = SimpleNamespace(binding_snapshot=None)
    assembly.retrieval = None
    assembly.engine = SimpleNamespace(prompts={release.spec.release_id: "controlled"})
    before = assembly.observe_execution_version(release)
    assert before.differences(release) == ()
    source.write_text("# 合成代码v2", encoding="utf-8")
    after = assembly.observe_execution_version(release)
    assert after.source_tree_hash != before.source_tree_hash
    assert "source_tree_hash" in after.differences(release)
