"""唯一开发装配：配置、实际可执行清单与独立manifest，不读取其他项目文件。"""

import hashlib
import sys
import tomllib
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from domain_tools.operations.completion import WorkOrderCompletionValidator
from domain_tools.operations.queries import registered_order_query
from domain_tools.operations.task_policy import require_task_target, task_orders
from domain_tools.operations.work_orders import (
    CreateWorkOrderInput,
    WorkOrderSandboxHandler,
    registered_work_order,
)
from harness.adapters.development_auth import LiveDevelopmentAuthenticator
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.bootstrap.providers import load_model_runtime
from harness.bootstrap.server import load_server_settings
from harness.contracts.hashing import canonical_hash
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec
from harness.contracts.state import ValidationReport
from harness.graphs.agent import AgentEngine
from harness.runtime.extensions import ExtensionChain, RegisteredExtension
from harness.runtime.models import GovernedModelGateway
from harness.runtime.releases import InstalledReleases, build_development_release
from harness.tools.reads import ReadCoordinator, RegisteredRead
from harness.tools.writes import RegisteredWrite, WriteCoordinator, WriteRegistry


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(root: Path) -> str:
    files = {
        path.relative_to(root).as_posix(): file_hash(path)
        for folder in ("packages", "domain_tools", "apps", "migrations")
        for path in (root / folder).rglob("*")
        if path.is_file() and path.suffix in {".py", ".sql"}
    }
    files["pyproject.toml"] = file_hash(root / "pyproject.toml")
    return canonical_hash(files)


class DevelopmentAssembly:
    def __init__(
        self,
        root: Path,
        *,
        owner: str,
        checkpointer: Any,
        extensions: tuple[RegisteredExtension, ...] = (),
        expected_release: ReleaseManifest | None = None,
    ) -> None:
        self.root = root
        self.settings = load_server_settings(root)
        self.store = PostgresRecoveryStore(
            self.settings.database_dsn.get_secret_value(), self.settings.admission
        )
        self.store.identity_source = LiveDevelopmentAuthenticator(
            self.settings.identity_file, self.settings.environment
        )
        self.extensions = ExtensionChain(
            extensions, record_decision=partial(self.store.record_extension, owner)
        )
        locked = {
            item["name"]: item.get("version")
            for item in tomllib.loads((root / "uv.lock").read_text("utf-8"))["package"]
        }
        installed = {}
        for name in (
            "langchain",
            "langchain-core",
            "langgraph",
            "langgraph-checkpoint",
            "langgraph-prebuilt",
            "langgraph-checkpoint-postgres",
            "langsmith",
            "psycopg",
            "psycopg-pool",
            "fastapi",
            "pydantic",
            "httpx",
            "httpcore",
            "h11",
            "anyio",
            "psutil",
            "uvicorn",
        ):
            try:
                installed[name] = version(name)
            except PackageNotFoundError:
                installed[name] = "not-installed"
        dependencies_match = all(locked.get(name) == actual for name, actual in installed.items())
        read_entry = registered_order_query()
        write_entries: tuple[RegisteredWrite, ...] = ()
        if self.settings.enable_write_sandbox:
            from harness.bootstrap.sandbox import load_sandbox_settings

            sandbox = load_sandbox_settings(root)
            write_entries = (
                registered_work_order(WorkOrderSandboxHandler(sandbox.base_url, sandbox.token)),
            )
            from domain_tools.operations.queries import registered_sandbox_query

            read_entry = registered_sandbox_query(write_entries[0].handler)
        self.reads = ReadCoordinator(
            self.store, (read_entry,), extensions=self.extensions, task_guard=require_task_target
        )
        self.writes = (
            WriteCoordinator(
                self.store,
                WriteRegistry(write_entries),
                extensions=self.extensions,
                task_guard=require_task_target,
            )
            if write_entries
            else None
        )
        entries: list[RegisteredRead | RegisteredWrite] = [
            *self.reads.entries.values(),
            *write_entries,
        ]
        agent_source = (root / "agents/operations.json").read_bytes()
        self._agent_file_hash = hashlib.sha256(agent_source).hexdigest()
        original = AgentSpec.model_validate_json(agent_source)
        allowed = {item.spec.as_object()["name"]: item for item in entries}
        spec = original.model_copy(
            update={
                "release_id": "operations-development@1",
                "graph_version": "create-agent-governed@1",
                "allowed_tools": tuple(
                    item for item in original.allowed_tools if item.name in allowed
                ),
            }
        )
        prompt = (
            (root / "prompts/operations/system.txt").read_text("utf-8")
            + "\n"
            + (root / "prompts/runtime/protocol.txt").read_text("utf-8")
        )
        env = {
            "environment": self.settings.environment,
            "auth_mode": self.settings.auth_mode,
            "identity_source_ref": canonical_hash(
                {"mode": "development-live@1", "path": str(self.settings.identity_file)}
            ),
            "python": sys.version.split()[0],
            "installed_runtime_versions": installed,
            "execution_worker_replicas": 1,
            "max_active_runs_per_worker": 1,
            "sandbox_enabled": self.settings.enable_write_sandbox,
            "read_source": "synthetic-fixture",
            "admission": self.settings.admission.model_dump(mode="json"),
        }
        if self.settings.enable_write_sandbox:
            env["sandbox_base_url"] = sandbox.base_url
            env["read_source"] = "write-sandbox"
        self.model_runtime = load_model_runtime(
            root,
            model_profile_ref=spec.model_profile_ref,
            environment_profile_hash=canonical_hash(env),
            now=datetime.now(UTC),
        )
        self.profile = self.model_runtime.profile
        if (
            self.profile.budget.execution_budget_ms
            != self.settings.admission.execution_timeout_seconds * 1000
        ):
            raise ValueError("执行预算与准入profile时限必须一致")
        self._dependencies_match = dependencies_match
        self._installed_versions = installed
        self.store.new_run_ready = lambda: self.model_ready
        profiles = {"execution": self.profile.model_dump(mode="json"), "environment": env}
        profiles["extensions"] = self.extensions.snapshot.as_object()
        if self.model_runtime.binding_snapshot is not None:
            profiles["model_binding"] = self.model_runtime.binding_snapshot.as_object()
        source = source_hash(root)
        manifest = build_development_release(
            spec=spec,
            source_hash=source,
            lock_hash=file_hash(root / "uv.lock"),
            prompt=prompt,
            tool_hashes={
                f"{item.name}@{item.version}": allowed[item.name].spec.content_hash
                for item in spec.allowed_tools
            },
            model_profile=JsonSnapshot.capture(self.profile.route.model_dump(mode="json")),
            profiles=JsonSnapshot.capture(profiles),
            environment_hash=canonical_hash(env),
        )
        spec = manifest.spec
        self.releases = InstalledReleases((manifest,))
        if expected_release is not None:
            self.releases.require(expected_release)
        self.store.installed_manifest_hashes = self.releases.hashes
        self._prompt_files = {
            name: file_hash(root / name)
            for name in ("prompts/operations/system.txt", "prompts/runtime/protocol.txt")
        }
        self.manifest = manifest
        self.engine = AgentEngine(
            self.store,
            GovernedModelGateway(
                self.store,
                self.model_runtime.transport,
                self.profile,
                self.extensions,
                release_guard=self.assert_compatible,
            ),
            self.reads,
            self.writes,
            checkpointer,
            owner,
            {spec.release_id: manifest},
            {spec.release_id: prompt},
            self.validate,
            require_task_target,
            self.extensions,
            release_guard=self.assert_compatible,
        )

    def assert_compatible(self, release: ReleaseManifest) -> None:
        self.releases.require(release)
        try:
            if (
                source_hash(self.root) != self.manifest.source_tree_hash
                or file_hash(self.root / "uv.lock") != self.manifest.dependency_lock_hash
                or file_hash(self.root / "agents/operations.json") != self._agent_file_hash
                or any(
                    file_hash(self.root / name) != digest
                    for name, digest in self._prompt_files.items()
                )
                or load_server_settings(self.root) != self.settings
                or hashlib.sha256(
                    "\n".join(
                        (self.root / name).read_text("utf-8") for name in self._prompt_files
                    ).encode("utf-8")
                ).hexdigest()
                != self.manifest.prompt_hash
                or any(version(name) != actual for name, actual in self._installed_versions.items())
            ):
                raise RuntimeFault("INSTALLED_RELEASE_DRIFT", 503)
        except (OSError, ValueError, PackageNotFoundError):
            raise RuntimeFault("INSTALLED_RELEASE_DRIFT", 503) from None
        if not self._dependencies_match or not self.model_runtime.ready():
            raise RuntimeFault("RELEASE_MODEL_CONFIGURATION_UNAVAILABLE", 503)

    @property
    def model_ready(self) -> bool:
        try:
            self.assert_compatible(self.manifest)
            return True
        except RuntimeFault:
            return False

    async def validate(self, run: ClaimedRun, candidate: AnswerCandidate) -> ValidationReport:
        receipts = await self.store.write_receipts(run.principal, run.run_id)
        if candidate.task_outcome == "achieved" and receipts:
            # 参数来自授权持久回执关联的operation，不接受模型自报订单。
            targets = task_orders(run.request.message)
            if len(targets) != 1:
                return ValidationReport(
                    validator_ref=run.manifest.spec.completion_validator_ref,
                    input_hash=canonical_hash(candidate.validation_input()),
                    verdict="fail",
                    checks={"unambiguous_user_target": False},
                    checked_at=datetime.now(UTC),
                )
            expected = CreateWorkOrderInput(order_id=next(iter(targets)))
            return await WorkOrderCompletionValidator(self.store).validate(
                run.principal,
                run.run_id,
                expected,
                candidate,
                run.manifest.spec.completion_validator_ref,
            )
        reads = await self.store.read_receipts(run.principal, run.run_id)
        checks = {"task_source": candidate.task.source_turn_id == run.run_id}
        if candidate.task_outcome == "achieved":
            targets = task_orders(run.request.message)
            backed = {
                str(item.data["order_id"])
                for item in reads
                if isinstance(item.data, dict)
                and item.data.get("order_id")
                and item.args_hash == canonical_hash({"order_id": item.data["order_id"]})
            }
            checks["user_targets_covered"] = (
                bool(targets)
                and targets <= backed
                and all(target in candidate.answer for target in targets)
            )
            checks["query_evidence"] = candidate.answer_kind == "answer" and any(
                isinstance(item.data, dict)
                and bool(item.data.get("order_id"))
                and item.args_hash == canonical_hash({"order_id": item.data["order_id"]})
                and str(item.data["order_id"]) in candidate.answer
                for item in reads
            )
            checks["fixture_disclosed"] = any(
                label in candidate.answer for label in ("合成", "示例", "沙箱")
            )
        else:
            checks["no_unreported_write"] = not receipts
            checks["outcome_consistent"] = (candidate.answer_kind, candidate.task_outcome) in {
                ("clarification", "needs_input"),
                ("abstention", "not_achieved"),
            }
        passed = all(checks.values())
        return ValidationReport(
            validator_ref=run.manifest.spec.completion_validator_ref,
            input_hash=canonical_hash(candidate.validation_input()),
            verdict=("needs_clarification" if candidate.answer_kind == "clarification" else "pass")
            if passed
            else "fail",
            checks=checks,
            evidence_refs=tuple(item.operation_id for item in reads),
            checked_at=datetime.now(UTC),
        )
