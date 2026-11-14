"""唯一开发装配：配置、实际可执行清单与独立manifest，不读取其他项目文件。"""

from __future__ import annotations

import hashlib
import sys
import tomllib
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domain_tools.operations.completion import WorkOrderCompletionValidator
from domain_tools.operations.queries import registered_order_query
from domain_tools.operations.task_policy import require_task_target, task_orders
from domain_tools.operations.work_orders import (
    CreateWorkOrderInput,
    WorkOrderSandboxHandler,
    registered_work_order,
)
from harness.adapters.artifact_retention_postgres import PostgresArtifactRetention
from harness.adapters.artifacts_postgres import PostgresArtifacts
from harness.adapters.development_auth import LiveDevelopmentAuthenticator
from harness.adapters.evaluation_files import read_freeze
from harness.adapters.postgres.judge_authority import PostgresJudgeAuthority
from harness.adapters.context_postgres import PostgresContextSource
from harness.adapters.summaries_postgres import PostgresSummaryStore
from harness.adapters.recovery_postgres import PostgresRecoveryStore
from harness.adapters.retention_postgres import PostgresRetention
from harness.bootstrap.providers import load_model_runtime
from harness.bootstrap.server import load_server_settings
from harness.contracts.artifacts import ArtifactProfile, ArtifactRetentionPolicy
from harness.contracts.context import ContextProfile, MemoryBoundaryProfile, SummaryCommitProfile
from harness.context.boundary import require_memory_boundary
from harness.context.pipeline import ContextPipeline
from harness.contracts.hashing import canonical_hash
from harness.contracts.evaluation_versions import ExecutionVersion
from harness.contracts.evaluation_gates import JudgeOutput
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import AnswerCandidate, ClaimedRun, RuntimeFault
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import AgentSpec, KnowledgeBinding
from harness.contracts.state import ValidationReport
from harness.graphs.agent import AgentEngine
from harness.evaluation.assets import decode
from harness.runtime.extensions import ExtensionChain, RegisteredExtension
from harness.runtime.models import GovernedModelGateway
from harness.runtime.releases import InstalledReleases, build_development_release
from harness.tools.artifacts import registered_artifact_reader
from harness.tools.reads import ReadCoordinator, RegisteredRead
from harness.tools.writes import RegisteredWrite, WriteCoordinator, WriteRegistry

if TYPE_CHECKING:
    from domain_tools.knowledge.completion import KnowledgeCompletionValidator
    from domain_tools.knowledge.context import KnowledgeContext
    from harness.adapters.retrieval_postgres import PostgresRetrievalRepository
    from harness.bootstrap.retrieval import RetrievalRuntime


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
        knowledge = self.settings.agent_profile == "travel"
        self.judge_mode = self.settings.agent_profile == "judge"
        if knowledge and (not self.settings.rag_enabled or self.settings.enable_write_sandbox):
            raise ValueError("知识profile必须显式启用RAG且不启用运营写沙箱")
        if self.judge_mode and (self.settings.rag_enabled or self.settings.enable_write_sandbox):
            raise ValueError("Judge profile不直接启用RAG或运营写沙箱")
        self._agent_path = f"agents/{self.settings.agent_profile}.json"
        self._prompt_paths = (("prompts/judge/system.txt",) if self.judge_mode else
                              (f"prompts/{self.settings.agent_profile}/system.txt",
                               "prompts/knowledge/protocol.txt" if knowledge else "prompts/runtime/protocol.txt"))
        self._summary_prompt_path = "prompts/context/summary.txt"
        self._summary_prompt = (root / self._summary_prompt_path).read_text("utf-8")
        self._summary_file_hash = file_hash(root / self._summary_prompt_path)
        self.context_profile = ContextProfile(history_mode="current_only" if knowledge or self.judge_mode else "success_history",
            summary_prompt_hash=hashlib.sha256(self._summary_prompt.encode("utf-8")).hexdigest())
        self.store = PostgresRecoveryStore(
            self.settings.database_dsn.get_secret_value(), self.settings.admission
        )
        self.store.identity_source = LiveDevelopmentAuthenticator(
            self.settings.identity_file, self.settings.environment
        )
        self.judge_authority: PostgresJudgeAuthority | None = None
        self.judge_material_repository: PostgresRetrievalRepository | None = None
        if self.judge_mode:
            self.store.answer_authority = self.authorize_judge_answer
            self.store.resource_authority_ref = "judge-target@1"
            if self.settings.judge_freeze_directory is not None:
                from harness.adapters.retrieval_postgres import PostgresRetrievalRepository

                frozen, blobs = read_freeze(self.settings.judge_freeze_directory)
                source = PostgresRetrievalRepository(self.settings.database_dsn.get_secret_value(), self.store,
                                                    self.store.identity_source.current)
                self.judge_material_repository = source
                self.judge_authority = PostgresJudgeAuthority(
                    self.store, frozen, blobs, source.judge_reference_material, "knowledge-read@1",
                )
                self.store.new_run_authority = self.judge_authority.authorize_public_read
                self.store.resource_authority = self.judge_authority.authorize_public_read
        self.artifacts = PostgresArtifacts(self.store, ArtifactProfile())
        self.store.retention_policy = self.settings.retention
        # API只登记/查看；实际副本清理端口仅由显式运维命令安装。
        self.retention = PostgresRetention(self.store)
        self.artifact_retention = PostgresArtifactRetention(self.artifacts, ArtifactRetentionPolicy(), executor_installed=True)
        self.summary_store = PostgresSummaryStore(self.store)
        self.summary_commit_profile = SummaryCommitProfile()
        self.store.summary_authority = self.summary_store.commit_in_transaction
        self.store.summary_authority_ref = self.summary_commit_profile.version
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
        self.retrieval: RetrievalRuntime | None = None
        self.knowledge_validator: KnowledgeCompletionValidator | None = None
        self.knowledge_context: KnowledgeContext | None = None
        task_guard = require_task_target
        if knowledge:
            from domain_tools.knowledge.retrieval import registered_knowledge, require_query
            from harness.bootstrap.retrieval import load_retrieval

            self.retrieval = load_retrieval(root, enabled=True, execution=self.store,
                                            current_identity=self.store.identity_source.current)
            if self.retrieval is None:
                raise RuntimeFault("KNOWLEDGE_RUNTIME_REQUIRED", 503)
            read_entry = registered_knowledge(self.retrieval.service)
            task_guard = require_query
            installed["pymilvus"] = version("pymilvus")
            dependencies_match = dependencies_match and locked.get("pymilvus") == installed["pymilvus"]
        else:
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
            self.store, (read_entry, registered_artifact_reader(self.artifacts)),
            extensions=self.extensions, task_guard=task_guard
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
        self.store.delegated_read_hashes = frozenset(
            item.spec.content_hash for item in self.reads.entries.values() if item.accounting_ref is not None
        )
        if knowledge:
            from domain_tools.knowledge.completion import KnowledgeCompletionValidator
            from domain_tools.knowledge.context import KnowledgeContext

            if self.retrieval is None:
                raise RuntimeFault("KNOWLEDGE_RUNTIME_REQUIRED", 503)
            self.knowledge_validator = KnowledgeCompletionValidator(self.store, self.retrieval.service)
            self.knowledge_context = KnowledgeContext(self.reads, self.knowledge_validator, self.artifacts)
            self.store.answer_authority = self.authorize_answer
            self.store.resource_authority = self.retrieval.repository.authorize_public_read
            self.store.resource_authority_ref = "knowledge-read@1"
            self.store.judge_material_authority = self.retrieval.repository.judge_reference_material
            self.store.judge_material_authority_ref = "knowledge-read@1"
            self.store.retrieval_evidence_authority = self.retrieval.repository.evaluation_retrieval_evidence
        agent_source = (root / self._agent_path).read_bytes()
        self._agent_file_hash = hashlib.sha256(agent_source).hexdigest()
        original = AgentSpec.model_validate_json(agent_source)
        allowed = {item.spec.as_object()["name"]: item for item in entries}
        spec = original.model_copy(
            update={
                "release_id": ("judge-development@1" if self.judge_mode else
                               "knowledge-development@1" if knowledge else "operations-development@1"),
                "graph_version": "rag-chat-governed@1" if knowledge else "create-agent-governed@1",
                "allowed_tools": tuple(
                    item for item in original.allowed_tools if item.name in allowed
                ),
            }
        )
        if knowledge:
            if self.retrieval is None:
                raise RuntimeFault("KNOWLEDGE_RUNTIME_REQUIRED", 503)
            pins = {p.knowledge_base_id: p for p in self.retrieval.release.snapshots}
            if set(pins) != {b.knowledge_base_id for b in spec.knowledge_bindings}:
                raise ValueError("Agent知识库与部署pin集合必须完全一致")
            spec = AgentSpec.model_validate({**spec.model_dump(mode="python"), "knowledge_bindings": tuple(
                KnowledgeBinding(knowledge_base_id=b.knowledge_base_id, snapshot_id=pins[b.knowledge_base_id].snapshot_id,
                                  retrieval_profile_ref=b.retrieval_profile_ref) for b in spec.knowledge_bindings)})
            if any(b.retrieval_profile_ref not in {p.ref for p in self.retrieval.release.profiles} for b in spec.knowledge_bindings):
                raise ValueError("Agent引用的检索profile未安装")
        prompt = "\n".join((root / name).read_text("utf-8") for name in self._prompt_paths)
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
        if knowledge:
            env["read_source"] = "authorized-knowledge@1"
            env["agent_profile"] = "travel"
        if self.judge_mode:
            env["read_source"] = "judge-authorized-target@1"
            env["agent_profile"] = "judge"
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
        profiles = {"execution": self.profile.model_dump(mode="json"), "environment": env,
                    "retention": self.settings.retention.model_dump(mode="json")}
        profiles["artifacts"] = self.artifacts.profile.model_dump(mode="json")
        profiles["artifact_retention"] = self.artifact_retention.policy.model_dump(mode="json")
        profiles["context"] = self.context_profile.model_dump(mode="json")
        profiles["context_boundary"] = MemoryBoundaryProfile().model_dump(mode="json")
        profiles["summary_commit"] = self.summary_commit_profile.model_dump(mode="json")
        profiles["extensions"] = self.extensions.snapshot.as_object()
        if self.judge_mode:
            profiles["judge"] = {"implementation": "governed-harness-judge@1"}
            profiles["resource_authority_ref"] = "judge-target@1"
        if self.model_runtime.binding_snapshot is not None:
            profiles["model_binding"] = self.model_runtime.binding_snapshot.as_object()
        if self.retrieval is not None:
            profiles["retrieval"] = self.retrieval.release.model_dump(mode="json")
            profiles["resource_authority_ref"] = "knowledge-read@1"
            profiles["read_accounting"] = {digest: "child-model-ledger@1" for digest in self.store.delegated_read_hashes}
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
            release_prefix=("judge-development" if self.judge_mode else
                            "knowledge-development" if knowledge else "operations-development"),
        )
        spec = manifest.spec
        require_memory_boundary(manifest)
        self.releases = InstalledReleases((manifest,))
        if expected_release is not None:
            self.releases.require(expected_release)
        self.store.installed_manifest_hashes = self.releases.hashes
        self._prompt_files = {
            name: file_hash(root / name)
            for name in self._prompt_paths
        }
        self.manifest = manifest
        gateway = GovernedModelGateway(
            self.store, self.model_runtime.transport, self.profile, self.extensions,
            release_guard=self.assert_compatible, context_authority=self.authorize_model_context)
        self.context_pipeline = ContextPipeline(PostgresContextSource(self.store), self.artifacts,
            gateway.complete_summary, self.profile, self.context_profile, self._summary_prompt, self.summary_store)
        gateway.context_builder = self.context_pipeline
        self.engine = AgentEngine(
            self.store,
            gateway,
            self.reads,
            self.writes,
            checkpointer,
            owner,
            {spec.release_id: manifest},
            {spec.release_id: prompt},
            self.validate,
            task_guard,
            self.extensions,
            release_guard=self.assert_compatible,
            prepare_context=self.knowledge_context.prepare if self.knowledge_context else None,
            authorize_context=self.authorize_model_context,
            artifact_projector=self.artifacts,
        )

    async def authorize_model_context(self, run: ClaimedRun, owner: str) -> None:
        await self.artifacts.authorize_context(run, owner)
        await self.context_pipeline.authorize(run, owner)
        if self.knowledge_context is not None:
            await self.knowledge_context.authorize(run, owner)
        if self.judge_mode:
            if self.judge_authority is None:
                raise RuntimeFault("JUDGE_FREEZE_NOT_CONFIGURED", 503)
            await self.judge_authority.authorize(run)

    def assert_compatible(self, release: ReleaseManifest) -> None:
        self.releases.require(release)
        try:
            if (
                source_hash(self.root) != self.manifest.source_tree_hash
                or file_hash(self.root / self._summary_prompt_path) != self._summary_file_hash
                or file_hash(self.root / "uv.lock") != self.manifest.dependency_lock_hash
                or file_hash(self.root / self._agent_path) != self._agent_file_hash
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
            if self.retrieval is not None:
                self.retrieval.configuration_guard()
        except (OSError, ValueError, PackageNotFoundError):
            raise RuntimeFault("INSTALLED_RELEASE_DRIFT", 503) from None
        if not self._dependencies_match or not self.model_runtime.ready():
            raise RuntimeFault("RELEASE_MODEL_CONFIGURATION_UNAVAILABLE", 503)

    @property
    def model_ready(self) -> bool:
        try:
            self.assert_compatible(self.manifest)
            if self.judge_mode and (self.judge_authority is None or not self.judge_authority.ready_for(self.manifest)):
                return False
            return True
        except RuntimeFault:
            return False

    def observe_execution_version(self, release: ReleaseManifest) -> ExecutionVersion:
        """由Worker在线程中调用；重新读取本项目文件/安装信息，不访问模型或PG。"""
        installed = self.releases.require(release)
        consistent = True
        try:
            self.assert_compatible(installed)
        except RuntimeFault:
            consistent = False
        tool_hashes = {}
        for binding in installed.spec.allowed_tools:
            entry = self.reads.entries.get(binding.name)
            if entry is None:
                if self.writes is None:
                    raise RuntimeFault("TOOL_NOT_REGISTERED", 503)
                entry_spec = self.writes.registry.get(binding.name).spec
            else:
                entry_spec = entry.spec
            tool_hashes[f"{binding.name}@{binding.version}"] = entry_spec.content_hash
        profiles = installed.profiles.as_object()
        raw_environment = profiles.get("environment")
        if not isinstance(raw_environment, dict):
            raise RuntimeFault("EXECUTION_ENVIRONMENT_EVIDENCE_MISSING", 503)
        environment = dict(raw_environment)
        versions = {}
        for name in self._installed_versions:
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = "not-installed"
        environment["python"] = sys.version.split()[0]
        environment["installed_runtime_versions"] = versions
        environment["admission"] = self.store.profile.model_dump(mode="json")
        profiles["environment"] = environment
        profiles["execution"] = self.profile.model_dump(mode="json")
        profiles["extensions"] = self.extensions.snapshot.as_object()
        profiles["artifacts"] = self.artifacts.profile.model_dump(mode="json")
        profiles["context"] = self.context_profile.model_dump(mode="json")
        profiles["summary_commit"] = self.summary_commit_profile.model_dump(mode="json")
        profiles["artifact_retention"] = self.artifact_retention.policy.model_dump(mode="json")
        profiles["retention"] = self.store.retention_policy.model_dump(mode="json")
        if self.model_runtime.binding_snapshot is not None:
            profiles["model_binding"] = self.model_runtime.binding_snapshot.as_object()
        if self.retrieval is not None:
            profiles["retrieval"] = self.retrieval.release.model_dump(mode="json")
        prompt = self.engine.prompts[installed.spec.release_id]
        observed = ReleaseManifest.model_validate({**installed.model_dump(mode="json"),
            "source_tree_hash": source_hash(self.root), "dependency_lock_hash": file_hash(self.root / "uv.lock"),
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tool_schema_hashes": tool_hashes, "model_profile": self.profile.route.model_dump(mode="json"),
            "profiles": profiles, "environment_profile_hash": canonical_hash(environment)})
        return ExecutionVersion(observed_at=datetime.now(UTC), observed_manifest_hash=observed.content_hash(),
            source_tree_hash=observed.source_tree_hash, dependency_lock_hash=observed.dependency_lock_hash,
            prompt_hash=observed.prompt_hash, tool_schema_hashes_hash=canonical_hash(tool_hashes),
            model_profile_hash=canonical_hash(observed.model_profile.as_object()), profiles_hash=canonical_hash(profiles),
            environment_profile_hash=observed.environment_profile_hash, configuration_consistent=consistent)

    async def validate(self, run: ClaimedRun, candidate: AnswerCandidate) -> ValidationReport:
        if self.judge_mode:
            if self.judge_authority is None:
                raise RuntimeFault("JUDGE_FREEZE_NOT_CONFIGURED", 503)
            await self.judge_authority.authorize(run)
            request = self.judge_authority.input_for(run)
            valid = False
            try:
                output = JudgeOutput.model_validate(decode(candidate.answer.encode("utf-8")))
                valid = (output.case_id == request.case_id and output.target_run_id == request.target_run_id
                         and output.target_result_hash == request.target_result_hash
                         and output.freeze_hash == request.freeze_hash
                         and set(output.scores) == set(request.dimensions)
                         and candidate.answer_kind == "answer" and candidate.task_outcome == "achieved"
                         and candidate.task.source_turn_id == run.run_id and not candidate.task.evidence_refs)
            except (TypeError, ValueError):
                pass
            return ValidationReport(validator_ref=run.manifest.spec.completion_validator_ref,
                input_hash=canonical_hash(candidate.validation_input()), verdict="pass" if valid else "fail",
                checks={"judge_output_bound": valid}, checked_at=datetime.now(UTC))
        if self.knowledge_validator is not None:
            return await self.knowledge_validator.validate(run, self.engine.owner, candidate)
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

    async def authorize_judge_answer(self, conn: Any, owner: str, run: ClaimedRun,
                                     candidate: AnswerCandidate) -> None:
        if self.judge_authority is None:
            raise RuntimeFault("JUDGE_FREEZE_NOT_CONFIGURED", 503)
        request = self.judge_authority.input_for(run)
        await self.judge_authority.authorize_in_transaction(conn, run, request)

    async def authorize_answer(self, conn: Any, owner: str, run: ClaimedRun, candidate: AnswerCandidate) -> None:
        """答案公开与业务commit事务内复查；只执行本地PG查询和纯完成校验。"""
        if self.retrieval is None:
            raise RuntimeFault("KNOWLEDGE_AUTHORITY_NOT_INSTALLED", 503)
        from domain_tools.knowledge.completion import knowledge_receipts, validate_knowledge
        from domain_tools.knowledge.retrieval import receipt_output
        from harness.adapters.postgres.queries import PostgresEvidenceQueries

        receipts = knowledge_receipts(await PostgresEvidenceQueries.read_receipts(conn, run.run_id))
        outputs = tuple(receipt_output(run, receipt) for receipt in receipts)
        for output in outputs:
            for result in output.results:
                await self.retrieval.repository.authorize_result_in_transaction(conn, run.principal, owner, run.run_id, result)
        report = validate_knowledge(run, candidate, outputs, receipts)
        if (report.verdict not in {"pass", "needs_clarification"} or not all(report.checks.values())
            or report.input_hash != candidate.validation.input_hash or report.checks != candidate.validation.checks
            or report.evidence_refs != candidate.validation.evidence_refs):
            raise RuntimeFault("KNOWLEDGE_COMPLETION_AUTHORITY_CHANGED", 403)

    async def require_optional_schema(self) -> None:
        if self.retrieval is not None:
            await self.retrieval.repository.require_schema()
        if self.judge_material_repository is not None:
            await self.judge_material_repository.require_schema()
