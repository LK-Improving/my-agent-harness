"""显式新适配器隔离探针；实际成功前不导出可信能力证据或启用新模式。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from harness.adapters.models.openai_chat_completions import (
    OpenAIChatCompletionsClient,
    installed_chat_protocol,
)
from harness.bootstrap.assembly import DevelopmentAssembly
from harness.bootstrap.event_loop import run
from harness.bootstrap.settings import load_qwen_settings
from harness.contracts.execution import ExecutionProfile
from harness.contracts.hashing import RunRequest, canonical_hash
from harness.contracts.providers import (
    ModelCapabilityEvidence,
    ModelProfile,
    ModelProviderConfiguration,
    ProviderConnection,
)
from harness.contracts.runtime import EvaluationRequest, Principal, SessionRequest
from harness.contracts.snapshots import JsonSnapshot
from harness.contracts.specs import ToolSpec
from harness.providers.catalog import CHAT_PROTOCOL, builtin_registry
from harness.providers.resolver import resolve_model_profile
from harness.runtime.models import GovernedModelGateway
from harness.runtime.releases import build_development_release


async def probe(root: Path) -> None:
    folder = root / ".runtime"
    base = DevelopmentAssembly(root, owner="capability-probe", checkpointer=None)
    if base.model_runtime.binding_snapshot is not None:
        raise ValueError("探针须从尚未启用新providers的独立验收环境执行")
    registry = builtin_registry()
    implementation = installed_chat_protocol()
    settings = load_qwen_settings(root / ".env", include_retrieval=False)
    connection = ProviderConnection(
        ref="dashscope-local-live@1",
        provider_id="dashscope",
        protocol_ref=CHAT_PROTOCOL,
        base_url=settings.base_url,
        credential_profile_ref=settings.credential_profile_ref,
        timeout_seconds=settings.timeout_seconds,
    )
    profile = ModelProfile(
        ref=base.manifest.spec.model_profile_ref,
        connection_ref=connection.ref,
        model_id=settings.chat_model,
        required_capabilities=("tool_calling",),
        capability_evidence_ref="dashscope-local-live-proof@1",
        max_output_tokens=base.profile.budget.max_output_tokens,
    )
    configuration = ModelProviderConfiguration(connections=(connection,), profiles=(profile,))
    now = datetime.now(UTC)
    # 仅隔离测试夹具：预先声明待验证条件以构造具体客户端，不写可信evidence文件。
    # report_hash明确标记尚未验证；只有下方真实结果断言通过后才晋升为可装配证据。
    candidate = ModelCapabilityEvidence(
        ref=profile.capability_evidence_ref,
        provider_id="dashscope",
        model_id=profile.model_id,
        task_type="chat",
        definition_hash=registry.get("dashscope").content_hash(),
        catalog_hash=registry.catalog("dashscope").content_hash(),
        connection_hash=connection.content_hash(),
        implementation=implementation,
        environment_profile_hash=base.manifest.environment_profile_hash,
        verified_capabilities=("tool_calling",),
        context_window_tokens=262144,
        max_output_tokens=profile.max_output_tokens,
        report_hash=canonical_hash(
            {"phase": "UNVERIFIED_EXPLICIT_TEST_CANDIDATE", "nonce": uuid4().hex}
        ),
        verified_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    probe_budget = base.profile.budget.model_copy(
        update={"max_input_tokens": 128000, "max_total_tokens": 250000}
    )
    selected = resolve_model_profile(
        registry,
        configuration,
        profile.ref,
        installed=(implementation,),
        evidence=(candidate,),
        budget=probe_budget,
        environment_profile_hash=base.manifest.environment_profile_hash,
        now=now,
        safety_margin_tokens=base.profile.safety_margin_tokens,
    )
    execution = ExecutionProfile.model_validate(
        {
            **base.profile.model_dump(mode="json"),
            "budget": probe_budget.model_dump(mode="json"),
            "route": selected.route.model_dump(mode="json"),
        }
    )
    profiles = base.manifest.profiles.as_object()
    profiles["execution"] = execution.model_dump(mode="json")
    profiles["model_binding"] = selected.snapshot.as_object()
    profiles["verification_phase"] = "isolated-unverified-candidate"
    manifest = build_development_release(
        spec=base.manifest.spec,
        source_hash=base.manifest.source_tree_hash,
        lock_hash=base.manifest.dependency_lock_hash,
        prompt=base.engine.prompts[base.manifest.spec.release_id],
        tool_hashes={
            key: str(value) for key, value in base.manifest.tool_schema_hashes.as_object().items()
        },
        model_profile=JsonSnapshot.capture(execution.route.model_dump(mode="json")),
        profiles=JsonSnapshot.capture(profiles),
        environment_hash=base.manifest.environment_profile_hash,
    )
    identities = base.store.identity_source.identities()
    actor = next(value for value in identities.values() if value.subject_id == "requester")
    actor = Principal.model_validate(actor)
    store = base.store
    store.installed_manifest_hashes = frozenset({manifest.content_hash()})
    await store.register_release(actor, manifest)
    context = await store.create_evaluation(
        actor,
        EvaluationRequest(
            candidate_release_id=manifest.spec.release_id,
            dataset_hash="c" * 64,
            gate_policy_version="explicit-provider-probe@1",
            resource_scopes=actor.resource_scopes,
            expires_at=now + timedelta(minutes=15),
        ),
    )
    session = await store.create_session(
        actor,
        SessionRequest(agent_id=manifest.spec.agent_id, evaluation_context_id=context.context_id),
    )
    await store.create_run(
        actor,
        session,
        RunRequest(
            request_id=uuid4().hex,
            message="仅验证新适配器协议，不执行业务工具",
            expected_revision=0,
        ),
    )
    owner = uuid4().hex
    await store.register_worker(owner)
    claimed = await store.claim(owner)
    if claimed is None:
        raise RuntimeError("隔离探针未获得自身执行槽")
    await store.started(owner, claimed, manifest.content_hash())
    try:
        tool = ToolSpec.model_validate(base.reads.entries["get_work_order"].spec.as_object())
        tools = JsonSnapshot.capture(
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": "按订单编号查询合成工单",
                        "parameters": tool.input_schema,
                    },
                }
            ]
        )
        messages = JsonSnapshot.capture(
            [
                {
                    "role": "system",
                    "content": (
                        "本轮是工具协议测试。忽略容量填充文本，只调用get_work_order一次，"
                        "参数order_id必须是synthetic-capability-probe，不要输出普通回答。"
                    ),
                },
                {
                    "role": "user",
                    "content": "容量填充开始："
                    + "测" * 20000
                    + "。容量填充结束。现在查询synthetic-capability-probe。",
                },
            ]
        )
        transport = OpenAIChatCompletionsClient(
            selected, lambda: load_qwen_settings(root / ".env", include_retrieval=False).api_key
        )
        result = await GovernedModelGateway(store, transport, execution, base.extensions).complete(
            claimed, owner, messages, tools
        )
        calls = result.tool_calls.as_object_array()
        assert len(calls) == 1 and calls[0]["function"]["name"] == tool.name
        assert (
            json.loads(calls[0]["function"]["arguments"])["order_id"]
            == "synthetic-capability-probe"
        )
        assert result.usage.input_tokens is not None and result.usage.output_tokens is not None
        observed_capacity = result.usage.input_tokens + result.usage.output_tokens
        required_capacity = (
            base.profile.budget.max_input_tokens
            + profile.max_output_tokens
            + base.profile.safety_margin_tokens
        )
        assert observed_capacity >= required_capacity, (
            "实测输入输出量不足以支持本地准入预算，不晋升证据"
        )
        report = {
            "phase": "real-probe-passed",
            "adapter_ref": implementation.adapter_ref,
            "provider": "dashscope",
            "model": profile.model_id,
            "manifest_hash": manifest.content_hash(),
            "run_id": claimed.run_id,
            "model_call_id": result.model_call_id,
            "request_fingerprint": result.request_fingerprint,
            "environment_profile_hash": manifest.environment_profile_hash,
            "source_tree_hash": manifest.source_tree_hash,
            "dependency_lock_hash": manifest.dependency_lock_hash,
            "tool_call_valid": True,
            "usage": result.usage.model_dump(mode="json"),
            "observed_capacity_lower_bound": observed_capacity,
            "admission_capacity_required": required_capacity,
            "configured_output_accepted": profile.max_output_tokens,
            "verified_at": datetime.now(UTC).isoformat(),
        }
        report_hash = canonical_hash(report)
        verified = candidate.model_copy(
            update={
                "context_window_tokens": observed_capacity,
                "report_hash": report_hash,
                "verified_at": datetime.now(UTC),
                "expires_at": datetime.now(UTC) + timedelta(hours=24),
            }
        )
        # 晋升时再次按正式Schema校验，不保存候选的虚构窗口或未验证report_hash。
        verified = ModelCapabilityEvidence.model_validate_json(verified.model_dump_json())
        (folder / "m2r07-newproviders-capability-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
        )
        (folder / "m2r07-newproviders-config.json").write_text(
            configuration.model_dump_json(indent=2), "utf-8"
        )
        (folder / "m2r07-newproviders-evidence.json").write_text(
            json.dumps(
                {
                    "version": "model-evidence@1",
                    "evidence": [verified.model_dump(mode="json")],
                    "revoked_refs": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "utf-8",
        )
        environment = {
            key: value
            for key, value in dotenv_values(folder / "m2r07.env").items()
            if value is not None
        }
        environment.update(
            {
                "HARNESS_MODEL_PROVIDERS_FILE": ".runtime/m2r07-newproviders-config.json",
                "HARNESS_MODEL_EVIDENCE_FILE": ".runtime/m2r07-newproviders-evidence.json",
                "HARNESS_ACCEPTANCE_PREFIX": "m2r07-newproviders",
                "HARNESS_ACCEPTANCE_EXPECT_ADAPTER": implementation.adapter_ref,
            }
        )
        (folder / "m2r07-newproviders.env").write_text(
            "".join(key + "=" + value + "\n" for key, value in environment.items()), "utf-8"
        )
        print("新适配器真实探针通过；正式能力证据与专用验收环境已在本地生成。", flush=True)
        print(
            json.dumps(
                {
                    "adapter_ref": implementation.adapter_ref,
                    "observed_capacity_lower_bound": observed_capacity,
                    "report_hash": report_hash,
                }
            ),
            flush=True,
        )
    finally:
        # 这是协议探针，不将没有执行业务工具的调用标成任务成功或写入成功历史。
        await store.fail(owner, claimed.run_id, "DIAGNOSTIC_PROBE_FINISHED")
        await store.finish_slot(owner)
        await store.release_worker(owner)


if __name__ == "__main__":
    run(probe(Path.cwd().resolve()))
