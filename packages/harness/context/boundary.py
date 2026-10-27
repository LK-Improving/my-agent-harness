"""首期会话记忆能力与完整输入硬上限；不加载供应商或静默裁剪。"""

from pydantic import ValidationError

from harness.contracts.context import MemoryBoundaryProfile
from harness.contracts.execution import ExecutionProfile
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import RuntimeFault


def require_memory_boundary(release: ReleaseManifest) -> MemoryBoundaryProfile:
    value = release.profiles.as_object().get("context_boundary")
    try:
        boundary = MemoryBoundaryProfile.model_validate({} if value is None else value)
    except ValidationError:
        raise RuntimeFault("MEMORY_CAPABILITY_NOT_INSTALLED", 422) from None
    if release.spec.memory_profile_ref != boundary.memory_profile_ref:
        raise RuntimeFault("MEMORY_PROFILE_NOT_INSTALLED", 422)
    return boundary


def context_input_limit(profile: ExecutionProfile) -> int:
    window = profile.route.context_window_tokens
    if window is None:
        raise RuntimeFault("MODEL_CONTEXT_WINDOW_UNVERIFIED", 503)
    return min(profile.budget.max_input_tokens,
               window - profile.budget.max_output_tokens - profile.safety_margin_tokens)


def require_context_capacity(input_tokens: int, profile: ExecutionProfile) -> None:
    if input_tokens > context_input_limit(profile):
        raise RuntimeFault("CONTEXT_BUDGET_EXCEEDED", 429)
