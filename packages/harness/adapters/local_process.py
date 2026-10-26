"""本机进程终止核验；不终止进程，不接受心跳或PID字符串作为证明。"""

import os
import platform
import secrets
import uuid
from pathlib import Path

import psutil

from harness.contracts.execution import ProcessIdentity, TerminationProof
from harness.contracts.hashing import canonical_hash
from harness.contracts.runtime import RuntimeFault


def local_host_key() -> str:
    folder = Path.cwd().resolve() / ".runtime"
    folder.mkdir(exist_ok=True)
    path = folder / "recovery-host.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(secrets.token_hex(32))
    value = path.read_text("ascii")
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise RuntimeFault("LOCAL_HOST_IDENTITY_UNAVAILABLE")
    return value


def current_process() -> ProcessIdentity:
    namespace = (
        os.readlink("/proc/self/ns/pid") if Path("/proc/self/ns/pid").exists() else "windows-host"
    )
    return ProcessIdentity(
        host_id=canonical_hash(
            {"node": platform.node(), "machine": str(uuid.getnode()), "local_key": local_host_key()}
        ),
        boot_time=psutil.boot_time(),
        pid=os.getpid(),
        started_at=psutil.Process().create_time(),
        namespace_id=namespace,
        execution_class="local-no-subprocess@1",
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if Path("/proc/sys/kernel/random/boot_id").exists()
        else "unavailable",
        start_marker=process_marker(os.getpid()),
    )


def process_marker(pid: int) -> str:
    process = psutil.Process(pid)
    stat = Path(f"/proc/{pid}/stat")
    if platform.system() == "Linux":
        # 使用内核启动tick，不能因校时改变进程身份。
        return "linux-start-ticks:" + stat.read_text().rsplit(")", 1)[1].split()[19]
    return "kernel-creation-time:" + repr(process.create_time())


def prove_terminated(old: ProcessIdentity) -> TerminationProof:
    local = current_process()
    if old.host_id != local.host_id:
        raise RuntimeFault("RECOVERY_HOST_MISMATCH")
    if (
        old.boot_id != "unavailable"
        and local.boot_id != "unavailable"
        and old.boot_id != local.boot_id
    ):
        return TerminationProof(process=old, reason="local_host_rebooted")
    if old.namespace_id != local.namespace_id:
        raise RuntimeFault("RECOVERY_PROCESS_NAMESPACE_MISMATCH")
    try:
        if process_marker(old.pid) == old.start_marker:
            # 包括停止/僵死/无心跳状态：只要PID仍匹配，就不声明其已终止。
            raise RuntimeFault("OLD_WORKER_STILL_PRESENT")
    except psutil.NoSuchProcess:
        return TerminationProof(process=old, reason="recorded_process_absent")
    except FileNotFoundError:
        if psutil.pid_exists(old.pid):
            raise RuntimeFault("PROCESS_IDENTITY_CHANGED_RETRY") from None
        return TerminationProof(process=old, reason="recorded_process_absent")
    except (psutil.AccessDenied, PermissionError):
        raise RuntimeFault("PROCESS_TERMINATION_NOT_PROVEN") from None
    return TerminationProof(process=old, reason="pid_reused_with_different_start_time")
