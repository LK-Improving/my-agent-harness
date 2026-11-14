"""单执行槽协调器；退出前等待引擎及其子任务真正停止。"""

import asyncio
import logging
from collections.abc import Callable
from uuid import uuid4

from harness.contracts.failures import ProviderError
from harness.contracts.evaluation_versions import ExecutionVersion
from harness.contracts.release import ReleaseManifest
from harness.contracts.runtime import ClaimedRun, RunEngine, RuntimeFault, RuntimeStore
from harness.contracts.writes import ExecutionSuspended, SuspensionProof

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        store: RuntimeStore,
        engine: RunEngine,
        *,
        identity: str | None = None,
        checkpoint_timeout_seconds: float = 10,
        version_observer: Callable[[ReleaseManifest], ExecutionVersion] | None = None,
    ) -> None:
        self.store = store
        self.engine = engine
        # 每次进程启动必须生成新身份，禁止借同一配置身份接管旧登记。
        self.identity = identity or uuid4().hex
        self.checkpoint_timeout_seconds = checkpoint_timeout_seconds
        self.version_observer = version_observer
        self._terminal_versions: dict[str, tuple[int, bool]] = {}
        self._stop = asyncio.Event()

    def stop_claiming(self) -> None:
        """停止领取，当前执行与心跳继续到安全退出。"""
        self._stop.set()

    async def _monitor(self, run: ClaimedRun) -> None:
        while True:
            await self.store.heartbeat(self.identity)
            if self._stop.is_set():
                await self.store.worker_unready(self.identity)
            if not await self.store.continue_run(self.identity, run.run_id):
                raise RuntimeFault("EXECUTION_STOP_REQUESTED")
            await asyncio.sleep(1)

    async def _execute(self, run: ClaimedRun) -> None:
        self._terminal_versions.pop(run.run_id, None)
        try:
            start_ref = None
            if self.version_observer is not None:
                start_ref, valid = await self._observe_version(run, "start")
                if not valid:
                    raise RuntimeFault("EXECUTION_VERSION_UNVERIFIED", 503)
            observed = self.engine.observed_manifest_hash(run.manifest)
            if start_ref is None:
                await self.store.started(self.identity, run, observed)
            else:
                await self.store.started(self.identity, run, observed, version_ref=start_ref)
        except Exception:
            if self.version_observer is not None:
                await self._terminal_version(run)
            raise
        execution = asyncio.create_task(self.engine.execute(run))
        monitor = asyncio.create_task(self._monitor(run))
        suspended = False
        try:
            done, _ = await asyncio.wait({execution, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if monitor in done:
                # 连接丢失、撤权、取消或超时先停止执行。不能先释放槽再等待任务。
                await monitor
                raise RuntimeFault("EXECUTION_MONITOR_STOPPED")
            candidate = await execution
            before_commit = getattr(self.engine, "before_commit", None)
            if before_commit is not None:
                await before_commit(run, candidate)
            if self.version_observer is None:
                await self.store.complete(self.identity, run, candidate)
            else:
                reference, valid = await self._terminal_version(run)
                if not valid:
                    raise RuntimeFault("EXECUTION_VERSION_UNVERIFIED", 503)
                await self.store.complete(self.identity, run, candidate, version_ref=reference)
        except ExecutionSuspended:
            suspended = True
            raise
        finally:
            execution.cancel()
            monitor.cancel()
            # 引擎必须同时排空内部子任务；未排空就不返回、不解除持久登记。
            await asyncio.gather(execution, monitor, return_exceptions=True)
            if not suspended and self.version_observer is not None and run.run_id not in self._terminal_versions:
                await self._terminal_version(run)

    async def _observe_version(self, run: ClaimedRun, phase: str) -> tuple[int, bool]:
        observation = None
        error_code = None
        try:
            if self.version_observer is None:
                raise RuntimeFault("EXECUTION_VERSION_OBSERVER_NOT_INSTALLED", 503)
            observation = ExecutionVersion.model_validate(await asyncio.to_thread(self.version_observer, run.manifest))
        except Exception as error:
            error_code = type(error).__name__
        reference = await self.store.record_execution_version(
            self.identity, run, phase, observation, error_code)
        return reference, observation is not None and not observation.differences(run.manifest)

    async def _terminal_version(self, run: ClaimedRun) -> tuple[int, bool]:
        if run.run_id not in self._terminal_versions:
            self._terminal_versions[run.run_id] = await self._observe_version(run, "terminal")
        return self._terminal_versions[run.run_id]

    async def _fail(self, run: ClaimedRun, code: str) -> None:
        if self.version_observer is None:
            await self.store.fail(self.identity, run.run_id, code)
        else:
            reference, _ = await self._terminal_version(run)
            await self.store.fail(self.identity, run.run_id, code, version_ref=reference)

    async def serve(self) -> None:
        await self.store.register_worker(self.identity)
        drained = False
        try:
            while not self._stop.is_set():
                availability = getattr(self.engine, "execution_available", None)
                if availability is not None and not availability():
                    raise RuntimeFault("EXECUTION_DEPENDENCY_UNAVAILABLE", 503)
                await self.store.heartbeat(self.identity)
                run = await self.store.claim(self.identity)
                if run is None:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=0.5)
                    except TimeoutError:
                        pass
                    continue
                try:
                    await self._execute(run)
                except ExecutionSuspended as exc:
                    # _execute的finally已排空任务；引擎还必须查验真实checkpoint/interrupt。
                    proof_reader = getattr(self.engine, "suspension_proof", None)
                    if proof_reader is None:
                        await self._fail(run, "CHECKPOINT_SUSPENSION_NOT_INSTALLED")
                    else:
                        try:
                            async with asyncio.timeout(self.checkpoint_timeout_seconds):
                                proof = SuspensionProof.model_validate(
                                    await proof_reader(run, exc.operation_id)
                                )
                            if proof.operation_id != exc.operation_id or proof.run_id != run.run_id:
                                raise RuntimeFault("CHECKPOINT_SUSPENSION_MISMATCH")
                            await self.store.park_write(self.identity, proof)
                        except Exception:
                            await self._fail(run, "CHECKPOINT_SUSPENSION_UNVERIFIED")
                except ProviderError as exc:
                    await self._fail(run, exc.failure.code.value)
                except RuntimeFault as exc:
                    await self._fail(run, exc.code)
                except Exception:
                    # 仅记录关联身份，禁止把异常正文/请求/凭据写入日志。
                    logger.error("run_failed run_id=%s worker_id=%s", run.run_id, self.identity)
                    await self._fail(run, "EXECUTION_FAILED")
                await self.store.finish_slot(self.identity)
                self._terminal_versions.pop(run.run_id, None)
            drained = True
        finally:
            # 数据库异常或外部强制取消保留登记，后续进程不得自动接管。
            if drained:
                await self.store.release_worker(self.identity)
            else:
                try:
                    await self.store.worker_unready(self.identity)
                except Exception:
                    # 数据库故障时保留原登记，不能借清理失败转让执行权。
                    pass
