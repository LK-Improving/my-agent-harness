"""单执行槽协调器；退出前等待引擎及其子任务真正停止。"""

import asyncio
import logging
from uuid import uuid4

from harness.contracts.failures import ProviderError
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
    ) -> None:
        self.store = store
        self.engine = engine
        # 每次进程启动必须生成新身份，禁止借同一配置身份接管旧登记。
        self.identity = identity or uuid4().hex
        self.checkpoint_timeout_seconds = checkpoint_timeout_seconds
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
        observed = self.engine.observed_manifest_hash(run.manifest)
        await self.store.started(self.identity, run, observed)
        execution = asyncio.create_task(self.engine.execute(run))
        monitor = asyncio.create_task(self._monitor(run))
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
            await self.store.complete(self.identity, run, candidate)
        finally:
            execution.cancel()
            monitor.cancel()
            # 引擎必须同时排空内部子任务；未排空就不返回、不解除持久登记。
            await asyncio.gather(execution, monitor, return_exceptions=True)

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
                        await self.store.fail(
                            self.identity, run.run_id, "CHECKPOINT_SUSPENSION_NOT_INSTALLED"
                        )
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
                            await self.store.fail(
                                self.identity, run.run_id, "CHECKPOINT_SUSPENSION_UNVERIFIED"
                            )
                except ProviderError as exc:
                    await self.store.fail(self.identity, run.run_id, exc.failure.code.value)
                except RuntimeFault as exc:
                    await self.store.fail(self.identity, run.run_id, exc.code)
                except Exception:
                    # 仅记录关联身份，禁止把异常正文/请求/凭据写入日志。
                    logger.error("run_failed run_id=%s worker_id=%s", run.run_id, self.identity)
                    await self.store.fail(self.identity, run.run_id, "EXECUTION_FAILED")
                await self.store.finish_slot(self.identity)
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
