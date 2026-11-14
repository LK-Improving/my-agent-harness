"""可选诊断接收端仅接收固定字段。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Observation:
    trace_id: str
    event_type: str
    seq: int
    timestamp: datetime


class ObservationSink(Protocol):
    def offer(self, observation: Observation) -> None:
        """非阻塞尝试收取，可在背压时丢弃。"""
        ...


class NoopObservationSink:
    def offer(self, observation: Observation) -> None:
        pass
