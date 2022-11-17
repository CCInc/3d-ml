from dataclasses import dataclass
from typing import Optional


@dataclass
class LrSchedulerConfig:
    monitor: str = "val/loss"
    interval: str = "epoch"
    frequency: int = 1


@dataclass
class LrScheduler:
    scheduler: Optional[object] = None
    config: Optional[LrSchedulerConfig] = None
