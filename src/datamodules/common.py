from dataclasses import dataclass
from typing import Optional

from src.transforms import BaseTransform


@dataclass
class DataModuleTransforms:
    train: Optional[BaseTransform] = None
    val: Optional[BaseTransform] = None
    test: Optional[BaseTransform] = None


@dataclass
class DataModuleConfig:
    data_dir: Optional[str] = "data/"
    batch_size: Optional[int] = 64
    num_workers: Optional[int] = 0
    pin_memory: Optional[bool] = False
