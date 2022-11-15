from dataclasses import dataclass
from typing import Optional
from src.transforms import BaseTransform

@dataclass
class DataModuleTransforms:
    train: Optional[BaseTransform] = None
    val: Optional[BaseTransform] = None
    test: Optional[BaseTransform] = None
