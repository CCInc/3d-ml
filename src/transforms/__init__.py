from typing import Any, Optional
from src.transforms.augment import RandomScaleAnisotropic, RandomTranslate
from torch_geometric.transforms import FixedPoints
from torch_geometric.transforms import BaseTransform

# This module exposes all transforms from a fixed location

def Compose(transforms: Optional[list[Any]]) -> BaseTransform:
    """Composes several transforms together, flattening them if they are multi-dimensional

    Args:
        transforms (Optional[list[Any]]): List of transforms to flatten

    Returns:
        BaseTransform: A single composed transform
    """
    # put these imports inside the function body to not pollute the module
    from src.utils.utils import flatten_nested_lists
    import torch_geometric.transforms

    # Flatten the arbitrarily nested transform list
    flattened_transforms = flatten_nested_lists(transforms)
    # Remove any null values
    flattened_transforms = [t for t in flattened_transforms if t is not None]
    # Compose into a single transform
    return torch_geometric.transforms.Compose(flattened_transforms)