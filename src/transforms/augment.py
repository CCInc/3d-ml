import torch
from typing import Sequence
from torch_geometric.data import Data

### Ref:
# https://github.com/torch-points3d/torch-points3d/blob/66e8bf22b2d98adca804c753ac3f0013ff4ec731/torch_points3d/core/data_transform/transforms.py#L517-L555
# https://github.com/guochengqian/openpoints/blob/ed0500b304597253717ba618d0a41d5286e48792/transforms/point_transformer_gpu.py#L183-L213
###
class RandomScaleAnisotropic:
    r""" Scales node positions by a randomly sampled factor ``s1, s2, s3`` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \left[
        \begin{array}{ccc}
            s1 & 0 & 0 \\
            0 & s2 & 0 \\
            0 & 0 & s3 \\
        \end{array}
        \right]


    for three-dimensional positions.

    Parameters
    -----------
    scale:
        scaling factor interval, e.g. ``(a, b)``, then scale \
        is randomly sampled from the range \
        ``a <=  b``. \
    """

    def __init__(self, scale: Sequence=[2. / 3, 3. / 2]):
        assert len(scale) == 2
        self.scale_min, self.scale_max = scale
        assert self.scale_min <= self.scale_max

    def __call__(self, data: Data) -> Data:
        scale = self.scale_max + torch.rand(3) * (self.scale_max - self.scale_min)
        data.pos = data.pos * scale
        # Not sure if the following is needed. Seems to have been used for modelnet in tp3d.
        if "norm" in data:
            data.norm = data.norm / scale
            data.norm = torch.nn.functional.normalize(data.norm, dim=1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scale_min}, {self.scale_max})'

### Ref:
# https://github.com/guochengqian/openpoints/blob/ed0500b304597253717ba618d0a41d5286e48792/transforms/point_transformer_gpu.py#L183-L213
###
class RandomTranslate:
    r""" Translates node positions by a randomly sampled factor ``t1, t2, t3`` within a
    given interval.

    Parameters
    -----------
    delta:
        per-axis translation interval, e.g. ``[t1, t2, t3]``, then translation \
        is randomly sampled from the range \
        ``[(-t1, t1), (-t2, t2), (-t3, t3)]``. \
    """

    def __init__(self, delta: Sequence=[0.2, 0.2, 0.2]):
        assert len(delta) == 3
        self.delta = delta

    def __call__(self, data: Data) -> Data:
        translation = torch.rand(3) # in the range [0, 1]
        translation = (translation - 0.5) * 2 # rescale to [-1, 1]
        translation *= torch.tensor(self.delta) # rescale to [(-t1, t1), (-t2, t2), (-t3, t3)]

        data.pos = data.pos + translation # apply translation
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.delta})'