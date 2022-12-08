import hydra
import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.utils import count_trainable_params

test_cases = [("seg_pointnet++.yaml", 2, 6, 963682), ("seg_pointnet++.yaml", 4, 8, 964260)]


@RunIf(openpoints=True)
@pytest.mark.parametrize(
    "cfg_model, num_classes, num_feats, num_params", test_cases, indirect=["cfg_model"]
)
def test_seg_pointnetpp(cfg_model, num_classes, num_feats, num_params):
    model = hydra.utils.instantiate(cfg_model.model, num_classes=num_classes, num_feats=num_feats)

    assert count_trainable_params(model) == num_params
    assert model.net.encoder.channel_list[0] == num_feats
    assert model.net.head.head[-1][0].out_channels == num_classes
