import hydra
import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.utils import count_trainable_params

test_cases = [("cls_pointnet++.yaml", 2, 6, 1_462_978), ("cls_pointnet++.yaml", 4, 8, 1_463_620)]


@RunIf(openpoints=True)
@pytest.mark.parametrize(
    "cfg_model, num_classes, num_feats, num_params", test_cases, indirect=["cfg_model"]
)
def test_cls_pointnetpp(cfg_model, num_classes, num_feats, num_params):
    model = hydra.utils.instantiate(cfg_model.model, num_classes=num_classes, num_feats=num_feats)

    assert count_trainable_params(model) == num_params
    assert model.net.encoder.channel_list[0] == num_feats
    assert model.net.prediction.head[-1][0].out_features == num_classes
