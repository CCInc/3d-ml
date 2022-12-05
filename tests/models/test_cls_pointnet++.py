import hydra
import pytest

from tests.helpers.helpers import count_trainable_params


@pytest.mark.parametrize("cfg_model", ["cls_pointnet++.yaml"], indirect=["cfg_model"])
def test_cls_pointnetpp(cfg_model):

    model = hydra.utils.instantiate(cfg_model.model, num_classes=2, num_feats=6)

    assert count_trainable_params(model) == 1_462_978
