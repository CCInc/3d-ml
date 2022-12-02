import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from tests.helpers.run_if import RunIf

train_experiments = [
    ["experiment=cls_modelnet_pointnet++"],
    ["experiment=seg_s3dis1x1_pointnet++"],
]


@RunIf(openpoints=True)
@pytest.mark.parametrize("cfg_train", train_experiments, indirect=["cfg_train"])
def test_train_config(cfg_train: DictConfig):
    _test_config(cfg_train)


eval_experiments = [
    ["model=cls_pointnet++", "data=cls_modelnet2048", "ckpt_path=."],
    ["model=seg_pointnet++", "data=seg_s3dis1x1", "ckpt_path=."],
]


@RunIf(openpoints=True)
@pytest.mark.parametrize("cfg_eval", eval_experiments, indirect=["cfg_eval"])
def test_eval_config(cfg_eval: DictConfig):
    _test_config(cfg_eval)


def _test_config(cfg: DictConfig):
    assert cfg
    assert cfg.data.datamodule
    assert cfg.model
    assert cfg.trainer

    HydraConfig().set_config(cfg)

    with open_dict(cfg):
        cfg.trainer.accelerator = "cpu"

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    hydra.utils.instantiate(
        cfg.model, num_classes=datamodule.num_classes, num_feats=datamodule.num_feats
    )
    hydra.utils.instantiate(cfg.trainer)
