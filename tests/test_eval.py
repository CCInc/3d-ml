import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train
from tests.helpers.run_if import RunIf

test_experiments = [
    (
        ["experiment=cls_modelnet_pointnet++"],
        ["model=cls_pointnet++", "data=cls_modelnet2048", "ckpt_path=."],
        1,
    ),
    (
        ["experiment=seg_s3dis1x1_pointnet++"],
        ["model=seg_pointnet++", "data=seg_s3dis1x1", "ckpt_path=."],
        2,
    ),
]


@pytest.mark.slow
@RunIf(openpoints=True, min_gpus=1)
@pytest.mark.parametrize(
    "cfg_train, cfg_eval, max_epochs", test_experiments, indirect=["cfg_train", "cfg_eval"]
)
def test_eval(tmp_path: str, cfg_train: DictConfig, cfg_eval: DictConfig, max_epochs: int):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = max_epochs
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(os.path.join(tmp_path, "checkpoints"))

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = os.path.join(tmp_path, "checkpoints/last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001
