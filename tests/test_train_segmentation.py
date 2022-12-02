import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.train import train
from tests.helpers.run_if import RunIf


@RunIf(openpoints=True, min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train_seg):
    """Run for 1 train, val and test step on GPU."""
    HydraConfig().set_config(cfg_train_seg)
    with open_dict(cfg_train_seg):
        cfg_train_seg.trainer.fast_dev_run = True
    train(cfg_train_seg)


@RunIf(openpoints=True, min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train_seg):
    """Train 1 epoch on GPU with mixed-precision."""
    HydraConfig().set_config(cfg_train_seg)
    with open_dict(cfg_train_seg):
        cfg_train_seg.trainer.max_epochs = 1
        cfg_train_seg.trainer.precision = 16
    train(cfg_train_seg)


@RunIf(openpoints=True, min_gpus=1)
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train_seg):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_train_seg)
    with open_dict(cfg_train_seg):
        cfg_train_seg.trainer.max_epochs = 1
        cfg_train_seg.trainer.val_check_interval = 0.5
    train(cfg_train_seg)


@RunIf(openpoints=True, min_gpus=1)
@pytest.mark.slow
def test_train_resume(tmp_path, cfg_train_seg):
    """Run 1 epoch, finish, and resume for another epoch."""
    with open_dict(cfg_train_seg):
        cfg_train_seg.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train_seg)
    metric_dict_1, _ = train(cfg_train_seg)

    files = os.listdir(os.path.join(tmp_path, "checkpoints"))
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train_seg):
        cfg_train_seg.ckpt_path = os.path.join(tmp_path, "checkpoints", "last.ckpt")
        cfg_train_seg.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train_seg)

    files = os.listdir(os.path.join(tmp_path, "checkpoints"))
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
