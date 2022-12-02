from typing import List

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(tmp_path, request) -> DictConfig:

    cfg = cfg_train_global(overrides=request.param)

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(tmp_path, request) -> DictConfig:

    cfg = cfg_eval_global(overrides=request.param)
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


def cfg_train_global(overrides: List[str] = None) -> DictConfig:
    if overrides is None:
        overrides = []
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=overrides,
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "gpu"
            cfg.trainer.devices = 1
            cfg.data.datamodule.config.num_workers = 0
            cfg.data.datamodule.config.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


def cfg_eval_global(overrides: List[str] = None) -> DictConfig:
    if overrides is None:
        overrides = []
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=overrides,
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "gpu"
            cfg.trainer.devices = 1
            cfg.data.datamodule.config.num_workers = 0
            cfg.data.datamodule.config.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg
