import os

import pytest
from _pytest.fixtures import SubRequest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(tmp_path: str, request: SubRequest) -> DictConfig:
    """
    Creates a training config set to a pytest temporary path
    Args:
        tmp_path(str): Pytest temporary path
        request(SubRequest): List of parameter overrides, used to dynamically change configs to test

    Returns:
        (DictConfig): The Hydra config object
    """
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=request.param,
        )

    # set defaults for all tests
    with open_dict(cfg):
        cfg.paths.root_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
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

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(tmp_path: str, request: SubRequest) -> DictConfig:
    """
    Creates an evaluation config set to a pytest temporary path
    Args:
        tmp_path(str): Pytest temporary path
        request(SubRequest): List of parameter overrides, used to dynamically change configs to test

    Returns:
        (DictConfig): The Hydra config object
    """

    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=request.param,
        )

    # set defaults for all tests
    with open_dict(cfg):
        cfg.paths.root_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_test_batches = 0.1
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1
        cfg.data.datamodule.config.num_workers = 0
        cfg.data.datamodule.config.pin_memory = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = None

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_model(request: SubRequest) -> DictConfig:
    """
    Creates a Model config to test models individually
    Args:
        request(SubRequest): List of parameter overrides, used to dynamically change configs to test

    Returns:
        (DictConfig): The Hydra config object
    """
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name=os.path.join("model", request.param),
            return_hydra_config=True,
        )

    # Manufacture the required data config
    with open_dict(cfg):
        cfg.data = {}
        cfg.data.monitor_split = "Train"

    yield cfg

    GlobalHydra.instance().clear()
