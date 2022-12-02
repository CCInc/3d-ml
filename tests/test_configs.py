import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.data.datamodule
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "cpu"

    datamodule = hydra.utils.instantiate(cfg_train.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    hydra.utils.instantiate(
        cfg_train.model, num_classes=datamodule.num_classes, num_feats=datamodule.num_feats
    )
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.data.datamodule
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    with open_dict(cfg_eval):
        cfg_eval.trainer.accelerator = "cpu"

    datamodule = hydra.utils.instantiate(cfg_eval.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    hydra.utils.instantiate(
        cfg_eval.model, num_classes=datamodule.num_classes, num_feats=datamodule.num_feats
    )
    hydra.utils.instantiate(cfg_eval.trainer)
