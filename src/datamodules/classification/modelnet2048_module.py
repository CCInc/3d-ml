if __name__ == "__main__":
    import pyrootutils
    root = pyrootutils.setup_root(__file__, pythonpath=True)

import glob
import os
from typing import Any, Dict, Optional, Tuple
from src.datamodules.classification.components.modelnet2048 import ModelNet2048Dataset
from src.datamodules.components.download import download_and_extract_archive

import h5py
import numpy as np

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torchvision.transforms import transforms


class ModelNet2048DataModule(LightningDataModule):
    dataset_md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    dataset_url = f'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[ModelNet2048Dataset] = None
        self.data_test: Optional[ModelNet2048Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        print(self.hparams.data_dir)
        download_and_extract_archive(self.dataset_url, self.hparams.data_dir, self.dataset_md5)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = ModelNet2048Dataset(self.hparams.data_dir, "train")
            self.data_test = ModelNet2048Dataset(self.hparams.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data" / "modelnet")
    dm = hydra.utils.instantiate(cfg)
    dm.prepare_data()

    # splits/transforms
    dm.setup(stage="fit")

    # use data
    for batch in dm.train_dataloader():
        print(batch, batch.batch)
