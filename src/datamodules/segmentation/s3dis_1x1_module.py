from dataclasses import dataclass
from typing import Optional

from torch_geometric.datasets import S3DIS

from src.datamodules.base_dataloader import Base3dDataModule
from src.datamodules.common import DataModuleConfig, DataModuleTransforms


@dataclass
class S3DIS1x1Config:
    test_area: int = 6


class S3DIS1x1DataModule(Base3dDataModule):
    def __init__(
        self,
        config: DataModuleConfig = DataModuleConfig(),
        transforms: DataModuleTransforms = DataModuleTransforms(),
        dataset_config: S3DIS1x1Config = S3DIS1x1Config(),
    ):
        super().__init__(config, transforms)
        self.dataset_config = dataset_config

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        S3DIS(
            self.config.data_dir,
            train=True,
            transform=self.transforms.train,
            test_area=self.dataset_config.test_area,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = S3DIS(
                self.config.data_dir,
                train=True,
                transform=self.transforms.train,
                test_area=self.dataset_config.test_area,
            )
            self.data_test = S3DIS(
                self.config.data_dir,
                train=False,
                transform=self.transforms.test,
                test_area=self.dataset_config.test_area,
            )
