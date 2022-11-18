from typing import Optional

from src.datamodules.base_dataloader import Base3dDataModule
from src.datamodules.classification.components.modelnet2048 import ModelNet2048Dataset
from src.datamodules.common import DataModuleConfig, DataModuleTransforms
from src.datamodules.components.download import download_and_extract_archive


class ModelNet2048DataModule(Base3dDataModule):
    dataset_md5 = "c9ab8e6dfb16f67afdab25e155c79e59"
    dataset_url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

    def __init__(
        self,
        config: DataModuleConfig = DataModuleConfig(),
        transforms: DataModuleTransforms = DataModuleTransforms(),
    ):
        super().__init__(config, transforms)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        return 40

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        download_and_extract_archive(self.dataset_url, self.config.data_dir, self.dataset_md5)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = ModelNet2048Dataset(
                self.config.data_dir, "train", self.transforms.train
            )
            self.data_test = ModelNet2048Dataset(
                self.config.data_dir, "test", self.transforms.test
            )
