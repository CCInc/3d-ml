from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.common import DataModuleConfig, DataModuleTransforms
from src.utils.batch import SimpleBatch


class Base3dDataModule(LightningDataModule):
    def __init__(
        self,
        config: DataModuleConfig = DataModuleConfig(),
        transforms: DataModuleTransforms = DataModuleTransforms(),
    ):
        super().__init__()

        self.config = config
        self.transforms = transforms

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        if self.data_train is not None:
            return self.data_train.num_classes
        if self.data_val is not None:
            return self.data_val.num_classes
        if self.data_test is not None:
            return self.data_test.num_classes
        raise NotImplementedError()

    @property
    def num_feats(self):
        if self.data_train is not None:
            return self.data_train.num_features
        if self.data_val is not None:
            return self.data_val.num_features
        if self.data_test is not None:
            return self.data_test.num_features
        raise NotImplementedError()

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
            collate_fn=SimpleBatch.from_data_list,
            dataset=self.data_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.data_val:
            return DataLoader(
                collate_fn=SimpleBatch.from_data_list,
                dataset=self.data_val,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                shuffle=False,
            )
        else:
            return None

    def test_dataloader(self):
        return DataLoader(
            collate_fn=SimpleBatch.from_data_list,
            dataset=self.data_test,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
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
