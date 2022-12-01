import os
import shutil

import pytest
import torch

from src.datamodules.classification.modelnet2048_module import ModelNet2048DataModule
from src.datamodules.common import DataModuleConfig


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [32, 128])
def test_modelnet2048(tmp_path, batch_size):
    config = DataModuleConfig(
        data_dir=os.path.join(tmp_path, "modelnet2048"),
        batch_size=batch_size
    )

    dm = ModelNet2048DataModule(config=config)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_test
    assert os.path.exists(config.data_dir)

    dm.setup()
    assert dm.data_train and dm.data_test
    assert dm.train_dataloader() and dm.test_dataloader()
    assert dm.num_classes == 40
    assert dm.num_feats == 0
    assert len(dm.data_train) == 9840
    assert len(dm.data_test) == 2468

    batch = next(iter(dm.train_dataloader()))
    assert batch.x is None
    assert batch.pos.dtype == torch.float32
    assert batch.pos.size() == torch.Size([batch_size, 2048, 3])
    assert batch.y.dtype == torch.int64
    assert batch.y.size() == torch.Size([batch_size])

    # rm data folder to save space
    shutil.rmtree(config.data_dir)
