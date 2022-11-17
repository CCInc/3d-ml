from pathlib import Path

import pytest
import torch

from src.datamodules.classification.modelnet2048_module import ModelNet2048DataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_modelnet2048(batch_size):
    data_dir = "data/modelnet2048"

    dm = ModelNet2048DataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_test
    assert Path(data_dir).exists()

    dm.setup()
    assert dm.data_train and dm.data_test
    assert dm.train_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_test)
    assert num_datapoints == 12_308

    batch = next(iter(dm.train_dataloader()))
    x, y = batch.pos, batch.y
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
