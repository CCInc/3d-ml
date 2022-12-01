from pathlib import Path

import pytest
import torch

from src.datamodules.common import DataModuleConfig
from src.datamodules.classification.modelnet2048_module import ModelNet2048DataModule

@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [32, 128])
def test_modelnet2048(batch_size):
    config = DataModuleConfig(
        data_dir="data/modelnet2048",
        batch_size=batch_size
    )

    dm = ModelNet2048DataModule(config=config)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_test
    assert Path(config.data_dir).exists()

    dm.setup()
    assert dm.data_train and dm.data_test
    assert dm.train_dataloader() and dm.test_dataloader()
    assert dm.num_classes == 40
    assert dm.num_feats == 0

    num_datapoints = len(dm.data_train) + len(dm.data_test)
    assert num_datapoints == 12_308

    batch = next(iter(dm.train_dataloader()))
    assert batch.x is None
    x, y = batch.pos, batch.y
    assert x.dtype == torch.float32
    assert x.size() == torch.Size([batch_size, 2048, 3])
    assert y.dtype == torch.int64
    assert y.size() == torch.Size([batch_size])
