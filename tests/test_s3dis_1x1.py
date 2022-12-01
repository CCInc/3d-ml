import os
import shutil

import pytest
import torch

from src.datamodules.common import DataModuleConfig
from src.datamodules.segmentation.s3dis_1x1_module import S3DIS1x1DataModule


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [32, 128])
def test_s3dis_1x1(tmp_path, batch_size):
    config = DataModuleConfig(
        data_dir=os.path.join(tmp_path, "s3dis_1x1"),
        batch_size=batch_size
    )

    dm = S3DIS1x1DataModule(config=config)
    dm.prepare_data()
    assert not dm.data_train and not dm.data_test
    assert os.path.exists(config.data_dir)

    dm.setup()
    assert dm.data_train and dm.data_test
    assert dm.train_dataloader() and dm.test_dataloader()
    assert dm.num_classes == 13
    assert dm.num_feats == 6
    assert len(dm.data_train) == 20291
    assert len(dm.data_test) == 3294

    batch = next(iter(dm.train_dataloader()))
    assert batch.x.dtype == torch.float32
    assert batch.x.size() == torch.Size([batch_size, 4096, 6])
    assert batch.pos.dtype == torch.float32
    assert batch.pos.size() == torch.Size([batch_size, 4096, 3])
    assert batch.y.dtype == torch.int64
    assert batch.y.size() == torch.Size([batch_size, 4096])

    # rm data folder to save space
    shutil.rmtree(tmp_path)
