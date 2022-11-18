import pytest
import shutil
from pathlib import Path

from src.datamodules.common import DataModuleConfig
from src.datamodules.segmentation.s3dis_1x1_module import S3DIS1x1DataModule

@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [32, 128])
def test_s3dis_1x1(tmp_path, batch_size):
    config = DataModuleConfig(
        data_dir=Path('data') / 's3dis_1x1',
        batch_size=batch_size
    )

    dm = S3DIS1x1DataModule(config=config)
    dm.prepare_data()
    assert not dm.data_train and not dm.data_test

    expected_data_location = Path(config.data_dir)
    assert expected_data_location.exists()

    dm.setup()
    assert dm.data_train and dm.data_test
    assert dm.train_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch.pos, batch.y
    print(x.shape, y.shape)
    assert len(x) == batch_size
    assert len(y) == batch_size

    # rm created folder
    shutil.rmtree(expected_data_location)