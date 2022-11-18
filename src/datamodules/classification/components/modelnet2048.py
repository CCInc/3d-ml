import glob
import os
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.transforms import BaseTransform


class ModelNet2048Dataset(Dataset):
    def __init__(self, data_dir, split, transform: Optional[BaseTransform] = None):
        self.data = self.load_data(data_dir, split)
        self.transform = transform

    def load_data(self, data_dir, partition):
        all_data = []
        for h5_name in glob.glob(
            os.path.join(data_dir, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition)
        ):
            with h5py.File(h5_name, "r") as f:
                xyzs = f["data"][:].astype("float32")
                labels = f["label"][:].astype("int64")
                for xyz, label in zip(xyzs, labels):
                    all_data.append(Data(pos=torch.from_numpy(xyz), y=int(label[0])))

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        if self.transform:
            d = self.transform(d)

        return d
