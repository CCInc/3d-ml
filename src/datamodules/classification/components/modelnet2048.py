import glob
import os
from typing import Optional

import h5py
import torch
from torch_geometric.data import Data, Dataset

from src.transforms import BaseTransform


class ModelNet2048Dataset(Dataset):
    def __init__(self, data_dir, split, transform: Optional[BaseTransform] = None):
        super().__init__(transform=transform)
        self.data = self.load_data(data_dir, split)

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
        # Since we inherit from pytorch geometric's dataset, the transform will automatically get
        # applied
        d = self.data[idx]
        return d
