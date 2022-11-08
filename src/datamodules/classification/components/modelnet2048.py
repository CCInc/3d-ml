import glob
import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch


class ModelNet2048Dataset(Dataset):
    def __init__(self, data_dir, split):
        self.data = self.load_data(data_dir, split)

    def load_data(self, data_dir, partition):
        all_data = []
        for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
            with h5py.File(h5_name, 'r') as f:
                xyzs = f['data'][:].astype('float32')
                labels = f['label'][:].astype('int64')
                # print(xyz.shape, label.shape)
                for xyz, label in zip(xyzs, labels):
                    all_data.append(Data(pos=torch.from_numpy(xyz), y=torch.from_numpy(label)))

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d
