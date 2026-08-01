from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn

class spiceData(Dataset):
    def __init__(self, x, batch_size=32, train=True):
        self.x = x
        self.temp_x = None
        self.y = None
        self.batch_size = batch_size
        self.train = train

        self.on_epoch_end()

    def __len__(self):
        return len(self.x) // self.batch_size

    def on_epoch_end(self):
        if self.y is None or self.train:
            np.random.shuffle(self.x)
            self.temp_x = self.x.copy()
            self.y = np.zeros_like(self.x, dtype=bool)

            axis0_counts = self.x.sum(axis=1)
            axis0_bin_min_index = np.cumsum(self.x.sum(axis=1)) - axis0_counts
            index_use = axis0_bin_min_index + np.random.randint(low=0, high=axis0_counts)
            axis0, axis1 = np.where(self.x)

            self.temp_x[range(len(self.x)), axis1[index_use]] = False
            self.y[range(len(self.x)), axis1[index_use]] = True

    def get_batch(self, idx):
        if idx >= self.__len__():
            raise IndexError()

        idx1 = idx * self.batch_size
        idx2 = (idx + 1) * self.batch_size
        return self.temp_x[idx1:idx2], self.y[idx1:idx2]

    def __getitem__(self, idx):
        x, y = self.get_batch(idx)

        return torch.from_numpy(x).to(torch.float32), torch.from_numpy(y).to(torch.float32)


class spiceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(789, 64)
        self.linear2 = torch.nn.Linear(64, 789)


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x