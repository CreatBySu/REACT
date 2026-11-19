import os

import numpy as np
import torch


class DataLoader:
    def __init__(self, data, idx, seq_len, horizon, batch_size):
        self.data = data
        self.idx = idx
        self.batch_size = batch_size
        self.num_batch = len(idx) // batch_size
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, horizon + 1, 1)

    def shuffle(self):
        self.idx = self.idx[np.random.permutation(len(self.idx))]

    def __iter__(self):
        for i in range(self.num_batch):
            batch = self.idx[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.data[batch[:, None] + self.x_offsets]
            y = self.data[batch[:, None] + self.y_offsets][..., :1]
            yield x, y


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_year(root, year, input_dim):
    ptr = np.load(os.path.join(root, str(year), "his.npz"))
    return ptr["data"][..., :input_dim].astype(np.float32), StandardScaler(ptr["mean"], ptr["std"])


def load_index(root, year, split):
    return np.load(os.path.join(root, str(year), "idx_{}.npy".format(split)))
