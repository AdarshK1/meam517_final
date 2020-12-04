import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import copy
import time
import os
from import_helper import *
import pickle


class TrajDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir

        self.filenames = glob.glob(self.dir + "*")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        state, output = pickle.load(open(fname, 'rb'))

        # print(state, output)
        print(state.keys())
        # print(output)

        return state, output


if __name__ == "__main__":
    N = 1
    sample_fname = "/home/adarsh/software/meam517_final/data/"
    dset = TrajDataset(sample_fname)
    print(len(dset))
    for i in range(N):
        vals = dset.__getitem__(i)
        # print(vals)
