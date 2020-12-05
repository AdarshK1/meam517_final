import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import pickle


class TrajDataset(Dataset):
    def __init__(self, dir, with_x=True):
        self.dir = dir
        self.with_x = with_x

        self.filenames = glob.glob(self.dir + "*")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        state, output = pickle.load(open(fname, 'rb'))

        # print(state, output)
        # print(state.keys())
        # print(output.keys())
        # print(output)

        hmap = state["obstacles"].heightmap
        hmap = hmap[np.newaxis, :, :]
        x_sol = output["x_sol"]
        u_sol = output["u_sol"]
        # print(hmap.shape)

        if self.with_x:
            concatted_sols = np.concatenate([x_sol, u_sol], axis=1).flatten()
            # print(concatted_sols.shape)
            return hmap, concatted_sols

        return hmap, u_sol.flatten()


if __name__ == "__main__":
    N = 1
    sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
    dset = TrajDataset(sample_fname)
    print(len(dset))
    for i in range(N):
        vals = dset.__getitem__(i)
        # print(vals)
