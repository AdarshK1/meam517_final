import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import pickle


class TrajDataset(Dataset):
    def __init__(self, dir, with_x=True, max_u=np.array([25, 25, 10])):
        self.dir = dir
        self.with_x = with_x

        self.u_max = max_u

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
        # print(hmap)
        # hmap /= np.max(hmap)

        x_sol = output["x_sol"]
        u_sol = output["u_sol"]
        # print(hmap.shape)

        # print("u max", np.max(u_sol, axis=0))
        # print(u_sol.shape)
        # u_sol /= self.u_max
        # print("post dib u max", np.max(u_sol, axis=0))
        u1_sol = u_sol[:, 0] / self.u_max[0]
        u2_sol = u_sol[:, 1] / self.u_max[1]
        u3_sol = u_sol[:, 2] / self.u_max[2]
        # u_sol[abs(u_sol) < 0.05] = 0.0
        # x_sol /= np.max(x_sol)

        if self.with_x:
            concatted_sols = np.concatenate([x_sol, u_sol], axis=1).flatten()
            # print(concatted_sols.shape)
            # print(concatted_sols)
            return hmap, concatted_sols

        return hmap, u1_sol, u2_sol, u3_sol


if __name__ == "__main__":
    N = 1
    sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
    dset = TrajDataset(sample_fname)
    print(len(dset))
    for i in range(N):
        vals = dset.__getitem__(i)
        # print(vals)
