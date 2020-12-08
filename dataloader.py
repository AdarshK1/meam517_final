import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import pickle
from pydrake.solvers.mathematicalprogram import SolutionResult


class TrajDataset(Dataset):
    def __init__(self, dir, x_dim=3, with_u=True, u_dim=3, with_x=True, max_u=np.array([25, 25, 10]), keep_only_feasible=True, feasibility_classifier=False):
        self.dir = dir
        self.with_x = with_x

        self.u_max = max_u
        self.x_dim = x_dim
        self.with_u = with_u
        self.u_dim = u_dim

        self.filenames = glob.glob(self.dir + "*")

        self.keep_only_feasible = keep_only_feasible
        self.feasibility_classifier = feasibility_classifier

        if self.keep_only_feasible and not self.feasibility_classifier:
            self.only_feasible()

        print("Num fnames: ", len(self.filenames))

    def only_feasible(self):
        good_fnames = []
        for fname in self.filenames:
            state, output = pickle.load(open(fname, 'rb'))

            if output["result.get_solution_result()"] == SolutionResult.kSolutionFound:
                good_fnames.append(fname)

        self.filenames = good_fnames

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

        if self.feasibility_classifier:
            if output["result.get_solution_result()"] == SolutionResult.kSolutionFound:
                feas = np.array([1])
            else:
                feas = np.array([0])
            return hmap, feas

        x_sol = output["x_sol"]
        u_sol = output["u_sol"]

        u1_sol = u_sol[:, 0] / self.u_max[0]
        u2_sol = u_sol[:, 1] / self.u_max[1]
        u3_sol = u_sol[:, 2] / self.u_max[2]

        if self.with_x and self.with_u:
            concatted_sols = np.concatenate([x_sol, u_sol], axis=1).flatten()
            # print(concatted_sols.shape)
            # print(concatted_sols)
            return hmap, concatted_sols
        elif self.with_x:
            print(x_sol.shape)
            x_sol = x_sol[:, :self.x_dim]
            x_sol /= 3.14
            return hmap, x_sol

        return hmap, u1_sol, u2_sol, u3_sol


if __name__ == "__main__":
    N = 1
    sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
    dset = TrajDataset(sample_fname, x_dim=3, with_u=False, u_dim=3, with_x=True)
    print(len(dset))
    for i in range(N):
        vals = dset.__getitem__(i)
        # print(vals)
