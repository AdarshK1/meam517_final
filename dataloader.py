import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import pickle
from pydrake.solvers.mathematicalprogram import SolutionResult

# from import_helper import *
# from obstacles import Obstacles
# # from viz_helper import *
# # from constraints import *
from helper import get_plant, resolve_frame, create_context_from_angles


class TrajDataset(Dataset):
    def __init__(self, dir, x_dim=3, with_u=True, u_dim=3, with_x=True, max_u=np.array([25, 25, 10]),
                 keep_only_feasible=True, feasibility_classifier=False, toe_xyz=False, toe_scale=np.array([0.6, 0.3, 0.1])):
        self.dir = dir
        self.with_x = with_x

        self.u_max = max_u
        self.x_dim = x_dim
        self.with_u = with_u
        self.u_dim = u_dim
        self.toe_xyz = toe_xyz
        self.toe_scale = toe_scale

        self.filenames = glob.glob(self.dir + "*")

        self.keep_only_feasible = keep_only_feasible
        self.feasibility_classifier = feasibility_classifier

        if self.keep_only_feasible and not self.feasibility_classifier:
            self.only_feasible()

        print("Num fnames: ", len(self.filenames))

        # builder = DiagramBuilder()
        # self.plant = DiagramBuilder().AddSystem(MultibodyPlant(0.0))
        # file_name = "leg_v2.urdf"
        # Parser(plant=self.plant).AddModelFromFile("leg_v2.urdf")
        # self.plant.Finalize()
        # self.plant_context = self.plant.CreateDefaultContext()


        self.context, self.single_leg, self.plant, self.plant_context = get_plant()

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

        if self.with_x and self.with_u:
            concatted_sols = np.concatenate([x_sol, u_sol], axis=1).flatten()
            return hmap, concatted_sols
        elif self.with_x:
            x_sol = x_sol[:, :self.x_dim]
            x_sol /= 3.14
            x1_sol = x_sol[:, 0]
            x2_sol = x_sol[:, 1]
            x3_sol = x_sol[:, 2]
            return hmap, x1_sol, x2_sol, x3_sol
        elif self.toe_xyz:
            # xx_sol = []
            # y_sol = []
            # z_sol = []
            # # print(x_sol.shape)
            # for i in range(x_sol.shape[0]):
            #     name2ang = {"body_hip": x_sol[i, 0],
            #                 "hip_upper": x_sol[i, 1],
            #                 "upper_lower": x_sol[i, 2]}
            #     context = create_context_from_angles(self.plant, name2ang)
            #
            #     # trans = get_world_position(context, self.single_leg, self.plant, self.plant_context, "toe0", None)
            #     trans = self.plant.CalcRelativeTransform(context,
            #                                              resolve_frame(self.plant, self.plant.world_frame()),
            #                                              resolve_frame(self.plant, self.plant.GetFrameByName("toe0"))).translation()
            #     # print("Trans", trans)
            #
            #     xx_sol.append(trans[0] / self.toe_scale[0])
            #     y_sol.append(trans[1] / self.toe_scale[1])
            #     z_sol.append(trans[2] / self.toe_scale[2])

            xx_sol = output["toe_x_sol"] / self.toe_scale[0]
            y_sol = output["toe_y_sol"] / self.toe_scale[1]
            z_sol = output["toe_z_sol"] / self.toe_scale[2]
            # print(xx_sol)
            # print(y_sol)
            # print(z_sol)
            return hmap, xx_sol, y_sol, z_sol

        u1_sol = u_sol[:, 0] / self.u_max[0]
        u2_sol = u_sol[:, 1] / self.u_max[1]
        u3_sol = u_sol[:, 2] / self.u_max[2]

        return hmap, u1_sol, u2_sol, u3_sol


if __name__ == "__main__":
    N = 1
    sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
    # dset = TrajDataset(sample_fname, x_dim=3, with_u=False, u_dim=3, with_x=True)
    dset = TrajDataset(sample_fname, x_dim=3, with_u=False, u_dim=3, with_x=False, toe_xyz=True)
    print(len(dset))
    for i in range(N):
        vals = dset.__getitem__(i)
        # print(vals)
