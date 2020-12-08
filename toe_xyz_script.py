import pickle
import glob
import numpy as np
from helper import get_plant, resolve_frame, create_context_from_angles

sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
output_fname = "/home/adarsh/software/meam517_final/data_v3/"

context, single_leg, plant, plant_context = get_plant()

filenames = glob.glob(sample_fname + "*")

for idx, fname in enumerate(filenames):
    open_f = open(fname, 'rb')
    state, output = pickle.load(open_f)
    open_f.close()

    x_sol = output["x_sol"]
    u_sol = output["u_sol"]

    xx_sol = []
    y_sol = []
    z_sol = []
    # print(x_sol.shape)
    for i in range(x_sol.shape[0]):
        name2ang = {"body_hip": x_sol[i, 0],
                    "hip_upper": x_sol[i, 1],
                    "upper_lower": x_sol[i, 2]}
        context = create_context_from_angles(plant, name2ang)

        # trans = get_world_position(context, self.single_leg, self.plant, self.plant_context, "toe0", None)
        trans = plant.CalcRelativeTransform(context,
                                            resolve_frame(plant, plant.world_frame()),
                                            resolve_frame(plant, plant.GetFrameByName("toe0"))).translation()
        # print("Trans", trans)

        xx_sol.append(trans[0])  # / toe_scale[0])
        y_sol.append(trans[1])  # / toe_scale[1])
        z_sol.append(trans[2])  # / toe_scale[2])

    xx_sol = np.array(xx_sol)
    y_sol = np.array(y_sol)
    z_sol = np.array(z_sol)

    output["toe_x_sol"] = xx_sol
    output["toe_y_sol"] = y_sol
    output["toe_z_sol"] = z_sol

    print(idx / len(filenames))
    out_fname = output_fname + fname.split("/")[-1]

    open_f = open(out_fname, 'wb')
    pickle.dump((state, output), open_f)
    open_f.close()
