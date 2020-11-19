import numpy as np


def gen_rand_obst_cubes(N, roi_dims=(0.5, 0.5), min_size=0.02, max_size=0.1):
    obst = []
    for i in range(N):
        size = np.random.rand() * (max_size - min_size) + min_size

        loc_x = np.random.rand() * (roi_dims[0] - size)
        loc_y = np.random.rand() * (roi_dims[1] - size)

        obst.append((loc_x, loc_y, size))