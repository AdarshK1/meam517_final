import numpy as np

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tforms
import cv2

from pydrake.all import *


class Obstacles:
    def __init__(self, N=5, multi_constraint=False):
        self.xy_offset = [.15, 0.1]
        self.roi_dims = np.array([0.25, 0.25])
        self.multi_constraint = multi_constraint

        self.voxels_per_meter = 200  # 0.5cm resolution
        self.heightmap = np.zeros(np.round(self.roi_dims * self.voxels_per_meter).astype(np.int))

        self.cubes = self.gen_rand_obst_cubes(N, 0.02, 0.1)
        # self.cubes = self.get_known_cubes()

        self.heightmap = cv2.rotate(self.heightmap, cv2.ROTATE_180)

        self.visualize_heightmap()

    def gen_rand_obst_cubes(self, N, min_size=0.02, max_size=0.05):
        obst = []
        for i in range(N):
            size = np.random.rand() * (max_size - min_size) + min_size

            loc_x = np.random.rand() * (self.roi_dims[0] - size) + size / 2
            loc_y = np.random.rand() * (self.roi_dims[1] - size) + size / 2

            hmap_x = int(loc_x * self.voxels_per_meter)  # + self.roi_dims[0] * self.voxels_per_meter / 2)
            hmap_y = int(loc_y * self.voxels_per_meter)  # + self.roi_dims[1] * self.voxels_per_meter / 2)
            half_size_hmap = int(size * self.voxels_per_meter / 2)
            print("loc_x", loc_x, "loc_y", loc_y, "hmap_x", hmap_x, "hmap_y", hmap_y, "half_size_hmap", half_size_hmap)

            for r in range(hmap_x - half_size_hmap, hmap_x + half_size_hmap):
                for c in range(hmap_y - half_size_hmap, hmap_y + half_size_hmap):
                    if size > self.heightmap[r, c]:
                        self.heightmap[r, c] = size

            loc_x += self.xy_offset[0]
            loc_y += self.xy_offset[1]

            obst.append((loc_x, loc_y, size))
        return obst

    def visualize_heightmap(self):
        disp_image = cv2.normalize(self.heightmap, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        disp_image = cv2.resize(disp_image, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.imshow("test", disp_image)
        cv2.waitKey(1000)

    def get_known_cubes(self):
        return [(0.25, 0.1, 0.2), (0.25, 0.4, 0.2)]

    def add_constraints(self, prog, N, x, context, single_leg, plant, plant_context):
        world_frame = single_leg.world_frame()
        base_frame = single_leg.GetFrameByName("toe0")

        frame_names = ["toe0", "lower0", "upper0"]
        frames = []
        for name in frame_names:
            frames.append(single_leg.GetFrameByName(name))

        # print("base_frame", base_frame)

        # Functor for getting distance to obstacle
        class Obstacle_Distance:
            def __init__(self, obs_xyz, frame, multi_constraint=False):
                # XYZ position of obstacle center
                self.obs_xyz = obs_xyz
                self.multi_constraint = multi_constraint
                self.frame = frame

            def __call__(self, x):
                # Choose plant and context based on dtype.
                if x.dtype == float:
                    # print("using float")
                    plant_eval = plant
                    context_eval = plant_context
                else:
                    # print("using auto")
                    # Assume AutoDiff.
                    plant_eval = single_leg
                    context_eval = context

                plant_eval.SetPositionsAndVelocities(context_eval, x)

                # Do forward kinematics.
                toe_xyz = plant_eval.CalcRelativeTransform(context_eval, self.resolve_frame(plant_eval, world_frame),
                                                           self.resolve_frame(plant_eval, self.frame))
                distance = toe_xyz.translation() - self.obs_xyz
                return [distance.dot(distance) ** 0.5]

            def resolve_frame(self, plant, F):
                """Gets a frame from a plant whose scalar type may be different."""
                return plant.GetFrameByName(F.name(), F.model_instance())

        # Add constraints
        for cube in self.cubes:
            obs_xyz = [cube[0], cube[1], cube[2] / 2.0]
            radius = np.sqrt(3) * cube[2] / 2
            for i in range(N):
                for frame in frames:
                    distance_functor = Obstacle_Distance(obs_xyz, frame, multi_constraint=self.multi_constraint)
                    # print(x[i])
                    prog.AddConstraint(
                        distance_functor,
                        lb=[radius], ub=[float('inf')], vars=x[i])

    def draw(self, visualizer):
        for i, cube in enumerate(self.cubes):
            loc_x, loc_y, size = cube
            visualizer.vis["cube-" + str(i)].set_object(geom.Box([size, size, size]),
                                                        geom.MeshLambertMaterial(color=0xff22dd, reflectivity=0.8))
            visualizer.vis["cube-" + str(i)].set_transform(tforms.translation_matrix([loc_x, loc_y, size / 2.0]))

