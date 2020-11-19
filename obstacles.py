import numpy as np

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tforms

from pydrake.all import *

class Obstacles:
    def __init__(self, N=1):
        self.xy_offset = [.25, .25]
        self.cubes = self.gen_rand_obst_cubes(N)


    def gen_rand_obst_cubes(self, N, roi_dims=(0.25, 0.25), min_size=0.02, max_size=0.05):
        obst = []
        for i in range(N):
            size = np.random.rand() * (max_size - min_size) + min_size

            loc_x = np.random.rand() * (roi_dims[0] - size) + self.xy_offset[0]
            loc_y = np.random.rand() * (roi_dims[1] - size) + self.xy_offset[1]
            # loc_x = self.xy_offset[0]
            # loc_y = self.xy_offset[1]

            obst.append((loc_x, loc_y, size))
        return obst

    def add_constraints(self, prog, N, x, context, single_leg, plant, plant_context):
        world_frame = single_leg.world_frame()
        # print("world_frame", world_frame)
        base_frame = single_leg.GetFrameByName("toe0")
        # print("base_frame", base_frame)

        # Functor for getting distance to obstacle
        class Obstacle_Distance:
            def __init__(self, obs_xyz):
                # XYZ position of obstacle center
                self.obs_xyz = obs_xyz

            def __call__(self, x):
                # Choose plant and context based on dtype.
                if x.dtype == float:
                    plant_eval = plant
                    context_eval = plant_context
                else:
                    # Assume AutoDiff.
                    plant_eval = single_leg
                    context_eval = context
                # Do forward kinematics.
                single_leg.SetPositionsAndVelocities(context_eval, x)
                toe_xyz = plant_eval.CalcRelativeTransform(context_eval, self.resolve_frame(plant_eval, world_frame), self.resolve_frame(plant_eval, base_frame))
                distance = toe_xyz.translation() - self.obs_xyz
                # print(distance)
                return [distance.dot(distance)]

            def resolve_frame(self, plant, F):
                """Gets a frame from a plant whose scalar type may be different."""
                return plant.GetFrameByName(F.name(), F.model_instance())


        # Add constraints
        for cube in self.cubes:
            obs_xyz = [cube[0], cube[1], cube[2]/2.0]
            radius = np.sqrt(3) * cube[2]
            for i in range(N):
                prog.AddConstraint(
                Obstacle_Distance(obs_xyz),
                lb=[radius], ub=[float('inf')], vars=x[i])


    def draw(self, visualizer):
        i = 0
        for cube in self.cubes:
            loc_x, loc_y, size = cube
            visualizer.vis["cube-" + str(i)].set_object(geom.Box([size, size, size]),
                                    geom.MeshLambertMaterial(color=0xff22dd, reflectivity=0.8))
            visualizer.vis["cube-" + str(i)].set_transform(tforms.translation_matrix([loc_x, loc_y, size/2.0]))
