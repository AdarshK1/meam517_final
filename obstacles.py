import numpy as np

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tforms
import cv2
from scipy.spatial.transform import Rotation as R
from pydrake.all import *
from helper import *


class Obstacles:
    def __init__(self, N=5, multi_constraint=False):
        self.xy_offset = [.1, 0.0]
        self.roi_dims = np.array([0.5, 0.5])
        self.multi_constraint = multi_constraint

        self.voxels_per_meter = 25  # 2cm resolution
        self.heightmap = np.zeros(np.round(self.roi_dims * self.voxels_per_meter).astype(np.int))

        # if using random
        # self.cubes = self.gen_rand_obst_cubes(N, 0.02, 0.1)

        # if using known
        self.cubes = self.get_known_cubes()
        self.heightmap_from_known_obst()

        self.heightmap = cv2.rotate(self.heightmap, cv2.ROTATE_180)
        self.heightmap = cv2.flip(self.heightmap, 1)  # flip around y axis

        self.hmap_obst = []
        self.discretize_heightmap()
        # self.visualize_heightmap()

    def gen_rand_obst_cubes(self, N, min_size=0.02, max_size=0.05, min_height=0, max_height=0.1):
        """

        :param N:
        :param min_size:
        :param max_size:
        :param min_height:
        :param max_height:
        :return:
        """
        obst = []
        for i in range(N):
            size = np.random.rand() * (max_size - min_size) + min_size
            height = np.random.rand() * (max_height - min_height) + min_height

            loc_x = np.random.rand() * (self.roi_dims[0] - size) + size / 2
            loc_y = np.random.rand() * (self.roi_dims[1] - size) + size / 2

            hmap_x = int(loc_x * self.voxels_per_meter)  # + self.roi_dims[0] * self.voxels_per_meter / 2)
            hmap_y = int(loc_y * self.voxels_per_meter)  # + self.roi_dims[1] * self.voxels_per_meter / 2)
            half_size_hmap = int(size * self.voxels_per_meter / 2)
            print("loc_x", loc_x, "loc_y", loc_y, "hmap_x", hmap_x, "hmap_y", hmap_y, "half_size_hmap", half_size_hmap)

            for r in range(hmap_x - half_size_hmap, hmap_x + half_size_hmap):
                for c in range(hmap_y - half_size_hmap, hmap_y + half_size_hmap):
                    if size > self.heightmap[r, c]:
                        self.heightmap[r, c] = height

            loc_x += self.xy_offset[0]
            loc_y += self.xy_offset[1]

            obst.append((loc_x, loc_y, size, height))
        return obst

    def heightmap_from_known_obst(self):
        """

        """
        for i, cube in enumerate(self.cubes):
            loc_x, loc_y, size, height = cube

            hmap_x = int((loc_x - self.xy_offset[0]) * self.voxels_per_meter)
            hmap_y = int((loc_y - self.xy_offset[1]) * self.voxels_per_meter)
            half_size_hmap = int(size * self.voxels_per_meter / 2)

            print("loc_x", loc_x, "loc_y", loc_y, "hmap_x", hmap_x, "hmap_y", hmap_y, "half_size_hmap", half_size_hmap)

            # have to do this manually cuz min/max calls on pydrake expressions are a pain
            min_x = hmap_x - half_size_hmap if hmap_x - half_size_hmap > 0 else 0
            min_y = hmap_y - half_size_hmap if hmap_y - half_size_hmap > 0 else 0

            max_x = hmap_x + half_size_hmap if hmap_x + half_size_hmap < self.heightmap.shape[0] else \
            self.heightmap.shape[0]

            max_y = hmap_y + half_size_hmap if hmap_y + half_size_hmap < self.heightmap.shape[1] else \
            self.heightmap.shape[1]

            for r in range(min_x, max_x):
                for c in range(min_y, max_y):
                    if size > self.heightmap[r, c]:
                        self.heightmap[r, c] = height

    def discretize_heightmap(self, full_column=True):
        """

        :param full_column:
        """
        for i in range(self.heightmap.shape[0]):
            for j in range(self.heightmap.shape[1]):
                z_val = self.heightmap[i, j]
                x = i / self.voxels_per_meter
                y = j / self.voxels_per_meter
                size = 1.0 / self.voxels_per_meter
                # print("z_val", z_val, "x", x, "y", y, "size", size)

                if full_column:
                    for k in range(int(round(z_val * self.voxels_per_meter))):
                        height = k / self.voxels_per_meter
                        self.hmap_obst.append([x, y, size, height])

                elif z_val > 0:
                    self.hmap_obst.append([x, y, size, z_val])

        print("Num obst from hmap:", len(self.hmap_obst))

    def visualize_heightmap(self):
        """
        Visualize a normalized, resized heightmap for debugging.
        """
        disp_image = cv2.normalize(self.heightmap, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        disp_image = cv2.resize(disp_image, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.imshow("viz heightmap", disp_image)
        cv2.waitKey(1000)

    def get_known_cubes(self):
        """

        :return:
        """
        return [(0.25, 0.1, 0.2, 0.2), (0.25, 0.3, 0.1, 0.1)]
        # return [(0.25, 0.1, 0.2, 0.2), (0.25, 0.4, 0.1, 0.1)]
        # return [(0.25, 0.25, 0.2, 0.1)]

    def add_constraints(self, prog, N, x, context, single_leg, plant, plant_context, hmap_constraints=True):
        """

        :param prog:
        :param N:
        :param x:
        :param context:
        :param single_leg:
        :param plant:
        :param plant_context:
        :param hmap_constraints:
        :return:
        """
        world_frame = single_leg.world_frame()

        frame_names = ["toe0", "lower0", "upper0"]
        second_frame_names = [None, "toe0", "lower0"]
        frame_radii = {"toe0": 0.02, "lower0": 0.02, "upper0": 0.05}

        frames = [single_leg.GetFrameByName(name) for name in frame_names]

        # Functor for getting distance to obstacle
        class Obstacle_Distance:
            def __init__(self, obs_xyz, frame, multi_constraint=False, second_frame=None):
                # XYZ position of obstacle center
                self.obs_xyz = obs_xyz
                self.multi_constraint = multi_constraint
                self.frame = frame
                self.second_frame = second_frame

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
                link_xyz = plant_eval.CalcRelativeTransform(context_eval, self.resolve_frame(plant_eval, world_frame),
                                                            self.resolve_frame(plant_eval, self.frame))
                if self.second_frame is None:
                    distance = link_xyz.translation() - self.obs_xyz
                    # print("joint constraint: ", self.frame.name(), distance.dot(distance) ** 0.5, link_xyz.translation(), self.obs_xyz, distance)
                    return [distance.dot(distance) ** 0.5]

                second_link_xyz = plant_eval.CalcRelativeTransform(context_eval,
                                                                   self.resolve_frame(plant_eval, world_frame),
                                                                   self.resolve_frame(plant_eval, self.second_frame))

                # actually, this makes more sense as a dot product. project the link->obst vector onto the link vector
                # then subtract the link->obst projection from the obst vector, and that gives you distance from link
                # to obstacle

                # vector representing link
                # print("second_link_xyz", second_link_xyz.translation())
                # print("link_xyz", link_xyz.translation())
                # print(self.obs_xyz)

                link_vect = second_link_xyz.translation() - link_xyz.translation()
                # norm of link vector
                link_vect_norm = np.linalg.norm(link_vect)
                # link unit vector
                link_unit_vect = link_vect / link_vect_norm

                # vector going from link to obstacle
                link_to_obst_vect = self.obs_xyz - link_xyz.translation()

                obst_dist = link_to_obst_vect - (link_unit_vect * link_to_obst_vect.dot(link_unit_vect))
                # print("link_vect", link_vect)
                # print("link_to_obst_vect", link_to_obst_vect)
                # print("link_to_obst_vect.dot(link_unit_vect)", link_to_obst_vect.dot(link_unit_vect))
                # print("projection", (link_vect * (link_unit_vect * link_to_obst_vect.dot(link_unit_vect))))
                # print("obst_dist", obst_dist)
                distance = np.linalg.norm(obst_dist)

                # print("link constraint", self.frame.name(), self.second_frame.name(), distance)
                # print()
                return [distance]

            def resolve_frame(self, plant, F):
                """Gets a frame from a plant whose scalar type may be different."""
                return plant.GetFrameByName(F.name(), F.model_instance())

        def add_one_obstacle_constraints(radius, obs_xyz):
            for i in range(N):
                for j, frame in enumerate(frames):
                    distance_functor = Obstacle_Distance(obs_xyz, frame, multi_constraint=self.multi_constraint)

                    prog.AddConstraint(distance_functor,
                                       lb=[radius + frame_radii[frame.name()]], ub=[float('inf')], vars=x[i])

                    if second_frame_names[j] is not None:
                        distance_functor = Obstacle_Distance(obs_xyz, frame, multi_constraint=self.multi_constraint,
                                                             second_frame=single_leg.GetFrameByName(
                                                                 second_frame_names[j]))
                        prog.AddConstraint(distance_functor,
                                           lb=[radius + frame_radii[frame.name()]], ub=[float('inf')], vars=x[i])

        # Add constraints
        if not hmap_constraints:
            for cube in self.cubes:
                radius = np.sqrt(3) * cube[2] / 2
                obs_xyz = [cube[0], cube[1], radius]

                add_one_obstacle_constraints(radius, obs_xyz)

        elif hmap_constraints:
            for obst in self.hmap_obst:
                radius = np.sqrt(3) * obst[2] / 2
                obs_xyz = [obst[0], obst[1], obst[3]]

                add_one_obstacle_constraints(radius, obs_xyz)

    def draw(self, visualizer, hmap_obst=True):
        """

        :param hmap_obst:
        :param visualizer:
        """
        if not hmap_obst:
            for i, cube in enumerate(self.cubes):
                loc_x, loc_y, size, height = cube
                radius = np.sqrt(3) * size / 2

                visualizer.vis["sphere-" + str(i)].set_object(geom.Sphere(radius),
                                                              geom.MeshLambertMaterial(color=0xff22dd,
                                                                                       reflectivity=0.8))
                visualizer.vis["sphere-" + str(i)].set_transform(tforms.translation_matrix([loc_x, loc_y, radius]))

        elif hmap_obst:
            for i, sph in enumerate(self.hmap_obst):
                x, y, size, z = sph
                radius = np.sqrt(3) * size / 2
                visualizer.vis["sphere-" + str(i)].set_object(geom.Sphere(radius),
                                                              geom.MeshLambertMaterial(color=0xffdd22,
                                                                                       reflectivity=0.8))
                visualizer.vis["sphere-" + str(i)].set_transform(tforms.translation_matrix([x, y, z]))
