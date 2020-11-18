import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator

from pydrake.all import (
    DiagramBuilder, Simulator
)

from pydrake.multibody.tree import (
    JointActuatorIndex
)

from pydrake.geometry import SceneGraph
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.trajectories import PiecewisePolynomial
from pydrake.solvers.snopt import SnoptSolver

import constraints

importlib.reload(constraints)

from constraints import (
    AddCollocationConstraints,
    EvaluateDynamics
)

import time
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_system_graphviz

from pydrake.all import (
    ConnectMeshcatVisualizer, DiagramBuilder,
    Simulator
)

from pydrake.geometry import (
    SceneGraph, ConnectDrakeVisualizer
)

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import (
    TrajectorySource,
    Demultiplexer,
    ConstantVectorSource
)

import argparse


def get_plant():
    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(0.0))
    file_name = "leg_v2.urdf"
    Parser(plant=plant).AddModelFromFile(file_name)
    plant.Finalize()
    single_leg = plant.ToAutoDiffXd()

    plant_context = plant.CreateDefaultContext()
    context = single_leg.CreateDefaultContext()

    return context, single_leg, plant, plant_context


def get_limits(n_u, n_x, single_leg):
    # Store the actuator limits here
    effort_limits = np.zeros(n_u)
    for act_idx in range(n_u):
        effort_limits[act_idx] = \
            single_leg.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
    vel_limits = 15 * np.ones(n_x // 2)

    return effort_limits, vel_limits