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
from helper import get_plant, get_limits

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tforms

from obstacles import Obstacles


def find_step_trajectory(N, initial_state, final_state, apex_state, tf, obstacles=None):
    '''
    Parameters:
      N - number of knot points
      initial_state - starting configuration
      distance - target distance to throw the ball
    '''

    context, single_leg, plant, plant_context = get_plant()

    # Dimensions specific to the single_leg
    n_x = single_leg.num_positions() + single_leg.num_velocities()
    n_u = single_leg.num_actuators()

    # Store the actuator limits here
    effort_limits, vel_limits = get_limits(n_u, n_x, single_leg)

    # Create the mathematical program
    prog = MathematicalProgram()
    x = np.zeros((N, n_x), dtype="object")
    u = np.zeros((N, n_u), dtype="object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
        u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

    t0 = 0.0
    timesteps = np.linspace(t0, tf, N)

    # Add obstacle constraints
    if obstacles is not None:
        obstacles.add_constraints(prog, N, x, context, single_leg, plant, plant_context)

    # Add the kinematic constraints (initial state, final state)

    # Add constraints on the initial state
    A_init = np.identity(n_x)
    b_init = np.array(initial_state)
    prog.AddLinearEqualityConstraint(A_init, b_init, x[0].flatten())

    # Add constraints on the final state
    A_end = np.identity(n_x)
    b_end = np.array(final_state)
    prog.AddLinearEqualityConstraint(A_end, b_end, x[-1].flatten())

    if N > 2:
        A_apex = np.identity(n_x)
        b_apex = np.array(apex_state)
        prog.AddLinearEqualityConstraint(A_apex, b_apex, x[N // 2].flatten())

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, single_leg, context, N, x, u, timesteps)

    Q = np.eye(n_u * N)

    # multiplying the cost on abduction doesn't actually solve the crazy abdction problem, it makes it worse because
    # it now refuses to fight gravity!
    for i in range(N):
        Q[n_u * i] *= 0

    b = np.zeros([n_u * N, 1])
    # getting rid of cost on control for now, this is making it not fight gravity!
    # prog.AddQuadraticCost(Q, b, u.flatten())
    # print("Added quadcost")

    # TODO: Add bounding box constraints on the inputs and qdot

    ub = np.zeros([N * n_u])
    for i in range(N):
        ub[i * n_u] = effort_limits[0]
        ub[i * n_u + 1] = effort_limits[1]
        ub[i * n_u + 2] = effort_limits[2]

    lb = -ub
    prog.AddBoundingBoxConstraint(lb, ub, u.flatten())
    # print("Added effort bbox")

    ub = np.zeros([N * n_x])
    lb = np.zeros([N * n_x])

    for i in range(N):
        ub[i * n_x] = 0.785
        lb[i * n_x] = -0.785
        ub[i * n_x + 1] = 0
        lb[i * n_x + 1] = -3.14

        ub[i * n_x + 2] = 3.14
        lb[i * n_x + 2] = 0.25

        ub[i * n_x + 3] = vel_limits[0]
        lb[i * n_x + 3] = -vel_limits[0]

        ub[i * n_x + 4] = vel_limits[1]
        lb[i * n_x + 4] = -vel_limits[1]

        ub[i * n_x + 5] = vel_limits[2]
        lb[i * n_x + 5] = -vel_limits[2]

    prog.AddBoundingBoxConstraint(lb, ub, x.flatten())

    print("\n", "-" * 50)
    for i in range(N):
        # u_init = np.random.rand(n_u) * effort_limits * 2 - effort_limits
        u_init = np.zeros(n_u)
        prog.SetInitialGuess(u[i], u_init)

        if N < 3:
            x_init = initial_state + (i / N) * (final_state - initial_state)
            prog.SetInitialGuess(x[i], x_init)
            prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])

        elif N > 3 and i < N / 2:
            x_init = initial_state + (i / (N / 2) ) * (apex_state - initial_state)
            print(i, x[i].flatten(), x_init)
            prog.SetInitialGuess(x[i], x_init)
            prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])

        else:
            x_init = apex_state + ((i - N / 2) / (N / 2) ) * (final_state - apex_state)
            print(i, x[i].flatten(), x_init)
            prog.SetInitialGuess(x[i], x_init)
            prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])

    print("\n", "-" * 50)
    print("Costs")
    print("generic_costs", prog.generic_costs())
    print("linear_costs", prog.linear_costs())
    print("quadratic_costs", prog.quadratic_costs())
    print("-" * 50, "\n")

    # print("initial guesses")

    # Set up solver
    solver = SnoptSolver()
    result = solver.Solve(prog)
    # print("solved")

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)

    print("-" * 25)
    print(result.get_solution_result())
    print("-" * 25)

    # Reconstruct the trajecotry as a cubic hermite spline
    xdot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])

    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.FirstOrderHold(timesteps, u_sol.T)

    return x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swing a leg.')
    parser.add_argument('--use_viz', action='store_true')
    parser.add_argument('--obstacles', action='store_true')

    args = parser.parse_args()

    N = 5
    # nominal stance
    # initial_state = np.array([0, -2.0, 2.0, 0, 0, 0])

    # end of stance
    initial_state = np.array([0, -2.0, 2.0, 0, 0, 0])

    # apex
    apex_state = np.array([0, -2.75, 1.0, 0, 0, 0])

    # end of step
    # initial_state = np.array([0, -2.0, 1.5, 0, 0, 0])
    final_state = np.array([0, -2.0, 1.5, 0, 0, 0])

    # Initialize obstacles
    obstacles = None
    if args.obstacles:
        obstacles = Obstacles()

    # final_state = initial_state
    tf = 5.0
    x_traj, u_traj, prog, x_guess, u_guess = find_step_trajectory(N, initial_state, final_state, apex_state, tf, obstacles)

    # Create a MultibodyPlant for the arm
    file_name = "leg_v2.urdf"
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    single_leg = builder.AddSystem(MultibodyPlant(0.0))
    single_leg.RegisterAsSourceForSceneGraph(scene_graph)
    Parser(plant=single_leg).AddModelFromFile(file_name)
    single_leg.Finalize()
    # print("finalized leg")


    if args.use_viz:
        server_args = ['--ngrok_http_tunnel']

        zmq_url = "tcp://127.0.0.1:6000"
        web_url = "http://127.0.0.1:7000/static/"
        # Create meshcat
        visualizer = ConnectMeshcatVisualizer(
            builder,
            scene_graph,
            scene_graph.get_pose_bundle_output_port(),
            zmq_url=zmq_url,
            server_args=server_args)

        x_traj_source = builder.AddSystem(TrajectorySource(x_traj))
        u_traj_source = builder.AddSystem(TrajectorySource(u_traj))

        demux = builder.AddSystem(Demultiplexer(np.array([3, 3])))
        to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(single_leg))
        zero_inputs = builder.AddSystem(ConstantVectorSource(np.zeros(3)))


        builder.Connect(zero_inputs.get_output_port(), single_leg.get_actuation_input_port())
        builder.Connect(x_traj_source.get_output_port(), demux.get_input_port())
        builder.Connect(demux.get_output_port(0), to_pose.get_input_port())
        builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(single_leg.get_source_id()))
        builder.Connect(scene_graph.get_query_output_port(), single_leg.get_geometry_query_input_port())

        ConnectDrakeVisualizer(builder, scene_graph)

        diagram = builder.Build()
        diagram.set_name("diagram")

        # Visualize obstacles
        if args.obstacles:
            obstacles.draw(visualizer)

        visualizer.load()
        print("\n!!!Open the visualizer by clicking on the URL above!!!")

        # Visualize the motion for `n_playback` times
        n_playback = 1
        for i in range(n_playback):
            # Set up a simulator to run this diagram.
            simulator = Simulator(diagram)
            simulator.Initialize()
            time.sleep(1)
            simulator.set_target_realtime_rate(0.5)
            simulator.AdvanceTo(tf)
            time.sleep(2)
