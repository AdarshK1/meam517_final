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
    AddFinalLandingPositionConstraint,
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


def find_throwing_trajectory(N, initial_state, distance, tf):
    '''
    Parameters:
      N - number of knot points
      initial_state - starting configuration
      distance - target distance to throw the ball

    '''

    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(0.0))
    file_name = "leg_v2.urdf"
    Parser(plant=plant).AddModelFromFile(file_name)
    plant.Finalize()
    single_leg = plant.ToAutoDiffXd()

    plant_context = plant.CreateDefaultContext()
    context = single_leg.CreateDefaultContext()

    # Dimensions specific to the single_leg
    n_x = single_leg.num_positions() + single_leg.num_velocities()
    n_u = single_leg.num_actuators()
    print("num_positions", single_leg.num_positions())
    print("num_velocities", single_leg.num_velocities())
    # print(single_leg.())
    print("n_x", n_x)
    print("n_u", n_u)

    # Store the actuator limits here
    effort_limits = np.zeros(n_u)
    for act_idx in range(n_u):
        effort_limits[act_idx] = \
            single_leg.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
    vel_limits = 15 * np.ones(n_x // 2)

    # Create the mathematical program
    prog = MathematicalProgram()
    x = np.zeros((N, n_x), dtype="object")
    u = np.zeros((N, n_u), dtype="object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
        u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

    t_land = prog.NewContinuousVariables(1, "t_land")

    t0 = 0.0
    timesteps = np.linspace(t0, tf, N)
    x0 = x[0]
    xf = x[-1]

    # Add the kinematic constraints (initial state, final state)
    # TODO: Add constraints on the initial state
    A_init = np.identity(n_x)
    b_init = np.array(initial_state)
    prog.AddLinearEqualityConstraint(A_init, b_init, x[0].flatten())

    # Add the kinematic constraint on the final state
    # AddFinalLandingPositionConstraint(prog, xf, distance, t_land)

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, single_leg, context, N, x, u, timesteps)
    print("Added collocation")

    # TODO: Add the cost function here
    Q = np.zeros([n_u * N, n_u * N])
    dt = timesteps[1]
    for i in range(1, N - 1):
        for j in range(n_u):
            Q[n_u * i + j] += dt * 2
            Q[n_u * (i + 1) + j] += dt * 2

    b = np.zeros([n_u * N, 1])
    prog.AddQuadraticCost(Q, b, u.flatten())
    print("Added quadcost")

    # TODO: Add bounding box constraints on the inputs and qdot
    ub = np.zeros([N * n_u])
    for i in range(N):
        ub[i * n_u] = effort_limits[0]
        ub[i * n_u + 1] = effort_limits[1]
        ub[i * n_u + 2] = effort_limits[2]

    lb = -ub
    prog.AddBoundingBoxConstraint(lb, ub, u.flatten())
    print("Added effort bbox")

    ub = np.zeros([N * n_x])

    for i in range(N):
        ub[i * n_x] = 3.14
        ub[i * n_x + 1] = 3.14
        ub[i * n_x + 2] = 3.14

        ub[i * n_x + 3] = vel_limits[0]
        ub[i * n_x + 4] = vel_limits[1]
        ub[i * n_x + 5] = vel_limits[2]

    lb = -ub
    prog.AddBoundingBoxConstraint(lb, ub, x.flatten())
    print("Added vel bbox")

    # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
    for i in range(N):

        u_init = np.random.rand(n_u) * effort_limits * 2 - effort_limits

        x0_interp = np.array([0, 0, 0, 0, 0, 0])
        xf_interp = np.array([np.pi, 0, 0, 0, 0, 0])
        x_init = x0_interp + (i / N) * (xf_interp - x0_interp)

        # print("u_init", u_init)
        # print("x_init", x_init)
        prog.SetInitialGuess(u[i], u_init)
        prog.SetInitialGuess(x[i], x_init)

    print("initial guesses")

    # Set up solver
    solver = SnoptSolver()
    result = solver.Solve(prog)
    print("solved")

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    t_land_sol = result.GetSolution(t_land)

    print(result.get_solution_result())

    # Reconstruct the trajecotry as a cubic hermite spline
    xdot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])

    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.FirstOrderHold(timesteps, u_sol.T)

    # print(x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u))

    return x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swing a leg.')
    parser.add_argument('--use_viz', action='store_true')

    args = parser.parse_args()

    print(args.use_viz)

    N = 2
    initial_state = np.zeros(6)
    tf = 5.0
    distance = 25.0
    x_traj, u_traj, prog, x_guess, u_guess = find_throwing_trajectory(N, initial_state, distance, tf)

    # %matplotlib notebook
    server_args = ['--ngrok_http_tunnel']

    # Start a single meshcat server instance to use for the remainder of this notebook.
    # from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
    # proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)
    zmq_url = "tcp://127.0.0.1:6000"
    web_url = "http://127.0.0.1:7000/static/"

    # Create a MultibodyPlant for the arm
    # file_name = "planar_arm.urdf"
    file_name = "leg_v2.urdf"
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    single_leg = builder.AddSystem(MultibodyPlant(0.0))
    single_leg.RegisterAsSourceForSceneGraph(scene_graph)
    Parser(plant=single_leg).AddModelFromFile(file_name)
    single_leg.Finalize()
    print("finalized leg")

    if args.use_viz:
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
        
        # print(single_leg.)

        builder.Connect(zero_inputs.get_output_port(), single_leg.get_actuation_input_port())
        builder.Connect(x_traj_source.get_output_port(), demux.get_input_port())
        builder.Connect(demux.get_output_port(0), to_pose.get_input_port())
        builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(single_leg.get_source_id()))

        ConnectDrakeVisualizer(builder, scene_graph)

        diagram = builder.Build()
        diagram.set_name("diagram")
        print(diagram)

        visualizer.load()
        print("\n!!!Open the visualizer by clicking on the URL above!!!")

        # Visualize the motion for `n_playback` times
        n_playback = 1
        # for i in range(n_playback):
        # Set up a simulator to run this diagram.
        simulator = Simulator(diagram)
        simulator.Initialize()
        simulator.set_target_realtime_rate(1)
        # time.sleep(15)
        simulator.AdvanceTo(tf)
        time.sleep(15)
