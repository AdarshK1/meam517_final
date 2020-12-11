from import_helper import *

import warnings
from pydrake.common.deprecation import DrakeDeprecationWarning

from obstacles import Obstacles
from viz_helper import *
from helper import *
from constraints import *

warnings.simplefilter("ignore", DrakeDeprecationWarning)


def find_step_trajectory(N, initial_state, final_state, apex_state, tf, obstacles=None, apex_hard_constraint=False,
                         td_hard_constraint=False, with_spline=True, x_spline_guess=None, u_spline_guess=None):
    """

    :param N:
    :param initial_state:
    :param final_state:
    :param apex_state:
    :param tf:
    :param obstacles:
    :param apex_hard_constraint:
    :param td_hard_constraint:
    :return:
    """
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

    if td_hard_constraint:
        # Add constraints on the final state
        A_end = np.identity(n_x)
        b_end = np.array(final_state)
        prog.AddLinearEqualityConstraint(A_end, b_end, x[-1].flatten())

    if N > 2 and apex_hard_constraint:
        A_apex = np.identity(n_x)
        b_apex = np.array(apex_state)
        prog.AddLinearEqualityConstraint(A_apex, b_apex, x[N // 2].flatten())

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, single_leg, context, N, x, u, timesteps)

    # Add constraint to remain above ground
    AddAboveGroundConstraint(prog, context, single_leg, plant, plant_context, x, N)

    Q = np.eye(n_u * N) * 0.25

    # multiplying the cost on abduction doesn't actually solve the crazy abduction problem, it makes it worse because
    # it now refuses to fight gravity!
    # setting ab cost to 0 so far yields the most logical results
    for i in range(N):
        Q[n_u * i] *= 0

    b = np.zeros([n_u * N, 1])
    # getting rid of cost on control for now, this is making it not fight gravity!
    if x_spline_guess is None:
        prog.AddQuadraticCost(Q, b, u.flatten())
    # print("Added quadcost")

    # Add rate limiting constraint
    # its unclear why this doesn't work..
    delta_u = np.array([20.0, 20.0, 20.0])
    delta_x = np.array([2.0, 2.0, 2.0, 100.0, 100.0, 100.0])
    # AddRateLimiterConstraint(prog, N, u, x, delta_x, delta_u)

    # Add constraint on effort limits
    AddEffortBBoxConstraints(prog, effort_limits, N, n_u, u)

    # add constraint on joint states/vels limits
    AddJointBBoxConstraints(prog, n_x, N, vel_limits, x)

    # add initial guesses and quadratic errors from nominal
    if x_spline_guess is None:
        AddInitialGuessQuadraticError(prog, initial_state, final_state, apex_state, N, n_u, n_x, u, x)
    else:
        # Set initial guess to interpolate between provided spline (x_traj)
        AddSplineGuessQuadraticError(prog, initial_state, final_state, apex_state, N, n_u, n_x, u, x, x_spline_guess,
                                     u_spline_guess, tf)

    # Set up solver
    solver = SnoptSolver()
    result = solver.Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)

    print("-" * 50)
    print(result.get_solution_result())
    print("-" * 50)

    # Reconstruct the trajectory as a cubic hermite spline
    xdot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])

    if not with_spline:
        out_dict = {"result.get_solution_result()": result.get_solution_result(),
                    "x_sol": x_sol,
                    "u_sol": u_sol,
                    "xdot_sol": xdot_sol,
                    "timesteps": timesteps,
                    "obstacles.heightmap": obstacles.heightmap,
                    "obstacles": obstacles}
        return out_dict

    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.FirstOrderHold(timesteps, u_sol.T)

    return x_traj, u_traj, x_sol, u_sol, xdot_sol, timesteps, prog, result


def multi_step_solve(N, initial_state, final_state, apex_state, tf, obstacles=None, apex_hard_constraint=False,
                     td_hard_constraint=False):
    x_traj = None
    u_traj = None

    for knot_num in [15, N]:
        print("Solving with N=" + str(knot_num))
        t3 = time.time()
        x_traj, u_traj, x_sol, u_sol, xdot_sol, timesteps, prog, result = find_step_trajectory(knot_num,
                                                                                               initial_state,
                                                                                               final_state,
                                                                                               apex_state,
                                                                                               tf,
                                                                                               obstacles,
                                                                                               x_spline_guess=x_traj,
                                                                                               u_spline_guess=u_traj,
                                                                                               apex_hard_constraint=apex_hard_constraint,
                                                                                               td_hard_constraint=td_hard_constraint,
                                                                                               with_spline=True)
        t4 = time.time()
        print("Time to solve: " + str(round(t4 - t3, 2)), "for ", knot_num)

    # Save coefficients for u trajectory
    u1_coef = np.zeros((N-1, 2))    # num segments x order of polynomial
    u2_coef = np.zeros((N-1, 2))
    u3_coef = np.zeros((N-1, 2))

    for i in range(N-1):
        poly_mat = u_traj.getPolynomialMatrix(i)

        u1_coef[i, :] = poly_mat[0][0].GetCoefficients()
        u2_coef[i, :] = poly_mat[1][0].GetCoefficients()
        u3_coef[i, :] = poly_mat[2][0].GetCoefficients()

    # print(u1_coef)

    out_dict = {"result.get_solution_result()": result.get_solution_result(),
                "x_sol": x_sol,
                "x_traj": x_traj,
                "u_sol": u_sol,
                "u_traj": u_traj,
                "xdot_sol": xdot_sol,
                "timesteps": timesteps,
                "obstacles": obstacles,
                "u1_coef": u1_coef,
                "u2_coef": u2_coef,
                "u3_coef": u3_coef}
    return out_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swing a leg.')
    parser.add_argument('--use_viz', action='store_true')
    parser.add_argument('--obstacles', action='store_true')
    parser.add_argument('--n_obst', default=4)
    parser.add_argument('--n_play', default=1)

    args = parser.parse_args()

    N = 35
    # nominal stance
    # initial_state = np.array([0, -2.0, 2.0, 0, 0, 0])

    # end of stance
    # initial_state = np.array([0, -2.5, 2.0, 0, 0, 0])

    # more aggressive apex
    # apex_state = np.array([0, -3.0, 0.5, 0, 0, 0])

    # less aggressive apex
    apex_state = np.array([0, -3.0, 1.5, 0, 0, 0])

    # end of step
    # initial_state = np.array([0, -2.0, 1.5, 0, 0, 0])
    # final_state = np.array([0, -1.5, 2.5, 0, 0, 0])

    # Large step
    initial_state = np.array([0, -2.5, 2.5, 0, 0, 0])
    final_state = np.array([0, -1.5, 2.2, 0, 0, 0])
    # final_state = initial_state
    # apex_state = initial_state

    # Small step
    # initial_state = np.array([0, -2.25, 1.75, 0, 0, 0])
    # final_state = np.array([0, -1.75, 1.95, 0, 0, 0])

    # Initialize obstacles
    obstacles = None
    if args.obstacles:
        obstacles = Obstacles(N=int(args.n_obst), multi_constraint=True)

    # final_state = initial_state
    tf = 2.0

    t1 = time.time()
    out_dict = multi_step_solve(N, initial_state, final_state, apex_state, tf)

    t2 = time.time()
    print("-" * 75)
    print("Time to solve: {}; Time per N: {}".format((t2 - t1), (t2 - t1) / N))
    print("-" * 75)
    if args.use_viz:
        do_viz(out_dict["x_traj"], out_dict["u_traj"], tf, int(args.n_play), obstacles)
