from import_helper import *


def find_step_trajectory(N, initial_state, final_state, apex_state, tf, obstacles=None, apex_hard_constraint=False, td_hard_constraint=False):
    '''
    Parameters:
      N - number of knot points
      initial_state - starting configuration
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
    # AddAboveGroundConstraint(prog, context, single_leg, plant, plant_context, x, N)

    Q = np.eye(n_u * N)

    # multiplying the cost on abduction doesn't actually solve the crazy abduction problem, it makes it worse because
    # it now refuses to fight gravity!
    # for i in range(N):
    #     Q[n_u * i] *= 0

    b = np.zeros([n_u * N, 1])
    # getting rid of cost on control for now, this is making it not fight gravity!
    # prog.AddQuadraticCost(Q, b, u.flatten())
    # print("Added quadcost")

    AddEffortBBoxConstraints(prog, effort_limits, N, n_u, u)

    AddJointBBoxConstraints(prog, n_x, N, vel_limits, x)

    AddInitialGuessQuadraticError(prog, initial_state, final_state, apex_state, N, n_u, n_x, u, x)

    # print("\n", "-" * 50)
    # print("Costs")
    # print("generic_costs", prog.generic_costs())
    # print("linear_costs", prog.linear_costs())
    # print("quadratic_costs", prog.quadratic_costs())
    # print("-" * 50, "\n")

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
    parser.add_argument('--n_obst', default=1)
    parser.add_argument('--n_play', default=1)

    args = parser.parse_args()

    N = 15
    # nominal stance
    # initial_state = np.array([0, -2.0, 2.0, 0, 0, 0])

    # end of stance
    # initial_state = np.array([0, -2.5, 2.0, 0, 0, 0])

    # apex
    apex_state = np.array([0, -3.0, 0.5, 0, 0, 0])

    # end of step
    # initial_state = np.array([0, -2.0, 1.5, 0, 0, 0])
    # final_state = np.array([0, -1.5, 2.5, 0, 0, 0])


    # Large step
    initial_state = np.array([0, -2.5, 2.5, 0, 0, 0])
    final_state = np.array([0, -1.5, 2.2, 0, 0, 0])

    # Small step
    # initial_state = np.array([0, -2.25, 1.75, 0, 0, 0])
    # final_state = np.array([0, -1.75, 1.95, 0, 0, 0])

    # Initialize obstacles
    obstacles = None
    if args.obstacles:
        obstacles = Obstacles(N=int(args.n_obst), multi_constraint=True)

    # final_state = initial_state
    tf = 2.0
    x_traj, u_traj, prog, x_guess, u_guess = find_step_trajectory(N, initial_state, final_state, apex_state, tf,
                                                                  obstacles)

    if args.use_viz:
        do_viz(x_traj, u_traj, tf, int(args.n_play), obstacles)
