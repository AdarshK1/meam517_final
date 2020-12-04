from import_helper import *
from pydrake.autodiffutils import AutoDiffXd

from helper import get_world_position


def cos(theta):
    return AutoDiffXd.cos(theta)


def sin(theta):
    return AutoDiffXd.sin(theta)


def EvaluateDynamics(single_leg, context, x, u):
    # Computes the dynamics xdot = f(x,u)

    single_leg.SetPositionsAndVelocities(context, x)

    M = single_leg.CalcMassMatrixViaInverseDynamics(context)
    B = single_leg.MakeActuationMatrix()
    g = single_leg.CalcGravityGeneralizedForces(context)
    C = single_leg.CalcBiasTerm(context)

    M_inv = np.zeros((3, 3))
    if (x.dtype == AutoDiffXd):
        M_inv = AutoDiffXd.inv(M)
    else:
        M_inv = np.linalg.inv(M)

    v_dot = M_inv @ (B @ u + g - C)
    return np.hstack((x[-3:], v_dot))


def CollocationConstraintEvaluator(single_leg, context, dt, x_i, u_i, x_ip1, u_ip1):
    h_i = np.zeros((6,), dtype=AutoDiffXd)
    # TODO: Evaluate the collocation constraint h using x_i, u_i, x_ip1, u_ip1, dt
    # You should make use of the EvaluateDynamics() function to compute f(x,u)

    f_dyn_i = EvaluateDynamics(single_leg, context, x_i, u_i)
    f_dyn_ip1 = EvaluateDynamics(single_leg, context, x_ip1, u_ip1)

    x_c = -0.125 * dt * (f_dyn_ip1 - f_dyn_i) + 0.5 * (x_ip1 + x_i)
    u_c = u_i + 0.5 * (u_ip1 - u_i)

    f_x_c = EvaluateDynamics(single_leg, context, x_c, u_c)

    x_dot_c = (1.5 / dt) * (x_ip1 - x_i) - 0.25 * (f_dyn_i + f_dyn_ip1)

    h_i = x_dot_c - f_x_c

    return h_i


def AddCollocationConstraints(prog, single_leg, context, N, x, u, timesteps):
    n_u = single_leg.num_actuators()
    n_x = single_leg.num_positions() + single_leg.num_velocities()

    for i in range(N - 1):
        def CollocationConstraintHelper(vars):
            x_i = vars[:n_x]
            u_i = vars[n_x:n_x + n_u]
            x_ip1 = vars[n_x + n_u: 2 * n_x + n_u]
            u_ip1 = vars[-n_u:]
            dt = timesteps[1] - timesteps[0]
            # print("called helper")
            return CollocationConstraintEvaluator(single_leg, context, dt, x_i, u_i, x_ip1, u_ip1)

        # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
        #       to prog
        # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
        # where vars = hstack(x[i], u[i], ...)
        lb = np.array([0, 0, 0, 0, 0, 0])
        ub = np.array([0, 0, 0, 0, 0, 0])
        vars = np.hstack((x[i], u[i], x[i + 1], u[i + 1]))
        prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)


def AddJointBBoxConstraints(prog, n_x, N, vel_limits, x):
    ub = np.zeros([N * n_x])
    lb = np.zeros([N * n_x])

    for i in range(N):
        # abduction joint limit
        ub[i * n_x] = 0.785
        lb[i * n_x] = -0.785

        # hip joint limit
        ub[i * n_x + 1] = 0
        lb[i * n_x + 1] = -3.14

        # knee joint limit
        ub[i * n_x + 2] = 3.14
        lb[i * n_x + 2] = 0.25

        ub[i * n_x + 3] = vel_limits[0]
        lb[i * n_x + 3] = -vel_limits[0]

        ub[i * n_x + 4] = vel_limits[1]
        lb[i * n_x + 4] = -vel_limits[1]

        ub[i * n_x + 5] = vel_limits[2]
        lb[i * n_x + 5] = -vel_limits[2]

    prog.AddBoundingBoxConstraint(lb, ub, x.flatten())


def AddEffortBBoxConstraints(prog, effort_limits, N, n_u, u):
    ub = np.zeros([N * n_u])
    for i in range(N):
        ub[i * n_u] = effort_limits[0]
        ub[i * n_u + 1] = effort_limits[1]
        ub[i * n_u + 2] = effort_limits[2]

    lb = -ub
    prog.AddBoundingBoxConstraint(lb, ub, u.flatten())


def AddInitialGuessQuadraticError(prog, initial_state, final_state, apex_state, N, n_u, n_x, u, x):
    for i in range(N):
        # u_init = np.random.rand(n_u) * effort_limits * 2 - effort_limits
        u_init = np.zeros(n_u)
        prog.SetInitialGuess(u[i], u_init)

        if N < 3:
            x_init = initial_state + (i / N) * (final_state - initial_state)
            prog.SetInitialGuess(x[i], x_init)
            prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])

        elif N > 3 and i < N / 2:
            x_init = initial_state + (i / (N / 2)) * (apex_state - initial_state)
            # print(i, x[i].flatten(), x_init)
            prog.SetInitialGuess(x[i], x_init)
            prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])

        else:
            x_init = apex_state + ((i - N / 2) / (N / 2)) * (final_state - apex_state)
            # print(i, x[i].flatten(), x_init)
            prog.SetInitialGuess(x[i], x_init)
            prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])


def AddSplineGuessQuadraticError(prog, initial_state, final_state, apex_state, N, n_u, n_x, u, x, x_spline_guess,
                                 u_spline_guess, tf):
    t = np.linspace(0, tf, N)
    for i in range(N):
        # u_init = np.random.rand(n_u) * effort_limits * 2 - effort_limits
        u_init = np.zeros(n_u)
        # prog.SetInitialGuess(u[i], u_init)

        prog.SetInitialGuess(x[i], x_spline_guess.value(t[i]))
        # print("x_spline_guess", x_spline_guess.value(t[i]))
        prog.SetInitialGuess(u[i], u_spline_guess.value(t[i]))
        # print("u_spline_guess", u_spline_guess.value(t[i]))

        # if N < 3:
        #     x_init = initial_state + (i / N) * (final_state - initial_state)
        #     prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])
        #
        # elif N > 3 and i < N / 2:
        #     x_init = initial_state + (i / (N / 2)) * (apex_state - initial_state)
        #     prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])
        #
        # else:
        #     x_init = apex_state + ((i - N / 2) / (N / 2)) * (final_state - apex_state)
        #     prog.AddQuadraticErrorCost(np.eye(int(n_x / 2)), x_init[:3], x[i][:3])


def AddAboveGroundConstraint(prog, context, single_leg, plant, plant_context, x, N):
    for i in range(N):
        prog.AddConstraint(
            (lambda state: [get_world_position(context, single_leg, plant, plant_context, "toe0", state)[2]]),
            lb=[0], ub=[float('inf')], vars=x[i])


def AddRateLimiterConstraint(prog, N, u, x, delta_x, delta_u):
    for i in range(N - 1):
        print(x[i])
        print(np.abs(x[i + 1] - x[i]))
        print(delta_x)
        # prog.AddConstraint(np.abs(x[i + 1] - x[i]), lb=np.zeros(6), ub=delta_x, vars=x[i:i+2])
        # prog.AddBoundingBoxConstraint(np.zeros(6), delta_x, np.abs(x[i + 1] - x[i]))
        # prog.AddLinearInequalityConstraint()
        prog.AddLinearConstraint(abs(x[i + 1] - x[i]) <= delta_x, vars=x[i:i + 2])
        # prog.AddConstraint(np.abs(u[i + 1] - u[i]) <= delta_u)
