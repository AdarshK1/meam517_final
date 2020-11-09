import numpy as np
from pydrake.autodiffutils import AutoDiffXd

from pydrake.solvers.mathematicalprogram import MathematicalProgram


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
