import numpy as np
from pydrake.autodiffutils import AutoDiffXd

from pydrake.solvers.mathematicalprogram import MathematicalProgram

def cos(theta):
    return AutoDiffXd.cos(theta)
def sin(theta):
    return AutoDiffXd.sin(theta)


def landing_constraint(vars):
    '''
    Impose a constraint such that if the ball is released at final state xf,
    it will land a distance d from the base of the robot
    '''
    l = 1
    g = 9.81
    constraint_eval = np.zeros((3,), dtype=AutoDiffXd)
    q = vars[:3]
    qdot = vars[3:6]
    t_land = vars[-1]
    pos = np.array([[-l*sin(q[0]) - l*sin(q[0] + q[1]) - l*sin(q[0] + q[1] + q[2])],
                    [-l*cos(q[0]) - l*cos(q[0] + q[1]) - l*cos(q[0] + q[1] + q[2])]])
    vel = np.array([[-l*qdot[2]*cos(q[0] + q[1] + q[2]) + qdot[0]*(-l*cos(q[0]) - l*cos(q[0] + q[1]) - l*cos(q[0] + q[1] + q[2])) + qdot[1]*(-l*cos(q[0] + q[1]) - l*cos(q[0] + q[1] + q[2]))],
                    [l*qdot[2]*sin(q[0] + q[1] + q[2]) + qdot[0]*(l*sin(q[0]) + l*sin(q[0] + q[1]) + l*sin(q[0] + q[1] + q[2])) + qdot[1]*(l*sin(q[0] + q[1]) + l*sin(q[0] + q[1] + q[2]))]])

    constraint_eval[0] = (pos[1][0] + vel[1][0] * t_land - 0.5 * g * t_land **2)
    constraint_eval[1] = pos[0][0] + vel[0][0] * t_land
    constraint_eval[2] = pos[1][0]

    return constraint_eval


def AddFinalLandingPositionConstraint(prog, xf, d, t_land):

    # TODO: Add the landing distance equality constraint as a system of inequality constraints
    # using prog.AddConstraint(landing_constraint, lb, ub, vars)
    lb = np.array([0, d, 2.0])
    ub = np.array([0, d, 100])
    combined_vars = np.append(xf, t_land)
    prog.AddConstraint(landing_constraint, lb, ub, combined_vars)

    # TODO: Add a constraint that t_land is positive
    prog.AddConstraint(t_land[0] >= 0)
    # pass

import numpy as np

from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.autodiffutils import AutoDiffXd


def EvaluateDynamics(planar_arm, context, x, u):
    # Computes the dynamics xdot = f(x,u)

    planar_arm.SetPositionsAndVelocities(context, x)

    M = planar_arm.CalcMassMatrixViaInverseDynamics(context)
    B = planar_arm.MakeActuationMatrix()
    g = planar_arm.CalcGravityGeneralizedForces(context)
    C = planar_arm.CalcBiasTerm(context)

    M_inv = np.zeros((3,3))
    if(x.dtype == AutoDiffXd):
        M_inv = AutoDiffXd.inv(M)
    else:
        M_inv = np.linalg.inv(M)
    v_dot = M_inv @ (B @ u + g - C)
    return np.hstack((x[-3:], v_dot))


def CollocationConstraintEvaluator(planar_arm, context, dt, x_i, u_i, x_ip1, u_ip1):
    h_i = np.zeros((6,), dtype=AutoDiffXd)
    # TODO: Evaluate the collocation constraint h using x_i, u_i, x_ip1, u_ip1, dt
    # You should make use of the EvaluateDynamics() function to compute f(x,u)

    f_dyn_i = EvaluateDynamics(planar_arm, context, x_i, u_i)
    f_dyn_ip1 = EvaluateDynamics(planar_arm, context, x_ip1, u_ip1)

    x_c = -0.125 * dt * (f_dyn_ip1 - f_dyn_i) + 0.5 * (x_ip1 + x_i)
    u_c = u_i + 0.5 * (u_ip1 - u_i)

    f_x_c = EvaluateDynamics(planar_arm, context, x_c, u_c)

    x_dot_c = (1.5 / dt) * (x_ip1 - x_i) - 0.25 * (f_dyn_i + f_dyn_ip1)

    h_i = x_dot_c - f_x_c

    return h_i

def AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps):
    n_u = planar_arm.num_actuators()
    n_x = planar_arm.num_positions() + planar_arm.num_velocities()


    for i in range(N - 1):
        def CollocationConstraintHelper(vars):
            x_i = vars[:n_x]
            u_i = vars[n_x:n_x + n_u]
            x_ip1 = vars[n_x + n_u: 2*n_x + n_u]
            u_ip1 = vars[-n_u:]
            dt = timesteps[1] - timesteps[0]
            # print("called helper")
            return CollocationConstraintEvaluator(planar_arm, context, dt, x_i, u_i, x_ip1, u_ip1)

        # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
        #       to prog
        # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
        # where vars = hstack(x[i], u[i], ...)
        lb = np.array([0, 0, 0, 0, 0, 0])
        ub = np.array([0, 0, 0, 0, 0, 0])
        vars = np.hstack((x[i], u[i], x[i+1], u[i+1]))
        prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
