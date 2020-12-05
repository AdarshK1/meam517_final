from import_helper import *

from find_trajectory import find_step_trajectory, multi_step_solve

from obstacles import Obstacles
from viz_helper import *
from helper import *
from constraints import *

from multiprocessing import Pool, Process
import threading
import pickle
from datetime import datetime


def randomize_state(initial, apex, final, angle_std=0.25, vel_std=0.5):
    initial[0] += np.random.random() * angle_std - angle_std / 2
    initial[3] += np.random.random() * vel_std - vel_std / 2

    initial[1] += np.random.random() * angle_std - angle_std / 2
    initial[4] += np.random.random() * vel_std - vel_std / 2

    initial[2] += np.random.random() * angle_std - angle_std / 2
    initial[5] += np.random.random() * vel_std - vel_std / 2

    apex[0] += np.random.random() * angle_std - angle_std / 2
    apex[3] += np.random.random() * vel_std - vel_std / 2

    apex[1] += np.random.random() * angle_std - angle_std / 2
    apex[4] += np.random.random() * vel_std - vel_std / 2

    apex[2] += np.random.random() * angle_std - angle_std / 2
    apex[5] += np.random.random() * vel_std - vel_std / 2

    final[0] += np.random.random() * angle_std - angle_std / 2
    final[3] += np.random.random() * vel_std - vel_std / 2

    final[1] += np.random.random() * angle_std - angle_std / 2
    final[4] += np.random.random() * vel_std - vel_std / 2

    final[2] += np.random.random() * angle_std - angle_std / 2
    final[5] += np.random.random() * vel_std - vel_std / 2

    return initial, apex, final


def call_find_trajectory(args):
    solved = multi_step_solve(args["N"],
                                args["initial_state"],
                                args["final_state"],
                                args["apex_state"],
                                args["tf"],
                                obstacles=args["obstacles"])

    # can't pickle trajectories
    solved.pop('x_traj', None)
    solved.pop('u_traj', None)

    # filename
    dt = datetime.now().strftime("%m_%d_%H_%M_%S")
    path = data_dir + "{}.pkl"

    # dump
    pickle.dump((args, solved), open(path.format(dt), 'wb'))
    return solved


if __name__ == '__main__':

    n_threads = 48
    n_outputs = 1000

    overall_counter = 0
    data_dir = "data_v2/"

    # print("FIX THE NOUGHTS")
    N = 35
    # default values
    apex_state = np.array([0, -3.0, 1.5, 0, 0, 0])
    initial_state = np.array([0, -2.5, 2.5, 0, 0, 0])
    final_state = np.array([0, -1.5, 2.2, 0, 0, 0])

    # standard deviations
    angle_std = 0.0  # 0.10
    vel_std = 0.0  # 0.25

    # final time
    tf = 2

    states = []

    # generate all the cases
    for i in range(n_outputs):
        # now randomize start, apex, final
        initial_state, apex_state, final_state = randomize_state(initial_state, apex_state, final_state, angle_std,
                                                                 vel_std)

        # obstacles
        n_obst = int(round(np.random.normal(4, 2)))
        obstacles = Obstacles(N=n_obst, multi_constraint=True)

        states.append({"N": N,
                       "initial_state": initial_state,
                       "apex_state": apex_state,
                       "final_state": final_state,
                       "obstacles": obstacles,
                       "tf": tf})

    threads = []
    initial_len = len(states)
    while len(states) > 0:
        time.sleep(2)
        active_count = 0
        for thread in threads:
            if thread.is_alive():
                active_count += 1

        print("Currently active threads: {}".format(active_count))
        print("Remaining States: {}".format(len(states)))
        print("Progress: {:.02f}".format(1 - len(states) / initial_len))
        print("-" * 75)
        if active_count < n_threads:
            threads.append(Process(target=call_find_trajectory, args=(states.pop(),)))
            threads[-1].start()
