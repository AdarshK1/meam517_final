from import_helper import *

from find_trajectory import find_step_trajectory

from multiprocessing import Pool
import pickle

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
    return find_step_trajectory(args["N"],
                                args["initial_state"],
                                args["final_state"],
                                args["apex_state"],
                                args["tf"],
                                obstacles=args["obstacles"],
                                with_spline=False)


if __name__ == '__main__':

    n_threads = 32
    n_outputs = 5000

    overall_counter = 0
    data_dir = "data/"

    for i in range(int(n_outputs / n_threads)):
        N = 35
        # default values
        apex_state = np.array([0, -3.0, 1.5, 0, 0, 0])
        initial_state = np.array([0, -2.5, 2.5, 0, 0, 0])
        final_state = np.array([0, -1.5, 2.2, 0, 0, 0])

        # standard deviations
        angle_std = 0.25
        vel_std = 0.5

        # final time
        tf = 2

        states = []

        # randomize n_threads of inputs
        for j in range(n_threads):
            # now randomize start, apex, final
            initial_state, apex_state, final_state = randomize_state(initial_state, apex_state, final_state, angle_std,
                                                                     vel_std)

            # obstacles
            n_obst = np.random.normal(10, 5)
            obstacles = Obstacles(N=n_obst, multi_constraint=True)

            states.append({"N": N,
                           "initial_state": initial_state,
                           "apex_state": apex_state,
                           "final_state": final_state,
                           "obstacles": obstacles,
                           "tf": tf})

        # now solve all the trajs
        with Pool(n_threads) as p:
            outputs = p.map(call_find_trajectory, states)

            print(len(states))
            print("states: ", states[0])
            print(len(outputs))
            print("outputs: ", outputs[0])

            for j in range(n_threads):

                path = data_dir + "{:0>6d}.pkl"
                pickle.dump((states[j], outputs[j]), open(path.format(overall_counter), 'wb'))
                overall_counter += 1