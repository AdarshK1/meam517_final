from import_helper import *

from find_trajectory import find_step_trajectory

from multiprocessing import Pool


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


if __name__ == '__main__':

    n_threads = 4
    n_outputs = 5000

    for i in range(int(n_outputs / n_threads)):
        for j in range(n_threads):
        N = 35
        # default values
        apex_state = np.array([0, -3.0, 1.5, 0, 0, 0])
        initial_state = np.array([0, -2.5, 2.5, 0, 0, 0])
        final_state = np.array([0, -1.5, 2.2, 0, 0, 0])

        # standard deviations
        angle_std = 0.25
        vel_std = 0.5

        # now randomize start, apex, final
        initial_state, apex_state, final_state = randomize_state(initial_state, apex_state, final_state, angle_std, vel_std)

        # final time
        tf = 3

        # obstacles
        n_obst = np.random.normal(10, 5)
        obstacles = Obstacles(N=n_obst, multi_constraint=True)

