import os

import matplotlib.pyplot as plt
import numpy as np


def plot(x, scores, running_avg_size=10, fig_file=None, title=None):
    plt.figure()

    # running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(scores[max(0, i - running_avg_size) : (i + 1)])
    # plt.plot(x, running_avg)
    plt.plot(x, scores)

    if title is not None:
        plt.title(title)

    # plt.show()

    if fig_file != None:
        plt.savefig(fig_file)
    else:
        plt.show()


if __name__ == "__main__":
    # read file
    runs = [1, 2, 3, 4, 5]
    dist_dir = "dist"

    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    # for run in runs:
    # read file
    run = 28
    file = f"C:/Users/Jakob Dybdahl/source/repos/msc-project/msc_project/scripts/results/TrafficJunctionContinuous6-v0/ddpg/debug/run{run}/results.csv"
    header_names = "success_rate,num_timeout,num_collision,epoch,average_reward,std_reward,max_reward,min_reward,average_test_reward,std_test_reward,max_test_reward,min_test_reward,steps,test_steps,total_env_steps,total_episodes,time".split(
        ","
    )
    data = np.genfromtxt(
        file,
        delimiter=",",
        skip_header=1,
    )

    names = [
        "success_rate",
        "num_timeout",
        "num_collision",
        "average_reward",
        "average_test_reward",
        "test_steps",
        "std_test_reward",
    ]

    for name in names:
        # get episode scores
        x = data[:, header_names.index("epoch")]
        scores = data[:, header_names.index(name)]

        # plot
        out_file = f"{dist_dir}/run{run}_{name}.png"
        plot(x, scores, fig_file=out_file, running_avg_size=1, title=name)
