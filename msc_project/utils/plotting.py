import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

header_names = "train_success_rate,train_num_timeout_failures,train_num_collisions_failures,train_num_minimum_reward_reached,train_total_collisions,train_total_timeouts,agent_noise_std,success_rate,num_timeout_failures,num_collision_failures,num_minimum_reward_reached,total_collisions,total_timeouts,epoch,average_reward,std_reward,max_reward,min_reward,average_test_reward,std_test_reward,max_test_reward,min_test_reward,steps,test_steps,total_env_steps,total_episodes,time".split(
    ","
)


def plot_with_confidence_interval():
    # read file
    runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = np.arange(1, 101, 1)
    data = np.empty((len(runs), len(x)))

    dist_dir = f"dist/ddpg-default-tjc-env"
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    for i, run in enumerate(runs):
        file = f"C:/Users/Jakob Dybdahl/source/repos/msc-project/msc_project/scripts/results/TrafficJunctionContinuous6-v0/ddpg/default-tjc-env/run{run}/results.csv"
        run_data = np.genfromtxt(
            file,
            delimiter=",",
            skip_header=1,
        )
        data[i] = run_data[:, header_names.index("success_rate")]

    df = pd.DataFrame(data).melt()

    ax = sns.lineplot(x="variable", y="value", data=df)
    ax.set(xlabel="epoch", ylabel="success rate")

    # plt.show()

    plt.savefig(f"{dist_dir}/success_rate.svg")


def get_data(base_dir, runs, header_name):
    data = None
    initialized = False

    for i, run in enumerate(runs):
        file = f"{base_dir}/run{run}/results.csv"
        run_data = np.genfromtxt(
            file,
            delimiter=",",
            skip_header=1,
        )

        if not initialized:
            data = np.empty((len(runs), run_data.shape[0]))
            initialized = True

        data[i] = run_data[:, header_names.index(header_name)]

    return data


if __name__ == "__main__":
    # plot_with_confidence_interval()
    runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    success_rates = get_data(
        "C:/Users/Jakob Dybdahl/source/repos/msc-project/msc_project/scripts/results/TrafficJunctionContinuous6-v0/ddpg/default-tjc-env",
        runs,
        "success_rate",
    )

    index_of_max_success_rate = np.unravel_index(np.argmax(success_rates), success_rates.shape)

    print(success_rates[8, 61])
    print(success_rates[index_of_max_success_rate])
    print(success_rates[8, 63])

    print("hello")

    # x = np.linspace(0, 15, 31)
    # data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    # df = pd.DataFrame(data).melt()
    # sns.lineplot(x="variable", y="value", data=df)
