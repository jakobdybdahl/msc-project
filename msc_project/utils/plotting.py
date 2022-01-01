import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

header_names = "train_success_rate,train_num_timeout_failures,train_num_collisions_failures,train_num_minimum_reward_reached,train_total_collisions,train_total_timeouts,agent_noise_std,success_rate,num_timeout_failures,num_collision_failures,num_minimum_reward_reached,total_collisions,total_timeouts,epoch,average_reward,std_reward,max_reward,min_reward,average_test_reward,std_test_reward,max_test_reward,min_test_reward,steps,test_steps,total_env_steps,total_episodes,time".split(
    ","
)

default_confidence_header_names = "success_rate,num_collision_failures,num_timeout_failures,test_steps,total_collisions,total_timeouts,average_test_reward"

header_ylabels = {
    "success_rate": "Success rate",
    "num_collision_failures": "Collision failures",
    "num_timeout_failures": "Timeout failures",
    "num_timeout": "Timeout failures",
    "num_collision": "Collision failures",
    "test_steps": "Steps",
    "total_collisions": "Collisions",
    "total_timeouts": "Timeouts",
    "average_test_reward": "Average reward per episode"
}

epoch_label = "Training epochs"


extension = "pdf"

def set_header_names(new_header_names):
    global header_names
    header_names = new_header_names.split(',')

def decorate_common_plot():
    plt.grid(True, linestyle="dashed", linewidth=0.5)

def plot_simple(x, values, fig_file, title=None, xlabel=None, ylabel=None):
    # plt.figure()
    plt.plot(x, values)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)
    
    decorate_common_plot()
        
    plt.savefig(fig_file)
    plt.close()

def plot_run_values(base_dir, dist_dir, run):
    assert run is not None

    file = f"{base_dir}/run{run}/results.csv"

    # skip if file doesnt exists
    if not os.path.exists(file):
        return

    dist_dir = f"{dist_dir}/run{run}-plots"
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    x = data[:, header_names.index("epoch")]
    for header in header_names:
        if header == "epoch":
            continue
        try:
            header_index = header_names.index(header)
        except ValueError:
            continue
        scores = data[:, header_index]
        out_file = f"{dist_dir}/{header}.{extension}"

        if header in header_ylabels:
            ylabel = header_ylabels[header]
        else:
            ylabel = header

        plot_simple(x, scores, out_file, xlabel=epoch_label, ylabel=ylabel)


def plot_confidence(base_dir, dist_dir, header_names):
    assert header_names is not None

    if "," in header_names:
        header_names = header_names.split(",")
    else:
        header_names = [header_names]

    runs = read_runs_in_folder(base_dir)
    assert len(runs) > 0

    for header_name in header_names:
        data = get_data(base_dir, runs, header_name)

        if data is None:
            print("Skipped plot with confidence interval due to error in data...")
            break

        df = pd.DataFrame(data).melt()
        ax = sns.lineplot(x="variable", y="value", data=df)

        if header_name in header_ylabels:
            ylabel = header_ylabels[header_name]
        else:
            ylabel = header_name

        ax.set(xlabel=epoch_label, ylabel=ylabel)
        # ax.grid(True, linestyle="dashed", linewidth=0.5)
        decorate_common_plot()

        fig_file = f"{dist_dir}/confidence-{header_name}.{extension}"
        plt.savefig(fig_file)
        plt.close()


def get_data(base_dir, runs, header_name):
    data = None
    initialized = False

    for i, run in enumerate(runs):
        file = f"{base_dir}/run{run}/results.csv"

        # skip if file doesnt exists
        if not os.path.exists(file):
            return None

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

def read_runs_in_folder(path_dir):
    path_dir = Path(path_dir)

    return [
        int(str(folder.name).split("run")[1]) for folder in path_dir.iterdir() if str(folder.name).startswith("run")
    ]


# TODO not working due to some wrong matplotlib..
def plot_everything(args):
    run_dir = Path(args.base_dir)

    def plot_experiment(algo, exp):
        exp_dir = run_dir / algo / exp

        dist_dir = f"{args.output_dir}/{algo}/{exp}"
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)

        runs = read_runs_in_folder(str(exp_dir))
        for run in runs:
            print(f"\tPlotting values for run # {run}")
            try:
                plot_run_values(exp_dir, dist_dir, run)
            except Exception:
                return

        print(f"\tPlotting confidence plots for {exp}")
        plot_confidence(exp_dir, dist_dir, default_confidence_header_names)

    def plot_algorithm(algo):
        algo_dir = run_dir / algo
        exps = [str(folder.name) for folder in algo_dir.iterdir()]
        for exp in exps:
            print(f"\tPlotting everything for experiment {exp}")
            plot_experiment(algo, exp)
            
    if args.algorithm_name is not None:
        print(f"Plotting everyting for algoritm '{args.algorithm_name}'")
        plot_algorithm(args.algorithm_name)
    else:
        algos = [str(folder.name) for folder in run_dir.iterdir()]
        for algo in algos:
            print(f"Plotting everything for algorithm '{algo}'")
            plot_algorithm(algo)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm_name", type=str, default=None, choices=[None, "ddpg", "maddpg", "gnn"])
    parser.add_argument("--experiment_name", type=str, default="debug")
    parser.add_argument("--base_dir", type=str, default="C:\Data\workspace\msc-thesis")
    parser.add_argument("--output_dir", type=str, default="dist-debug")
    parser.add_argument("--header_names", type=str, default=None)

    # plot type specific arguments
    # plot_type = 'everything'
    parser.add_argument("--run", type=int, default=None)

    # plot_type = 'confidence'
    parser.add_argument("--confidence_header_name", type=str, default=None)

    parser.add_argument("--plot_type", type=str, default="run", choices=["run", "confidence"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    work_dir = f"{args.base_dir}/{args.algorithm_name}/{args.experiment_name}"
    dist_dir = f"{args.output_dir}/{args.algorithm_name}/{args.experiment_name}"

    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    if args.header_names is not None:
        set_header_names(args.header_names)

    plot_type = args.plot_type

    if plot_type == "run":
        plot_run_values(work_dir, dist_dir, args.run)
    elif plot_type == "confidence":
        plot_confidence(work_dir, dist_dir, args.confidence_header_name)
    else:
        raise NotImplementedError()        
