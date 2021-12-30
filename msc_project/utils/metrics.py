import argparse

import numpy as np
from msc_project.utils.plotting import get_data, read_runs_in_folder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm_name", type=str, default=None, choices=[None, "ddpg", "maddpg", "gnn"])
    parser.add_argument("--experiment_name", type=str, default="default-tjc-env")
    parser.add_argument("--base_dir", type=str, default="C:\Data\workspace\msc-thesis")
    parser.add_argument("--header_name", type=str, default="success_rate")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    work_dir = f"{args.base_dir}/{args.algorithm_name}/{args.experiment_name}"

    runs = read_runs_in_folder(work_dir)
    data = get_data(work_dir, runs, args.header_name)

    max_i = np.unravel_index(data.argmax(), data.shape)

    print(data[max_i])
