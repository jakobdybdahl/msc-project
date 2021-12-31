import argparse

import numpy as np
from msc_project.utils.plotting import get_data, read_runs_in_folder, set_header_names


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm_name", type=str, default="ddpg", choices=[None, "ddpg", "maddpg", "gnn"])
    parser.add_argument("--experiment_name", type=str, default="default-tjc-env")
    parser.add_argument("--base_dir", type=str, default="C:\Data\workspace\msc-thesis")
    parser.add_argument("--header_name", type=str, default="success_rate")

    parser.add_argument("--header_names", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    work_dir = f"{args.base_dir}/{args.algorithm_name}/{args.experiment_name}"

    if args.header_names is not None:
        set_header_names(args.header_names)

    runs = read_runs_in_folder(work_dir)
    data = get_data(work_dir, runs, args.header_name)

    max_i = np.unravel_index(data.argmax(), data.shape)

    # run = 9
    # episodes =

    print(f"***** METRICS FOR *****")
    print(f"      {args.header_name}      ")
    print(f"***********************\n")
    print(f"Shape of data = {data.shape}")
    print(f"Max {args.header_name} = {data[max_i]}, run {runs[max_i[0]]} at epoch {max_i[1]}")
    print(f"Top 10 max:")

    x, y = np.unravel_index(np.argpartition(data, -10, axis=None)[-10:], data.shape)
    for max_i in zip(x, y):
        print(f"\tin run {runs[max_i[0]]} at epoch {max_i[1]} = {data[max_i]}")
