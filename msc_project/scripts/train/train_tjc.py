import os
import sys
from pathlib import Path

import gym
import numpy as np
import setproctitle
import torch as T
from msc_project.config import get_config
from msc_project.runner.tjc_runner import TJCRunner


def make_train_env(args):
    env = gym.make(args.env_name)

    # gym wraps the original env, so we get the inner and sets variables from args
    inner = env.env
    inner.set_r_fov(args.fov_radius)
    inner.arrive_prob = args.arrive_prob
    inner.step_cost = args.step_cost_factor
    inner.collision_cost = args.collision_cost

    return env


def parse_args(args, parser):
    parser.add_argument("--step_cost_factor", type=float, default=-0.01)
    parser.add_argument("--collision_cost", type=float, default=-100)
    parser.add_argument("--arrive_prob", type=float, default=0.05)
    parser.add_argument("--fov_radius", type=int, default=3)

    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if T.cuda.is_available():
        print("Setting GPU as device..")
        device = T.device("cuda:0")
    else:
        print("Setting CPU as device..")
        device = T.device("cp")

    env_name_formatted = all_args.env_name.split(":")[1]

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / env_name_formatted
        / all_args.algorithm_name
        / all_args.experiment_name
    )

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1]) for folder in run_dir.iterdir() if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(env_name_formatted)
        + "-"
        + str(all_args.experiment_name)
        + "-"
        + str(curr_run)
    )

    # TODO set seeds in torch and numpy - get seed from args?
    T.manual_seed(all_args.seed)
    T.cuda.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # create environment
    env = make_train_env(all_args)
    num_agents = env.n_agents

    # setup info for policies
    obs_shape = env.observation_space[0].shape
    cent_obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2] * env.n_agents
    policy_info = {
        "obs_space": gym.spaces.flatten_space(env.observation_space[0]),
        "act_space": gym.spaces.flatten_space(env.action_space[0]),
        "cent_act_dim": env.action_space[0].shape[0] * env.n_agents,
        "cent_obs_dim": cent_obs_dim,
    }

    config = {
        "args": all_args,
        "policy_info": policy_info,
        "env": env,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    total_num_steps = 0
    runner = TJCRunner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()

    # TODO save logs / report ?


if __name__ == "__main__":
    main(sys.argv[1:])
