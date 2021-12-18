import json
import os
import sys
from pathlib import Path

import gym
import numpy as np
import setproctitle
import torch as T
from msc_project.config import get_config
from msc_project.runner.tjc_runner import TJCRunner
from msc_project.utils.logger import EpochLogger

def make_train_env(args):
    env = gym.make(args.env_name)
    env.seed(args.seed)

    # prepare action space (represents all action spaces)
    act_space = env.action_space[0]
    act_space.seed(args.seed)

    # gym wraps the original env, so we get the inner and sets variables from args
    inner = env.env
    inner.set_r_fov(args.fov_radius)
    inner.arrive_prob = args.arrive_prob
    inner.step_cost = args.step_cost_factor
    inner.collision_cost = args.collision_cost
    inner.max_steps = args.max_agent_episode_steps

    if args.algorithm_name == 'maddpg':
        inner.reward_callback = lambda rewards, n_agents: [sum(rewards)] * n_agents

    return env, act_space


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
        device = T.device("cpu")

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

    # set seeds
    T.manual_seed(all_args.seed)
    T.cuda.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # create environment. act_space is to parse to policy_config
    env, act_space = make_train_env(all_args)
    num_agents = env.n_agents

    # setup logger and save args for this run
    logger = EpochLogger(output_dir=str(run_dir))
    logger.save_config(all_args.__dict__)

    # setup info for policies
    obs_shape = env.observation_space[0].shape
    cent_obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2] * env.n_agents
    policy_info = {
        "obs_space": gym.spaces.flatten_space(env.observation_space[0]),
        "act_space": act_space,
        "num_agents": num_agents,
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
        "logger": logger,
    }

    runner = TJCRunner(config=config)  # specific runner for tjc-gym env
    for _ in range(all_args.epochs):
        runner.run_epoch()

    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
