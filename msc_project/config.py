import argparse


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm_name", type=str, default="ddpg", choices=["maddpg", "ddpg"])
    parser.add_argument("--experiment_name", type=str, default="debug")
    parser.add_argument(
        "--num_env_steps", type=int, default=1000000, help="number of steps in environment to train for"
    )
    parser.add_argument("--seed", type=int, default=2, help="random seed for numpy/torch")

    # env
    parser.add_argument("--env_name", type=str, default="tjc_gym:TrafficJunctionContinuous6-v0")
    parser.add_argument(
        "--render", type=bool, default=True, help="wether the environment should be rendered at each step"
    )

    # buffer
    parser.add_argument(
        "--buffer_size", type=int, default=2000000, help="Max # of transitions that replay buffer can contain"
    )

    # network
    parser.add_argument(
        "--hidden_size1", type=int, default=400, help="Dimension of first hidden layer for actor/critic networks"
    )
    parser.add_argument(
        "--hidden_size2", type=int, default=300, help="Dimension of second hidden layer for actor/critic networks"
    )

    # optimizer
    parser.add_argument("--lr_actor", type=float, default=0.0001, help="learning rate for Adam optimizer")
    parser.add_argument("--lr_critic", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=0)

    # soft update
    parser.add_argument("--tau", type=float, default=0.001, help="Polyak update rate")

    # common algorithm parameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="number of transistions from buffer to train on at once"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for env")

    # exploration
    parser.add_argument(
        "--num_random_episodes",
        type=int,
        default=1,
        help="Number of episodes to add to buffer with purely random actions",
    )
    parser.add_argument("--act_noise_std_start", type=float, default=0.3, help="Action noise standard deviation")

    # train parameters
    parser.add_argument("--episode_length", type=int, default=1000, help="Max length for any episode")
    parser.add_argument(
        "--actor_train_interval_step", type=int, default=1, help="After how many critic updates actor should be updated"
    )
    parser.add_argument(
        "--train_interval_episode", type=int, default=1, help="Number of env steps between updates to actor/critic"
    )
    parser.add_argument(
        "--train_interval", type=int, default=1, help="Number of episodes between updates to actor/critic"
    )

    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Interval in episodes the learned model should be saved",
    )
    parser.add_argument(
        "--running_avg_size",
        type=int,
        default=50,
        help="How many of last episodes to include when calculating average reward. Used to find 'Best model'",
    )

    # continue from pretrained
    parser.add_argument("--model_dir", type=str, default=None)

    return parser
