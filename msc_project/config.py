import argparse


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm_name", type=str, default="ddpg", choices=["maddpg", "ddpg"])
    parser.add_argument("--experiment_name", type=str, default="debug")
    parser.add_argument("--seed", type=int, default=2, help="random seed for numpy/torch")

    # env
    parser.add_argument("--env_name", type=str, default="tjc_gym:TrafficJunctionContinuous6-v0")
    parser.add_argument(
        "--render",
        type=bool,
        dest="render",
        default=False,
        help="wether the environment should be rendered at each step",
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
        default=0,
        help="Number of episodes to add to buffer with purely random actions",
    )
    parser.add_argument(
        "--act_noise_std_start", type=float, default=0.3, help="Start value for action noise standard deviation"
    )
    parser.add_argument(
        "--act_noise_std_min",
        type=float,
        default=0.0,
        help="Min value for action noise standard deviation. Decays to this value at 'act_noise_decay_end_step'",
    )
    parser.add_argument(
        "--act_noise_decay_end_step",
        type=int,
        default=200000,
        help="Number of environment steps where noise should be added to actions",
    )

    # train parameters
    parser.add_argument(
        "--max_agent_episode_steps", type=int, default=500, help="Max number of steps for a single agent in an episode"
    )
    parser.add_argument(
        "--actor_train_interval_step", type=int, default=1, help="After how many critic updates actor should be updated"
    )
    parser.add_argument("--train_interval", type=int, default=1, help="Number of steps between updates to actor/critic")
    parser.add_argument("--episodes_per_epoch", type=int, default=15, help="Number of episodes in each epoch")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to run and train agent")

    # evaulate parameters
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate the policy at the end of each epoch",
    )

    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Interval in epochs the learned model should be saved",
    )
    parser.add_argument(
        "--running_avg_size",
        type=int,
        default=50,
        help="How many of last episodes to include when calculating average reward. Used to find 'Best model'",
    )

    # load existing model TODO - use somewhere
    parser.add_argument("--model_dir", type=str, default=None, help="Path to directory with a 'models' folder")

    return parser
