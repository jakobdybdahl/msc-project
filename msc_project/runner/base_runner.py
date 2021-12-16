import os

from msc_project.utils.buffer import PolicyBuffer
from numpy import exp
from torch.functional import align_tensors


class Runner(object):
    def __init__(self, config) -> None:
        """
        Base class for training MLP policies
        :param config: (dict) Dictionary containing parameters for training
        """

        self.args = config["args"]
        self.device = config["device"]

        self.num_env_steps = self.args.num_env_steps
        self.episode_length = self.args.episode_length
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.train_interval = self.args.train_interval
        self.save_interval = self.args.save_interval

        self.algorithm_name = self.args.algorithm_name
        self.policy_info = config["policy_info"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        # TODO parralel envs?
        self.env = config["env"]
        self.env_name = self.args.env_name

        # dir
        self.run_dir = config["run_dir"]
        self.save_dir = f"{self.run_dir}/models"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # setup trainers and policy dependent on algorithm to be used
        if self.algorithm_name == "ddpg":
            from msc_project.algorithms.ddpg.ddpg import DDPG as Trainer
        elif self.algorithm_name == "maddpg":
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.trainer = Trainer(self.args, self.env, self.device)

        # variables used during training
        self.total_env_steps = 0
        self.num_episodes = 0
        self.total_train_steps = 0
        self.last_train_t = 0  # total_env_steps value at last train
        self.last_save_t = 0  # total_env_steps value at last save

    def run(self):
        ep_env_steps = self.trainer.run_episode(self.num_episodes, training_episode=True, explore=True)

        self.total_env_steps += ep_env_steps
        self.num_episodes += 1

        return self.total_env_steps  # self.total_env_steps

    def save(self):
        print("Saving models..")
        self.policy.save_model()

    def load(self):
        print("Loading models..")
        self.policy.load_model()
