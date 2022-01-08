import os
import time

import numpy as np


class BaseRunner(object):
    def __init__(self, config) -> None:
        """
        Base class for training policies
        :param config: (dict) Dictionary containing parameters for training
        """

        self.args = config["args"]
        self.device = config["device"]

        self.is_eval_run = False
        self.num_epochs = self.args.epochs
        self.episodes_per_epoch = self.args.episodes_per_epoch

        self.max_agent_episode_steps = self.args.max_agent_episode_steps
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.train_interval = self.args.train_interval

        self.save_interval = self.args.save_interval
        self.num_eval_episodes = self.args.num_eval_episodes

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

        # logging
        self.best_score = None
        self.logger = config["logger"]
        self.start_time = time.time()

        # setup trainers and policy dependent on algorithm to be used
        if self.algorithm_name == "ddpg":
            from msc_project.algorithms.ddpg.ddpg_agent import DDPGAgent as Agent
        elif self.algorithm_name == "maddpg":
            from msc_project.algorithms.maddpg.maddpg_agent import MADDPGAgent as Agent
        elif self.algorithm_name == "gnn":
            from msc_project.algorithms.gnn.gnn_agent import GNNAgent as Agent
        elif self.algorithm_name == "gat-maddpg":
            from msc_project.algorithms.gat_maddpg.gat_maddpg_agent import GATMADDPGAgent as Agent
        else:
            raise NotImplementedError

        self.agent = Agent(self.args, self.policy_info, self.device)

        # load model if this is a evaluation run
        if self.args.model_dir is not None:
            self.is_eval_run = True
            self.agent.load_models(self.args.model_dir)

        # variables used during training
        self.total_env_steps = 0
        self.num_episodes = 0
        self.total_train_steps = 0
        self.last_train_t = 0  # total_env_steps value at last train
        self.last_save_ep = 0  # total_env_steps value at last save

    def eval_agent(self):
        raise NotImplementedError

    def run_eval(self):
        print("Running eval epoch..")
        for _ in range(self.episodes_per_epoch):
            self.run_episode(explore=False, training_episode=False)

    def run_epoch(self):
        if self.is_eval_run:
            self.run_eval()
            return

        for _ in range(self.episodes_per_epoch):
            # get episode
            ep_info = self.run_episode(explore=True, training_episode=True)
            self.num_episodes += 1

            # store episode info
            self.store_train_info(ep_info)

            # print
            self.print_between_train_ep(ep_info)

            self.total_env_steps += ep_info["env_steps"]

        # log sampled info during training
        self.log_epoch_info()

        # end of epoch handling
        epoch = self.num_episodes // self.episodes_per_epoch

        # save policy
        if epoch % self.save_interval == 0 or epoch == self.num_epochs:
            self.save_policy()

        self.eval_agent()

        # log general epoch info
        self.logger.log("epoch", epoch)
        self.logger.log("reward", with_min_and_max=True)
        self.logger.log("test_reward", with_min_and_max=True)
        self.logger.log("steps", average_only=True)
        self.logger.log("test_steps", average_only=True)
        self.logger.log("total_env_steps", self.total_env_steps)
        self.logger.log("total_episodes", self.num_episodes)
        self.logger.log("time", time.time() - self.start_time)

        self.logger.dump()

    def run_episode(self, explore=True, training_episode=True, warmup=False):
        raise NotImplementedError

    def store_train_info(self, ep_info):
        raise NotImplementedError

    def print_between_train_ep(self, ep_info):
        raise NotImplementedError

    def log_epoch_info(self):
        raise NotImplementedError

    def learn(self):
        self.agent.learn()

    def save_policy(self):
        print(f"Saving models for episode {self.num_episodes}")

        path = self.save_dir + f"/ep_{self.num_episodes}"
        if not os.path.exists(path):
            os.makedirs(path)

        self.agent.save_models(path)

    def save_best(self):
        # TODO call this method
        print(f"New best model found in episode {self.num_episodes}")

        path = self.save_dir + f"/best_ep_{self.num_episodes}"
        if not os.path.exists(path):
            os.makedirs(path)

        self.agent.save_models(path)

    def load(self):
        print("Loading models..")
        self.policy.load_model()

    def save_results(self):
        header = "episode, reward, steps, success, n_collisions, n_unique_collisions, timedout"
        file = self.run_dir / "results.csv"
        np.savetxt(str(file), self.results, header=header, delimiter=",", fmt="%s", comments="")
