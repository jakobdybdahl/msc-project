import os

import numpy as np


class BaseRunner(object):
    def __init__(self, config) -> None:
        """
        Base class for training policies
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

        self.running_avg_size = 40
        self.avg_buffer_size = self.running_avg_size
        self.avg_buffer_idx_cntr = 0
        self.avg_buffer = np.zeros(self.avg_buffer_size)

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
            from msc_project.algorithms.ddpg.ddpg import DDPGAgent as Agent
        elif self.algorithm_name == "maddpg":
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.agent = Agent(self.args, self.policy_info, self.device)

        self.trainer = Trainer(self.args, self.agent, self.env, self.device)

        # variables used during training
        self.total_env_steps = 0
        self.num_episodes = 0
        self.total_train_steps = 0
        self.last_train_t = 0  # total_env_steps value at last train
        self.last_save_t = 0  # total_env_steps value at last save

    def run(self):
        env_info = self.run_episode(explore=True, training_episode=True)

        self.log(env_info)

        self.total_env_steps += env_info["ep_steps"]
        self.num_episodes += 1
        return self.total_env_steps  # self.total_env_steps

    def log(self, env_info):
        steps = env_info["ep_steps"]
        score = env_info["ep_reward"]
        n_collisions = env_info["collisions"]
        n_unique_collisions = env_info["unique_collisions"]

        # keep track of last eps avg score
        idx = self.avg_buffer_idx_cntr % self.avg_buffer_size
        self.avg_buffer[idx] = env_info["ep_reward"]
        self.avg_buffer_idx_cntr += 1

        # calculate avg score
        max_els = min(self.avg_buffer_idx_cntr, self.avg_buffer_size)
        avg_score = self.avg_buffer[0:max_els].mean()

        print(f"Episode {self.num_episodes}. {steps} steps. Noise std: {self.agent.noise.std}")
        if n_collisions == 0:
            print(f"\tSUCCES\tScore = {score}\tAvg. score = {avg_score}")
        else:
            print(
                f"\tFAILED\tCollisions = {n_unique_collisions}/{n_collisions} \tScore = {score}\tAvg. score = {avg_score}"
            )

    def learn(self):
        self.agent.learn()

    def save(self):
        print("Saving models..")
        self.policy.save_model()

    def load(self):
        print("Loading models..")
        self.policy.load_model()
