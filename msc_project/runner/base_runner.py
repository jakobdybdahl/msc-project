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
        self.max_episode_length = self.args.max_episode_length
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.train_interval = self.args.train_interval
        self.save_interval = self.args.save_interval
        self.running_avg_size = self.args.running_avg_size

        self.algorithm_name = self.args.algorithm_name
        self.policy_info = config["policy_info"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        # logging
        self.results = []
        self.best_score = None

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
            from msc_project.algorithms.ddpg.ddpg_agent import DDPGAgent as Agent
        elif self.algorithm_name == "maddpg":
            from msc_project.algorithms.maddpg.maddpg_agent import MADDPGAgent as Agent
        else:
            raise NotImplementedError

        self.agent = Agent(self.args, self.policy_info, self.device)

        # variables used during training
        self.total_env_steps = 0
        self.num_episodes = 0
        self.total_train_steps = 0
        self.last_train_t = 0  # total_env_steps value at last train
        self.last_save_ep = 0  # total_env_steps value at last save

    def run(self):
        # get episode
        env_info = self.run_episode(explore=True, training_episode=True)

        # log info
        self.log(env_info)

        self.num_episodes += 1

        # periodically save
        if (self.num_episodes - self.last_save_ep) / self.save_interval >= 1:
            self.save()
            self.last_save_ep = self.num_episodes

        # TODO evaluate

        # set variables before next run
        self.total_env_steps += env_info["ep_steps"]
        return self.total_env_steps  # self.total_env_steps

    def log(self, env_info):
        steps = env_info["ep_steps"]
        score = env_info["ep_reward"]
        n_collisions = env_info["collisions"]
        n_unique_collisions = env_info["unique_collisions"]
        timedout = env_info["timedout"]
        success = not timedout and n_collisions == 0

        # save stats to results
        self.results.append([self.num_episodes, score, steps, success, n_collisions, n_unique_collisions, timedout])

        # calculate avg score
        max_els = min(len(self.results), self.running_avg_size)
        last_hist = np.array(self.results[-max_els:])
        avg_score = np.mean(last_hist[:, 1])

        # set best score (but first after number of episodes as size of running average)
        if self.num_episodes >= self.running_avg_size:
            if self.best_score == None:
                self.best_score = avg_score
                self.save_best()
            elif avg_score > self.best_score:
                self.best_score = avg_score
                self.save_best()

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
        print(f"Saving models for episode {self.num_episodes}")

        path = self.save_dir + f"/ep_{self.num_episodes}"
        if not os.path.exists(path):
            os.makedirs(path)

        self.agent.save_models(path)

    def save_best(self):
        print(f"New best model found in episode {self.num_episodes}")

        path = self.save_dir + "/best"
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
