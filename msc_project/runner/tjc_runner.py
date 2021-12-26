import sys
from os import system

import numpy as np
import torch as T
from msc_project.runner.base_runner import BaseRunner


class TJCRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config)

        # warmup
        num_warmup_eps = max((int(self.batch_size // self.max_agent_episode_steps) + 1, self.args.num_random_episodes))
        self.warmup(num_warmup_eps)

    def run_episode(self, explore=True, training_episode=True, warmup=False):
        ep_info = {}
        num_collisions = 0
        num_unique_collisions = 0
        num_steps = 0

        car_steps = np.zeros(self.num_agents)
        car_rewards = np.zeros(self.num_agents)

        render = self.args.render

        env = self.env
        obs = env.reset()
        dones = [False] * self.num_agents

        while not all(dones):
            if render:
                env.render()

            if warmup:
                actions = self.agent.get_random_actions(obs)
            else:
                actions = self.agent.get_actions(obs, explore=explore)

            nobs, rewards, dones, info = env.step(actions)

            if explore:
                self.agent.store_transistion(obs, actions, rewards, nobs, dones, info)

            obs = nobs

            if training_episode:
                self.total_env_steps += 1
                if self.last_train_t == 0 or ((self.total_env_steps - self.last_train_t) / self.train_interval) >= 1:
                    self.learn()
                    self.total_train_steps += 1
                    self.last_train_t = self.total_env_steps

            # update env info vars
            car_rewards += rewards
            num_collisions += info["collisions"]
            num_unique_collisions += info["unique_collisions"]
            car_steps += info["took_step"]
            num_steps += 1

        ep_info["env_steps"] = num_steps
        ep_info["agent_rewards"] = car_rewards
        ep_info["agent_steps"] = car_steps
        ep_info["collisions"] = num_collisions
        ep_info["unique_collisions"] = num_unique_collisions

        return ep_info

    def eval_agent(self):
        num_success = 0
        num_timeout = 0
        num_collision = 0

        for _ in range(self.num_eval_episodes):
            ep_info = self.run_episode(explore=False, training_episode=False)

            timeout = (ep_info["agent_steps"] >= self.max_agent_episode_steps).any()
            collision = ep_info["collisions"].any()

            num_timeout += 1 if timeout else 0
            num_collision += 1 if collision else 0
            num_success += 1 if not timeout and not collision else 0

            mean_steps = np.mean(ep_info["agent_steps"])
            sum_reward = np.sum(ep_info["agent_rewards"])
            self.logger.store(test_steps=mean_steps, test_reward=sum_reward)

        success_rate = num_success / self.num_eval_episodes

        # TODO compare with previous best score and save model if better

        self.logger.log("success_rate", success_rate)
        self.logger.log("num_timeout", num_timeout)
        self.logger.log("num_collision", num_collision)

    def log(self, ep_info):
        mean_steps = np.mean(ep_info["agent_steps"])
        sum_reward = np.sum(ep_info["agent_rewards"])

        self.logger.store(steps=mean_steps, reward=sum_reward)

        # TODO move into logger?
        print(f"Episode {self.num_episodes}. {ep_info['env_steps']} steps. Noise std: {self.agent.noise.std}")

        result = "SUCCESS" if not np.any(ep_info["collisions"]) else "FAILED"
        print(f"\t{result} \tCollisions = {ep_info['collisions']} \tScore = {sum_reward}")

        sys.stdout.flush()

    def warmup(self, num_warmup_eps):
        warmup_rewards = []
        print("Warm up...")
        for _ in range(num_warmup_eps):
            ep_info = self.run_episode(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(np.sum(ep_info["agent_rewards"]))
        warmup_reward = np.mean(warmup_rewards)
        print(f"Average reward during warm up: {warmup_reward}")
