import sys

import numpy as np
from msc_project.runner.base_runner import BaseRunner


class TJCRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.best_success_rate = -np.inf

        # init values to store for epoch logging
        self._init_train_values()

        # warmup
        if not self.is_eval_run:
            num_warmup_eps = max(
                (int(self.batch_size // self.max_agent_episode_steps) + 1, self.args.num_random_episodes)
            )
            self.warmup(num_warmup_eps)

    def _init_train_values(self):
        self.train_num_success = 0
        self.train_num_timeout_failures = 0
        self.train_num_collision_failures = 0
        self.train_num_minimum_reward_reached = 0

        self.train_total_collisions = 0
        self.train_total_timeouts = 0

    def run_episode(self, explore=True, training_episode=True, warmup=False):
        ep_info = {}
        num_collisions = 0
        num_unique_collisions = 0
        num_steps = 0
        minimum_reward_reached = False

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
                actions = self.agent.get_random_actions()
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

            if self.args.min_reward != None:
                if sum(car_rewards) < self.args.min_reward:
                    # terminate episode
                    dones = [True] * self.num_agents
                    minimum_reward_reached = True

        ep_info["env_steps"] = num_steps
        ep_info["agent_rewards"] = car_rewards
        ep_info["agent_steps"] = car_steps
        ep_info["collisions"] = num_collisions
        ep_info["unique_collisions"] = num_unique_collisions
        ep_info["minimum_reward_reached"] = minimum_reward_reached

        return ep_info

    def eval_agent(self):
        num_success = 0
        num_timeout_failures = 0
        num_collision_failures = 0
        num_minimum_reward_reached = 0

        total_collisions = 0
        total_timeouts = 0

        for _ in range(self.num_eval_episodes):
            ep_info = self.run_episode(explore=False, training_episode=False)

            result = self._get_ep_result(ep_info)

            num_minimum_reward_reached += 1 if result["minimum_reward_reached"] else 0
            num_timeout_failures += 1 if result["num_timeouts"] > 0 else 0
            num_collision_failures += 1 if result["num_collisions"] > 0 else 0
            num_success += 1 if result["success"] else 0

            total_collisions += result["num_collisions"]
            total_timeouts += result["num_timeouts"]

            mean_steps = result["mean_agent_steps"]
            sum_reward = result["sum_reward"]
            self.logger.store(test_steps=mean_steps, test_reward=sum_reward)

        success_rate = num_success / self.num_eval_episodes

        # check if better than previous and save if so
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.save_best()

        self.logger.log("success_rate", success_rate)
        self.logger.log("num_timeout_failures", num_timeout_failures)
        self.logger.log("num_collision_failures", num_collision_failures)
        self.logger.log("num_minimum_reward_reached", num_minimum_reward_reached)
        self.logger.log("total_collisions", total_collisions)
        self.logger.log("total_timeouts", total_timeouts)

    def log_epoch_info(self):
        success_rate = self.train_num_success / self.episodes_per_epoch

        self.logger.log("train_success_rate", success_rate)
        self.logger.log("train_num_timeout_failures", self.train_num_timeout_failures)
        self.logger.log("train_num_collisions_failures", self.train_num_collision_failures)
        self.logger.log("train_num_minimum_reward_reached", self.train_num_minimum_reward_reached)
        self.logger.log("train_total_collisions", self.train_total_collisions)
        self.logger.log("train_total_timeouts", self.train_total_timeouts)
        self.logger.log("agent_noise_std", self.agent.noise.std)

        # reset values
        self._init_train_values()

    def store_train_info(self, ep_info):
        result = self._get_ep_result(ep_info)
        self.logger.store(steps=result["mean_agent_steps"], reward=result["sum_reward"])

        # store local
        self.train_num_minimum_reward_reached += 1 if result["minimum_reward_reached"] else 0
        self.train_num_timeout_failures += 1 if result["num_timeouts"] > 0 else 0
        self.train_num_collision_failures += 1 if result["num_collisions"] > 0 else 0
        self.train_num_success += 1 if result["success"] else 0

        self.train_total_collisions += result["num_collisions"]
        self.train_total_timeouts += result["num_timeouts"]

    def print_between_train_ep(self, ep_info):
        ep_result = self._get_ep_result(ep_info)

        print(f"Episode {self.num_episodes}. {ep_result['env_steps']} steps. Noise std: {self.agent.noise.std}")

        result = "SUCCESS" if ep_result["success"] else "FAILED"
        print(f"\t{result} \tCollisions = {ep_result['num_collisions']} \tScore = {ep_result['sum_reward']}")

        sys.stdout.flush()

        sys.stdout.flush()

    def warmup(self, num_warmup_eps):
        warmup_rewards = []
        print("Warm up...")
        for _ in range(num_warmup_eps):
            ep_info = self.run_episode(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(np.sum(ep_info["agent_rewards"]))
        warmup_reward = np.mean(warmup_rewards)
        print(f"Average reward during warm up: {warmup_reward}")

    def _get_ep_result(self, ep_info):
        num_timeouts = (ep_info["agent_steps"] >= self.max_agent_episode_steps).sum()
        num_collisions = ep_info["collisions"].sum()
        sum_reward = ep_info["agent_rewards"].sum()
        mean_agent_steps = ep_info["agent_steps"].mean()

        minimum_reward_reached_failure = ep_info["minimum_reward_reached"]
        timeout_failure = num_timeouts > 0
        collision_failure = num_collisions > 0

        success = not minimum_reward_reached_failure and not timeout_failure and not collision_failure

        return {
            "success": success,
            "num_timeouts": num_timeouts,
            "num_collisions": num_collisions,
            "sum_reward": sum_reward,
            "minimum_reward_reached": minimum_reward_reached_failure,
            "env_steps": ep_info["env_steps"],
            "mean_agent_steps": mean_agent_steps,
        }
