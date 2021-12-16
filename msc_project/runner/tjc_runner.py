import numpy as np
import torch as T
from msc_project.runner.base_runner import BaseRunner


class TJCRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config)

        # TODO warm up - remove?
        num_warmup_eps = max((int(self.batch_size // self.episode_length) + 1, self.args.num_random_episodes))
        # self.warmup(num_warmup_eps)

    def run_episode(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        num_collisions = 0
        num_unique_collisions = 0
        num_steps = 0
        episode_rewards = []

        # TODO get from args?
        render = True

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
            episode_rewards.append(rewards)
            num_collisions += info["collisions"]
            num_unique_collisions += info["unique_collisions"]
            num_steps += 1

        ep_reward = np.sum(episode_rewards)
        env_info["ep_reward"] = ep_reward
        env_info["ep_steps"] = num_steps
        env_info["collisions"] = num_collisions
        env_info["unique_collisions"] = num_unique_collisions

        return env_info

    @T.no_grad()
    def warmup(self, num_warmup_eps):
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("Warm up...")
        for _ in range(num_warmup_eps):
            env_info = self.run_episode(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info["avg_ep_reward"])
        warmup_reward = np.mean(warmup_rewards)
        print(f"Average reward during warm up: {warmup_reward}")
