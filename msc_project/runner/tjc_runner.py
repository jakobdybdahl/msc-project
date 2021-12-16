import numpy as np
import torch as T
from msc_project.runner.base_runner import MlpRunner


class TJCRunner(MlpRunner):
    def __init__(self, config) -> None:
        super().__init__(config)

        # TODO warm up - remove?
        num_warmup_eps = max((int(self.batch_size // self.episode_length) + 1, self.args.num_random_episodes))
        # self.warmup(num_warmup_eps)

    def run_episode(self, explore=True, training_episode=True, warmup=False):
        env = self.env
        obs = env.reset()

        env_info = {}
        episode_rewards = []

        render = True

        for step in range(self.episode_length):
            if render:
                env.render()

            if warmup:
                actions = self.policy.get_random_actions(obs)
            else:
                actions = self.policy.get_actions(obs, explore=explore)

            nobs, rewards, dones, info = env.step(actions)

            episode_rewards.append(rewards)

            if explore:
                for i in range(self.num_agents):
                    self.buffer.insert(obs[i], actions[i], rewards[i], nobs[i], dones[i])

            obs = nobs

            if training_episode:
                self.total_env_steps += 1
                if self.last_train_t == 0 or ((self.total_env_steps - self.last_train_t) / self.train_interval) >= 1:
                    self.batch_train()
                    self.total_train_steps += 1
                    self.last_train_t = self.total_env_steps

            env_done = np.all(dones)
            if env_done:
                break

        avg_ep_reward = np.mean(np.sum(episode_rewards, axis=0))
        env_info["avg_ep_reward"] = avg_ep_reward

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
