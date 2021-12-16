import gym
import numpy as np
import torch as T
import torch.nn.functional as F
from msc_project.algorithms.base.trainer import Trainer
from msc_project.algorithms.ddpg.ddpg_policy import DDPGAgent
from msc_project.utils.util import to_torch


class DDPG(Trainer):
    def __init__(self, args, agent, env, device):
        self.args = args
        self.env = env
        self.device = device
        self.num_agents = env.n_agents

        self.agent = agent

        self.hist = []
        self.running_avg_size = 50

    def run_episode(self, episode_i, training_episode=True, explore=True):
        obs = self.env.reset()
        dones = [False] * self.num_agents
        score = 0
        steps = 0
        n_collisions = 0
        n_unique_collisions = 0
        render = True  # TODO make parameter

        while not all(dones):
            if render:
                self.env.render()

            actions = [self.agent.choose_action(obs[i], explore=explore) for i in range(self.num_agents)]

            obs_, rewards, dones, info = self.env.step(actions)

            for i in range(len(obs_)):
                done = dones[i] or not info["cars_on_road"][i]
                self.agent.remember(obs[i], actions[i], rewards[i], obs_[i], done)

            # if total_steps > warm_up_steps and not evaluate:
            if training_episode:
                self.agent.learn()

            obs = obs_

            score += sum(rewards)
            n_unique_collisions += info["unique_collisions"]
            n_collisions += info["collisions"]
            steps += 1

        self.hist.append([episode_i, score, steps, n_unique_collisions])

        last_hist = np.array(self.hist[-self.running_avg_size :])
        avg_score = np.mean(last_hist[:, 1])

        print(f"Episode {episode_i}. {steps} steps. Noise std: {self.agent.noise.std}")
        if n_collisions == 0:
            print(f"\tSUCCES\tScore = {score}\tAvg. score = {avg_score}")
        else:
            print(
                f"\tFAILED\tCollisions = {n_unique_collisions}/{n_collisions} \tScore = {score}\tAvg. score = {avg_score}"
            )

        return steps

    def train_policy_on_batch(self, batch):
        obs, acts, rewards, nobs, dones = batch

        obs = to_torch(obs).to(self.device)
        acts = to_torch(acts).to(self.device)
        rewards = to_torch(rewards).to(self.device)
        nobs = to_torch(nobs).to(self.device)
        dones = to_torch(dones).to(self.device)

        with T.no_grad():
            # TODO noise to target actor action?
            next_actions = self.policy.target_actor.forward(nobs)
            next_q_values = self.policy.target_critic.forward(nobs, next_actions)
            next_q_values[dones] = 0.0
            next_q_values = next_q_values.view(-1)

            target_q_values = rewards * self.gamma * next_q_values

        current_q_values = self.policy.critic.forward(obs, acts).view(-1)

        # critic loss
        self.policy.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        # delayed policy updates
        if self.num_updates % self.actor_update_interval == 0:
            # compute actor loss
            actor_loss = -self.policy.critic.forward(obs, self.policy.actor.forward(obs)).mean()
            self.policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor_optimizer.step()

            # update network parameters
            self.policy.update_networks()

    def prep_rollout(self):
        pass
        # self.policy.actor.eval()
        # self.policy.critic.eval()
        # self.policy.target_actor.eval()
        # self.policy.target_critic.eval()

    def prep_training(self):
        pass
        # self.policy.actor.train()
        # self.policy.critic.train()
        # self.policy.target_actor.train()
        # self.policy.target_critic.train()
