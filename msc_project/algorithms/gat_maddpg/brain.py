import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from msc_project.utils.noise import GaussianActionNoise

from .networks import ActorNetwork, CriticNetwork


class Brain:
    def __init__(self,
                 agent_count,
                 observation_size,
                 action_size,
                 alpha,
                 beta,
                 soft_update_tau,
                 discount_gamma,
                 noise,
                 device):

        self._soft_update_tau = soft_update_tau
        self._gamma = discount_gamma

        # actor networks
        self._actor_local = ActorNetwork(observation_size, action_size, "actor").to(device)
        self._actor_target = ActorNetwork(observation_size, action_size, "actor_target").to(device)

        # critic networks
        self._critic_local = CriticNetwork(observation_size * agent_count, action_size * agent_count, "critic").to(device)
        self._critic_target = CriticNetwork(observation_size * agent_count, action_size * agent_count, "critic_target").to(device)

        # optimizers
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=alpha)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=beta)

        self.noise = noise
        self.device = device
        self.n_agents = agent_count

    def get_actor_model_states(self):
        return self._actor_local.state_dict(), self._actor_target.state_dict()

    def get_critic_model_states(self):
        return self._critic_local.state_dict(), self._critic_target.state_dict()

    def act(self, fov, who_in_fov, agent_mask, target=False, explore=True, train=False):
        """
        :param observation: tensor of shape == (b, observation_size)
        :param target: true to evaluate with target
        :param noise: OU noise factor
        :param train: True for training mode else eval mode
        :return: action: tensor of shape == (b, action_size)
        """

        actor = self._actor_target if target else self._actor_local

        if train:
            actor.train()
        else:
            actor.eval()

        edge_index = self._get_edge_index(who_in_fov)
        data = Data(x=fov, edge_index=edge_index)

        action_values = actor.forward(data, agent_mask)

        noise = 0
        if explore and self.noise is not None:
            noise = torch.tensor(self.noise()).to(self.device)

        return action_values + noise

    def _get_edge_index(self, who_in_fov):
        edge_index = np.zeros((self.n_agents, self.n_agents))
        for agent_i in range(self.n_agents):
            for other_i in np.where(who_in_fov[agent_i] == 1)[0]:
                edge_index.append([agent_i, other_i])
                edge_index.append([other_i, agent_i])

        if len(edge_index) > 0:
            return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        #return torch.tensor([[], []], dtype=torch.long)
        return torch.tensor(edge_index, dtype=torch.long)

    def update_actor(self, all_obs, all_pred_actions, comms, ncomms):
        """
        Actor
        :param all_obs: array of shape == (b, observation_size * n_agents)
        :param all_pred_actions: array of shape == (b, action_size * n_agents)
        :return:
        """
        edge_index = self._get_edge_index(comms)
        data = Data(x=all_obs, edge_index=edge_index)
        actor_loss = -self._critic_local.forward(data, all_pred_actions).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

    def update_critic(self, rewards, dones,
                      all_obs, all_actions, all_next_obs, all_next_actions, comms, ncomms):
        """
        Critic receives observation and actions of all agents as input
        :param rewards: array of shape == (b, 1)
        :param dones: array of shape == (b, 1)
        :param all_obs: array of shape == (b, n_agents, observation_size)
        :param all_actions: array of shape == (b, n_agents, action_size)
        :param all_next_obs:  array of shape == (b, n_agents, observation_size)
        :param all_next_actions: array of shape == (b, n_agents, action_size)
        """

        edge_index = self._get_edge_index(ncomms)
        data = Data(x=all_next_obs, edge_index=edge_index)

        with torch.no_grad():
            q_target_next = self._critic_target.forward(data, all_next_actions)

        q_target = rewards + self._gamma * q_target_next * (1 - dones)

        edge_index = self._get_edge_index(comms)
        data = Data(x=all_obs, edge_index=edge_index)
        q_expected = self._critic_local.forward(data, all_actions)

        critic_loss = F.mse_loss(q_expected, q_target.detach())

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

    def update_targets(self):
        self._soft_update(self._actor_local, self._actor_target, self._soft_update_tau)
        self._soft_update(self._critic_local, self._critic_target, self._soft_update_tau)

    def save_models(self, path=None):
        self._actor_local.save_checkpoint(path)
        self._actor_target.save_checkpoint(path)
        self._critic_local.save_checkpoint(path)
        self._critic_target.save_checkpoint(path)

    def load_models(self, path=None):
        self._actor_local.load_checkpoint(path)
        self._actor_target.load_checkpoint(path)
        self._critic_local.load_checkpoint(path)
        self._critic_target.load_checkpoint(path)

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        :param local_model: model will be copied from
        :param target_model: model will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
