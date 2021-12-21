import time

import gym
import networkx as nx
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from msc_project.algorithms.gnn.buffer import GNNReplayBuffer
from msc_project.utils.noise import GaussianActionNoise
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.hidden_dim = input_dims * 2

        self.conv1 = GCNConv(self.input_dims, self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)

        self.device = device
        self.to(self.device)

    def forward(self, state, action, edge_index, agent_i):
        x = self.conv1(state, edge_index)
        x = F.relu(x[agent_i])

        state_value = self.fc1(x)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))

        state_action_value = self.q(state_action_value)

        return state_action_value


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.hidden_dim = input_dims * 2

        self.conv1 = GCNConv(self.input_dims, self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = device
        self.to(self.device)

    def forward(self, state, edge_index, agent_i):
        x = self.conv1(state, edge_index)
        x = F.relu(x[agent_i])

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.sigmoid(self.mu(x))

        return x


class GNNAgent(object):
    def __init__(self, policy_info) -> None:
        self.device = T.device("cuda")

        self.tau = 0.001
        self.gamma = 0.99
        self.alpha = 0.0001
        self.beta = 0.001

        self.buffer_size = int(5e5)
        self.batch_size = 32

        self.obs_space = policy_info["obs_space"]
        self.act_space = policy_info["act_space"]

        self.obs_dim = 294
        self.act_dim = 1
        self.n_agents = 6

        self.memory = GNNReplayBuffer(self.buffer_size, self.obs_dim, self.act_dim, self.n_agents)
        self.noise = GaussianActionNoise(mean=0, std=0.3, decay=1e-5, min_std=0.01)

        self.actor = ActorNetwork(self.alpha, self.obs_dim, 400, 300, self.act_dim, self.device)
        self.critic = CriticNetwork(self.beta, self.obs_dim, 400, 300, self.act_dim, self.device)

        self.target_actor = ActorNetwork(self.alpha, self.obs_dim, 400, 300, self.act_dim, self.device)
        self.target_critic = CriticNetwork(self.beta, self.obs_dim, 400, 300, self.act_dim, self.device)

        self.update_network_parameters(tau=1)

    def get_random_actions(self):
        return [self.act_space.sample()[0] for _ in range(self.n_agents)]

    def _get_edge_index(self, who_in_fov):
        edge_index = []
        for agent_i in range(self.n_agents):
            for other_i in np.where(who_in_fov[agent_i] == 1)[0]:
                edge_index.append([agent_i, other_i])
                edge_index.append([other_i, agent_i])

        if len(edge_index) > 0:
            return T.tensor(edge_index, dtype=T.long).t().contiguous()

        return T.tensor([[], []], dtype=T.long)

    def get_actions(self, obs, explore=True):
        fov, who_in_fov = obs[0], obs[1]

        actions = []

        self.actor.eval()
        for i in range(self.n_agents):
            # construct edge index of graph for this given agnet
            edge_index = agent._get_edge_index(who_in_fov[i]).to(self.device)
            x = T.tensor(fov, dtype=T.float).to(self.device)
            # data = Data(x=x, edge_index=edge_index).to(self.device)

            mu = self.actor.forward(x, edge_index, i)

            if self.noise != None and explore:
                noise = T.tensor(self.noise(mu.shape)).to(self.actor.device)
                mu_prime = T.add(mu, noise).to(self.device)
            else:
                mu_prime = mu
            mu_prime.clamp_(0, 1)  # inplace clamp

            action = mu_prime.view(-1).cpu().detach().numpy()[0]
            actions.append(action)
        self.actor.train()

        return actions

    def store_transistion(self, obs, acts, rewards, nobs, dones, info):
        fov = obs[0]
        comm = obs[1]
        nfov = nobs[0]
        ncomm = nobs[1]

        done = [dones[i] or not info["cars_on_road"][i] for i in range(len(dones))]

        self.memory.store_transition(fov, comm, acts, rewards, nfov, ncomm, done)

        # for i in range(self.n_agents):
        #     done = dones[i] or not info["cars_on_road"][i]
        #     self.memory.store_transition(fov[i], comm[i], acts[i], rewards[i], nfov[i], ncomm[i], done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        fovs, comms, actions, rewards, nfovs, ncomms, dones = self.memory.sample_buffer(self.batch_size)

        fovs = T.tensor(fovs, dtype=T.float).to(self.device)
        nfovs = T.tensor(nfovs, dtype=T.float).to(self.device)

        for agent_i in range(self.n_agents):
            a_comm = comms[agent_i]
            a_ncomm = ncomms[agent_i]
            a_actions = T.tensor(actions[agent_i], dtype=T.float).to(self.device)
            a_rewards = T.tensor(rewards[agent_i], dtype=T.float).to(self.device)
            a_dones = T.tensor(dones[agent_i]).to(self.device)

            critic_values = T.empty(self.batch_size)
            targets = T.empty(self.batch_size)

            for b in range(self.batch_size):
                edge_index = self._get_edge_index(a_comm[b]).to(self.device)
                next_edge_index = self._get_edge_index(a_ncomm[b]).to(self.device)

                bnfovs = nfovs[:, b, :].squeeze()
                bfovs = fovs[:, b, :].squeeze()

                critic_value = self.critic.forward(bfovs, a_actions[b], edge_index, agent_i)

                with T.no_grad():
                    target_action = self.target_actor.forward(bnfovs, next_edge_index, agent_i)
                    target_critic_value = self.target_critic.forward(bnfovs, target_action, next_edge_index, agent_i)
                    target_critic_value = 0.0 if a_dones[b] else target_critic_value
                    target = a_rewards[b] + self.gamma * target_critic_value

                targets[b] = target
                critic_values[b] = critic_value

            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(targets, critic_values)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.actor.optimizer.zero_grad()
            actor_losses = T.empty(self.batch_size)
            for b in range(self.batch_size):
                edge_index = self._get_edge_index(a_comm[b]).to(self.device)
                bnfovs = nfovs[:, b, :].squeeze()
                action = self.actor.forward(bnfovs, edge_index, agent_i)
                actor_loss = -self.critic.forward(bnfovs, action, edge_index, agent_i)
                actor_losses[b] = actor_loss
            actor_loss = T.mean(actor_losses)
            actor_loss.backward()
            self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)


def train():
    render = True
    explore = True

    for ep_i in range(1000):
        ep_info = {}
        num_collisions = 0
        num_unique_collisions = 0
        num_steps = 0

        car_steps = np.zeros(num_agents)
        car_rewards = np.zeros(num_agents)

        obs = env.reset()
        dones = [False] * num_agents

        while not all(dones):
            if render:
                env.render()

            actions = agent.get_actions(obs, explore=explore)

            nobs, rewards, dones, info = env.step(actions)

            if explore:
                agent.store_transistion(obs, actions, rewards, nobs, dones, info)

            obs = nobs

            agent.learn()

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

        print(ep_info)


if __name__ == "__main__":
    seed = 1

    env = gym.make("tjc_gym:TrafficJunctionContinuous6-v0")

    env.seed(seed)

    # set seeds
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)

    policy_info = {"obs_space": env.observation_space[0], "act_space": env.action_space[0]}

    agent = GNNAgent(policy_info)

    num_agents = env.n_agents

    # train()

    # obs = env.reset()
    # acts = agent.get_random_actions()
    # nobs, rewards, dones, info = env.step(acts)

    # agent.store_transistion(obs, acts, rewards, nobs, dones, info)

    # for i in range(agent.batch_size):
    #     agent.store_transistion(obs, acts, rewards, nobs, dones, info)

    # batch = agent.memory.sample_buffer(64)
    # obs, connected_with, action, reward, nobs, next_connected_with, done = batch

    # acts = agent.get_actions(nobs)

    # fov, who_in_fov = nobs[0], nobs[1]

    # edge_index = agent._get_edge_index(who_in_fov[0])
    # x = T.tensor(fov, dtype=T.float)
    # data = Data(x=x, edge_index=edge_index)

    # print(data.num_node_features)

    # agent.store_transistion()

    # import matplotlib.pyplot as plt

    # plt.figure(1, figsize=(14, 12))
    # nx.draw(to_networkx(data), cmap=plt.get_cmap("Set1"), with_labels=True, node_size=75, linewidths=6)
    # plt.show()

    time.sleep(10000)
