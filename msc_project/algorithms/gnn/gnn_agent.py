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
from torch_geometric.data.batch import Batch
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader
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

    def forward(self, data, action, agent_mask):
        x = self.conv1(data.x, data.edge_index)
        x = x[agent_mask, :]
        x = F.relu(x)

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

    def forward(self, data, agent_mask):
        x = self.conv1(data.x, data.edge_index)
        x = x[agent_mask, :]
        x = F.relu(x)

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
        self.batch_size = 128

        self.obs_space = policy_info["obs_space"]
        self.act_space = policy_info["act_space"]

        self.obs_dim = 294
        self.act_dim = 1
        self.n_agents = 6

        self.memory = GNNReplayBuffer(self.buffer_size, self.obs_dim, self.act_dim, self.n_agents)

        decay = (0.3 - 0.01) / 1200000
        self.noise = GaussianActionNoise(mean=0, std=0.3, decay=decay, min_std=0.01)

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
            data = Data(x=x, edge_index=edge_index)

            mu = self.actor.forward(data, i)

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

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # mini_batch = int(self.batch_size / self.n_agents)
        mini_batch = self.batch_size

        fovs, comms, actions, rewards, nfovs, ncomms, dones = self.memory.sample_buffer(mini_batch)

        fovs = T.tensor(fovs, dtype=T.float).to(self.device)
        nfovs = T.tensor(nfovs, dtype=T.float).to(self.device)

        for agent_i in range(self.n_agents):
            a_actions = T.tensor(actions[agent_i], dtype=T.float).to(self.device)
            a_rewards = T.tensor(rewards[agent_i], dtype=T.float).to(self.device)
            a_dones = T.tensor(dones[agent_i]).to(self.device)

            data_list = []
            ndata_list = []

            for b in range(mini_batch):
                a_comm = comms[agent_i][b]
                a_ncomm = ncomms[agent_i][b]

                bnfovs = nfovs[:, b, :].squeeze()
                bfovs = fovs[:, b, :].squeeze()

                edge_index = agent._get_edge_index(a_comm)
                n_edge_index = agent._get_edge_index(a_ncomm)

                data_list.append(Data(x=bfovs, edge_index=edge_index))
                ndata_list.append(Data(x=bnfovs, edge_index=n_edge_index))

            batch = Batch.from_data_list(data_list).to(self.device)
            nbatch = Batch.from_data_list(ndata_list).to(self.device)

            agent_mask = T.zeros_like(batch.batch, dtype=T.bool)
            agent_mask[agent_i : len(agent_mask) : self.n_agents] = True

            target_actions = self.target_actor.forward(nbatch, agent_mask)
            target_critic_value = self.target_critic.forward(nbatch, target_actions, agent_mask)
            critic_value = self.critic.forward(batch, a_actions, agent_mask)

            critic_value[a_dones] = 0.0
            target_critic_value = target_critic_value.view(-1)

            target = a_rewards + self.gamma * target_critic_value
            target = target.view(mini_batch, 1)

            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic.forward(batch, self.actor.forward(batch, agent_mask), agent_mask)
            actor_loss = T.mean(actor_loss)
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

            # if (num_steps + 1) % 10 == 0:
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

        mean_steps = np.mean(ep_info["agent_steps"])
        sum_reward = np.sum(ep_info["agent_rewards"])

        # TODO move into logger?
        print(f"Episode {ep_i}. {ep_info['env_steps']} steps. Noise std: {agent.noise.std}")

        result = "SUCCESS" if not np.any(ep_info["collisions"]) else "FAILED"
        print(f"\t{result} \tCollisions = {ep_info['collisions']} \tScore = {sum_reward}")


def playground():
    positions = [
        # going left
        ((0.575, 0.5375), (-1, 0)),
        ((0.1, 0.5375), (-1, 0)),
        # going right
        ((0.425, 0.4625), (1, 0)),
        # # ((0.1, -0.075), (1, 0)),
        # # going down
        ((0.4625, 0.575), (0, -1)),
        # # ((-0.075, -0.075), (0, -1)),
        # # ((-0.075, -0.3), (0, -1)),
        # # going up
        ((0.5375, 0.425), (0, 1)),
    ]

    agents = env.env._agents

    for i, state in enumerate(positions):
        agents[i].state.on_the_road = True
        agents[i].state.route = 1
        agents[i].state.direction = state[1]
        agents[i].state.position = state[0]

    env.render()

    acts = agent.get_random_actions()
    obs, _, _, _ = env.step(acts)

    for i in range(agent.batch_size):
        acts = agent.get_random_actions()
        nobs, rewards, dones, info = env.step(acts)

        agent.store_transistion(obs, acts, rewards, nobs, dones, info)

        obs = nobs

    agent.store_transistion(obs, acts, rewards, nobs, dones, info)

    for i in range(agent.batch_size):
        agent.store_transistion(obs, acts, rewards, nobs, dones, info)

    batch = agent.memory.sample_buffer(64)

    fovs, comms, action, reward, nfovs, ncomms, done = batch

    fovs = T.tensor(fovs, dtype=T.float).to(device)
    nfovs = T.tensor(nfovs, dtype=T.float).to(device)

    data_list = []
    ndata_list = []

    for b in range(64):
        a_comm = comms[2][b]
        a_ncomm = ncomms[2][b]

        bnfovs = nfovs[:, b, :].squeeze()
        bfovs = fovs[:, b, :].squeeze()

        edge_index = agent._get_edge_index(a_comm)
        n_edge_index = agent._get_edge_index(a_ncomm)

        data_list.append(Data(x=bfovs, edge_index=edge_index))
        ndata_list.append(Data(x=bnfovs, edge_index=n_edge_index))

    batch = Batch.from_data_list(data_list).to(device)
    nbatch = Batch.from_data_list(ndata_list).to(device)

    mask = T.zeros_like(batch.batch, dtype=T.bool)

    mask[1 : len(mask) : num_agents] = True

    test = agent.actor.conv1(x=batch.x, edge_index=batch.edge_index)

    x = test[mask, :]

    # data1 =

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


if __name__ == "__main__":
    seed = 1

    env = gym.make("tjc_gym:TrafficJunctionContinuous6-v0")
    env.env.collision_cost = -10
    env.env.step_cost = -0.01

    env.seed(seed)

    # set seeds
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)

    policy_info = {"obs_space": env.observation_space[0], "act_space": env.action_space[0]}

    agent = GNNAgent(policy_info)

    num_agents = env.n_agents

    # train()

    playground()
