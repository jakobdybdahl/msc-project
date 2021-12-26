import os
import time
from pathlib import Path

import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from msc_project.algorithms.gnn.buffer import GNNReplayBuffer
from msc_project.config import get_config
from msc_project.utils.noise import GaussianActionNoise
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GATv2Conv, GCNConv


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, device, use_gat):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.name = name

        # TODO review this setting of the input channels for the GCNConv layer
        self.hidden_dim = input_dims * 2

        if not use_gat:
            self.conv1 = GCNConv(self.input_dims, self.hidden_dim)
        else:
            self.conv1 = GATv2Conv(self.input_dims, self.hidden_dim, share_weights=True)

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

    def save_checkpoint(self, path):
        file = os.path.join(path, self.name)
        T.save(self.state_dict(), file)

    def load_checkpoint(self, path):
        file = os.path.join(path, self.name)
        self.load_state_dict(T.load(file))

    def save_best(self, path):
        file = os.path.join(path, self.name + "_best")
        T.save(self.state_dict(), file)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, device, use_gat=False):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.name = name

        # TODO review this setting of the input channels for the GCNConv layer
        self.hidden_dim = input_dims * 2

        if not use_gat:
            self.conv1 = GCNConv(self.input_dims, self.hidden_dim)
        else:
            self.conv1 = GATv2Conv(self.input_dims, self.hidden_dim, share_weights=True)

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

    def save_checkpoint(self, path):
        file = os.path.join(path, self.name)
        T.save(self.state_dict(), file)

    def load_checkpoint(self, path):
        file = os.path.join(path, self.name)
        self.load_state_dict(T.load(file))

    def save_best(self, path):
        file = os.path.join(path, self.name + "_best")
        T.save(self.state_dict(), file)


class GNNAgent(object):
    def __init__(self, args, policy_info, device, use_gat=False) -> None:
        self.device = device

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.lr_actor
        self.beta = args.lr_critic
        self.fc1_dimes = args.hidden_size1
        self.fc2_dims = args.hidden_size2

        self.obs_space = policy_info["obs_space"]
        self.act_space = policy_info["act_space"]
        self.obs_dim = policy_info["obs_space"].shape[0]
        self.act_dim = policy_info["act_space"].shape[0]

        self.n_agents = policy_info["num_agents"]
        self.batch_size = args.batch_size

        self.memory = GNNReplayBuffer(args.buffer_size, self.obs_dim, self.act_dim, self.n_agents)

        # TODO remove multiplication of n_agents if get_actions is refactored to only decay noise one time per call
        decay = (args.act_noise_std_start - args.act_noise_std_min) / (args.act_noise_decay_end_step * self.n_agents)
        self.noise = GaussianActionNoise(
            mean=0, std=args.act_noise_std_start, decay=decay, min_std=args.act_noise_std_min
        )

        self.actor = ActorNetwork(
            self.alpha, self.obs_dim, self.fc1_dimes, self.fc2_dims, self.act_dim, "actor", self.device, use_gat
        )
        self.critic = CriticNetwork(
            self.beta, self.obs_dim, self.fc1_dimes, self.fc2_dims, self.act_dim, "critic", self.device, use_gat
        )

        self.target_actor = ActorNetwork(
            self.alpha, self.obs_dim, self.fc1_dimes, self.fc2_dims, self.act_dim, "target_actor", self.device, use_gat
        )
        self.target_critic = CriticNetwork(
            self.beta, self.obs_dim, self.fc1_dimes, self.fc2_dims, self.act_dim, "target_critic", self.device, use_gat
        )

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
        # TODO how to remove this knowledge to the environment?
        fov, who_in_fov = obs[0], obs[1]

        actions = []

        self.actor.eval()
        # TODO can this be done in a batch? Since we are in eval mode
        # might not do any performance difference
        for i in range(self.n_agents):
            # construct edge index of graph for this given agnet
            edge_index = self._get_edge_index(who_in_fov[i]).to(self.device)
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
        # TODO remove knowledge on the environment
        fovs = obs[0]
        comms = obs[1]
        nfovs = nobs[0]
        ncomms = nobs[1]

        for agent_i in range(self.n_agents):
            done = dones[agent_i] or not info["cars_on_road"][agent_i]
            self.memory.store_transition(
                agent_i, fovs, comms[agent_i], acts[agent_i], rewards[agent_i], nfovs, ncomms[agent_i], done
            )

    def save_models(self, path=None):
        self.actor.save_checkpoint(path)
        self.target_actor.save_checkpoint(path)
        self.critic.save_checkpoint(path)
        self.target_critic.save_checkpoint(path)

    def load_models(self, path=None):
        self.actor.load_checkpoint(path)
        self.target_actor.load_checkpoint(path)
        self.critic.load_checkpoint(path)
        self.target_critic.load_checkpoint(path)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        agent_idx, fovs, comms, actions, rewards, nfovs, ncomms, dones = self.memory.sample_buffer(self.batch_size)

        agent_idx = T.tensor(agent_idx, dtype=T.int).to(self.device)
        fovs = T.tensor(fovs, dtype=T.float).to(self.device)
        nfovs = T.tensor(nfovs, dtype=T.float).to(self.device)
        actions = T.tensor(actions, dtype=T.float).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.device)
        dones = T.tensor(dones).to(self.device)

        # build batch graph
        data_list = []
        ndata_list = []

        for b in range(len(agent_idx)):
            comm = comms[b]
            ncomm = ncomms[b]

            edge_index = self._get_edge_index(comm)
            n_edge_index = self._get_edge_index(ncomm)

            data_list.append(Data(x=fovs[b], edge_index=edge_index))
            ndata_list.append(Data(x=nfovs[b], edge_index=n_edge_index))

        batch = Batch.from_data_list(data_list).to(self.device)
        nbatch = Batch.from_data_list(ndata_list).to(self.device)

        agent_mask = T.zeros_like(batch.batch, dtype=T.bool)
        for i, a in enumerate(agent_idx):
            agent_mask[i * self.n_agents + a] = True

        target_actions = self.target_actor.forward(nbatch, agent_mask)
        target_critic_value = self.target_critic.forward(nbatch, target_actions, agent_mask)
        critic_value = self.critic.forward(batch, actions, agent_mask)

        critic_value[dones] = 0.0
        target_critic_value = target_critic_value.view(-1)

        target = rewards + self.gamma * target_critic_value
        target = target.view(self.batch_size, 1)

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

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)


def train():
    render = True
    explore = True

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / "models"

    if not os.path.exists(str(run_dir)):
        os.makedirs(str(run_dir))

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

            if sum(car_rewards) < -3000:
                break

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

        if (ep_i + 1) % 5 == 0:

            agent.save_models(str(run_dir))


def playground():
    obs = env.reset()

    for i in range(agent.batch_size):
        env.render()
        acts = agent.get_random_actions()
        nobs, rewards, dones, info = env.step(acts)

        fovs = obs[0]
        nfovs = nobs[0]

        comms = obs[1]
        ncomms = nobs[1]

        agent.store_transistion(obs, acts, rewards, nobs, dones, info)

        obs = nobs

    batch = agent.memory.sample_buffer(64)

    agent_idx, fovs, comms, action, reward, nfovs, ncomms, done = batch

    fovs = T.tensor(fovs, dtype=T.float).to(device)
    nfovs = T.tensor(nfovs, dtype=T.float).to(device)

    data_list = []
    ndata_list = []

    for b in range(len(agent_idx)):
        comm = comms[b]
        ncomm = ncomms[b]

        edge_index = agent._get_edge_index(comm)
        n_edge_index = agent._get_edge_index(ncomm)

        data_list.append(Data(x=fovs[b], edge_index=edge_index))
        ndata_list.append(Data(x=nfovs[b], edge_index=n_edge_index))

    batch = Batch.from_data_list(data_list).to(device)
    nbatch = Batch.from_data_list(ndata_list).to(device)

    mask = T.zeros_like(batch.batch, dtype=T.bool)

    for i, a in enumerate(agent_idx):
        mask[i * num_agents + a] = True

    test = agent.actor.forward(batch, mask)

    # x = test[mask, :]

    print(agent_idx)
    print(mask)

    print("HELLO")

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
    device = T.device("cuda")

    env = gym.make("tjc_gym:TrafficJunctionContinuous6-v0")
    env.env.collision_cost = -10
    env.env.step_cost = -0.01

    env.seed(seed)

    # set seeds
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)

    num_agents = env.n_agents
    policy_info = {
        "obs_space": gym.spaces.flatten_space(env.observation_space[0]),
        "act_space": env.action_space[0],
        "num_agents": num_agents,
    }

    parser = get_config()
    args = parser.parse_args()

    args.buffer_size = int(5e5)

    agent = GNNAgent(args, policy_info, device, use_gat=True)

    train()

    # playground()
