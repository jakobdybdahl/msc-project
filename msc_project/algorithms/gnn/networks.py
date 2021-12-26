import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, GCNConv


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, device, use_gat):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.name = name

        # TODO review this setting of the input channels for the GNN layer
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


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, device, use_gat=False):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.name = name

        # TODO review this setting of the input channels for the GNN layer
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
