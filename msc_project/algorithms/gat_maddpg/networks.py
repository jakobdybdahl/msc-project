import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

import os

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNetwork(nn.Module):

    def __init__(self, observation_size, action_size, name):
        """
        :param observation_size: observation size
        :param action_size: action size
        """
        super(ActorNetwork, self).__init__()

        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        # Dynamic number of hidden neurons to handle for varying observation space.
        fc1_units = np.floor((observation_size + action_size) / 2).astype(int)
        fc2_units = np.floor(fc1_units / 2).astype(int)

        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)

        self.conv1 = GATv2Conv(observation_size, fc1_units, share_weights=True)

        # batch norm has bias included, disable linear layer bias
        self.fc1 = nn.Linear(fc1_units, fc1_units, bias=False)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=False)
        self.fc3 = nn.Linear(fc2_units, action_size, bias=False)

        self.name = name

        self.reset_parameters()

    def forward(self, data, agent_mask):
        """ map a states to action values
        :param observation: shape == (batch, observation_size)
        :return: action values
        """
        x = self.conv1(data.x, data.edge_index) # 1, 294 -- 2, 0
        #x = F.relu(x_conv[agent_mask, :])
        
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = F.relu(self.fc3(self.bn3(x)))
        return torch.sigmoid(x)


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def save_checkpoint(self, path):
        file = os.path.join(path, self.name)
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, path):
        file = os.path.join(path, self.name)
        self.load_state_dict(torch.load(file))

    def save_best(self, path):
        file = os.path.join(path, self.name + "_best")
        torch.save(self.state_dict(), file)


class CriticNetwork(nn.Module):

    def __init__(self, observation_size, action_size, name):
        """
        :param observation_size: Dimension of each state
        :param action_size: Dimension of each state
        """
        super(CriticNetwork, self).__init__()

        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        # Dynamic number of hidden neurons to handle for varying observation space.
        fc1_units = np.floor((observation_size + action_size) / 2).astype(int)
        fc2_units = np.floor(fc1_units / 2).astype(int)

        self.bn1 = nn.BatchNorm1d(fc1_units + action_size)
        self.bn2 = nn.BatchNorm1d(fc1_units)

        self.conv1 = GATv2Conv(observation_size, fc1_units, share_weights=True)

        self.fc1 = nn.Linear(fc1_units + action_size, fc1_units, bias=False)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.name = name

        self.reset_parameters()

    def forward(self, data, action):
        """ map (observation, actions) pairs to Q-values
        :param observation: shape == (batch, observation_size)
        :param action: shape == (batch, action_size)
        :return: q-values values
        """
        x = self.conv1(data.x, data.edge_index) # 128, 885
        #x = F.relu(x[agent_mask, :])

        x = torch.cat([x, action], dim=1) # 128, 891
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = self.fc3(x)

        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def save_checkpoint(self, path):
        file = os.path.join(path, self.name)
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, path):
        file = os.path.join(path, self.name)
        self.load_state_dict(torch.load(file))

    def save_best(self, path):
        file = os.path.join(path, self.name + "_best")
        torch.save(self.state_dict(), file)
