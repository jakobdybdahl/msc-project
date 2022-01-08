import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.bn1 = nn.BatchNorm1d(observation_size)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)

        # batch norm has bias included, disable linear layer bias
        self.fc1 = nn.Linear(observation_size, fc1_units, bias=False)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=False)
        self.fc3 = nn.Linear(fc2_units, action_size, bias=False)

        self.name = name

        self.reset_parameters()

    def forward(self, observation):
        """ map a states to action values
        :param observation: shape == (batch, observation_size)
        :return: action values
        """

        x = F.relu(self.fc1(self.bn1(observation)))
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
        self.fc1_dims = np.floor((observation_size + action_size) / 2).astype(int)
        self.fc2_dims = np.floor(self.fc1_dims / 2).astype(int)

        # self.bn1 = nn.BatchNorm1d(observation_size + action_size)
        # self.bn2 = nn.BatchNorm1d(fc1_units)

        # self.fc1 = nn.Linear(observation_size + action_size, fc1_units, bias=False)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, 1)

        self.name = name

        self.fc1 = nn.Linear(observation_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(action_size, self.fc2_dims)

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

        #self.reset_parameters()

    # def forward(self, observation, action):
    #     x = torch.cat([observation, action], dim=1)
    #     x = F.relu(self.fc1(self.bn1(x)))
    #     x = F.relu(self.fc2(self.bn2(x)))
    #     x = self.fc3(x)
    #     return x
    
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        state_action_value = F.relu(torch.add(state_value, action_value))

        state_action_value = self.q(state_action_value)

        return state_action_value

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
