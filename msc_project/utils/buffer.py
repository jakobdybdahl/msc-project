import numpy as np


class PolicyBuffer(object):
    """
    Buffer containing data corresponding to a single policy.

    :param num_agents: (int) number of agents controlled by the policy
    """

    def __init__(self, buffer_size, num_agents, obs_space_dim, act_space_dim):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_space_dim
        self.act_dim = act_space_dim

        self.mem_cntr = 0

        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.nobs = np.zeros_like(self.obs, dtype=np.float32)

        self.acts = np.zeros((self.buffer_size, self.num_agents, self.act_dim), dtype=np.float32)

        self.rewards = np.zeros(
            (
                self.buffer_size,
                self.num_agents,
            ),
            dtype=np.float32,
        )

        self.dones = np.ones_like(self.rewards, dtype=bool)

        # TODO keep track of when env returns done?

    def insert(self, obs, acts, rewards, nobs, dones):
        index = self.mem_cntr % self.buffer_size

        self.obs[index] = obs.copy()
        self.acts[index] = acts.copy()
        self.rewards[index] = rewards.copy()
        self.nobs[index] = nobs.copy()
        self.dones[index] = dones.copy()

        self.mem_cntr += 1

        return index

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.buffer_size)

        # batch of time step values for every agent in env
        batch = np.random.choice(max_mem, batch_size)
        obs = self.obs[batch]
        acts = self.acts[batch]
        rewards = self.rewards[batch]
        nobs = self.nobs[batch]
        dones = self.dones[batch]

        return obs, acts, rewards, nobs, dones
