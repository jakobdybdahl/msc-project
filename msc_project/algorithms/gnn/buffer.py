import numpy as np


class GNNReplayBuffer:
    """
    Replay buffer to store samples where the agents are connected in a graph structure.

    Assumes an undirected graph with unweighted edges.
    """

    def __init__(self, max_size, obs_dim, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.n_agents = n_agents

        self.agent_index_memory = np.zeros(self.mem_size, dtype=int)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.connected_with_memory = np.zeros((self.mem_size, n_agents), dtype=int)
        self.next_connected_with_memory = np.zeros((self.mem_size, n_agents), dtype=int)

        self.obs_memory = np.zeros((self.mem_size, self.n_agents, obs_dim), dtype=np.float32)
        self.next_obs_memory = np.zeros((self.mem_size, self.n_agents, obs_dim), dtype=np.float32)

    def store_transition(self, agent_index, obs, conn, action, reward, nobs, nconn, done):
        index = self.mem_cntr % self.mem_size

        self.obs_memory[index] = obs
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_obs_memory[index] = nobs
        self.done_memory[index] = done
        self.connected_with_memory[index] = conn
        self.next_connected_with_memory[index] = nconn
        self.agent_index_memory[index] = agent_index

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        agent_index = self.agent_index_memory[batch]
        obs = self.obs_memory[batch]
        connected_with = self.connected_with_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        nobs = self.next_obs_memory[batch]
        next_connected_with = self.next_connected_with_memory[batch]
        done = self.done_memory[batch] = self.done_memory[batch]

        return agent_index, obs, connected_with, action, reward, nobs, next_connected_with, done
