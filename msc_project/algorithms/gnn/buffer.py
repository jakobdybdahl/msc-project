import numpy as np


class GNNReplayBuffer:
    """
    Replay buffer to store samples where the agents are connected in a graph structure.

    Assumes an undirected graph with unweighted edges.
    """

    def __init__(self, max_size, input_shape, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.obs_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.next_obs_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.connected_with_memory = np.zeros((self.mem_size, n_agents), dtype=int)
        self.next_connected_with_memory = np.zeros((self.mem_size, n_agents), dtype=int)

    def store_transition(self, obs, action, reward, nobs, done):
        index = self.mem_cntr % self.mem_size

        fov, comm = obs[0], obs[1]
        nfov, ncomm = nobs[0], nobs[1]

        self.obs_memory[index] = fov
        self.connected_with_memory[index] = comm
        self.next_obs_memory[index] = nfov
        self.next_connected_with_memory[index] = ncomm

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        fov = self.obs_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        nfov = self.next_obs_memory[batch]
        dones = self.terminal_memory[batch]

        comm = self.connected_with_memory[batch]
        ncomm = self.next_connected_with_memory[batch]

        return (fov, comm), actions, rewards, (nfov, ncomm), dones
