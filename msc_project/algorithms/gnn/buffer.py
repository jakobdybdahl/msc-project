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

        self.obs_memory = []
        self.next_obs_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.connected_with_memory = []
        self.next_connected_with_memory = []

        for i in range(self.n_agents):
            self.obs_memory.append(np.zeros((self.mem_size, obs_dim), dtype=np.float))
            self.next_obs_memory.append(np.zeros((self.mem_size, obs_dim), dtype=np.float32))
            self.action_memory.append(np.zeros((self.mem_size, n_actions), dtype=np.float32))
            self.reward_memory.append(np.zeros(self.mem_size, dtype=np.float32))
            self.done_memory.append(np.zeros(self.mem_size, dtype=np.bool))
            self.connected_with_memory.append(np.zeros((self.mem_size, n_agents), dtype=int))
            self.next_connected_with_memory.append(np.zeros((self.mem_size, n_agents), dtype=int))

    def store_transition(self, obs, conns, actions, rewards, nobs, nconns, dones):
        index = self.mem_cntr % self.mem_size

        for a_idx in range(self.n_agents):
            self.obs_memory[a_idx][index] = obs[a_idx]
            self.action_memory[a_idx][index] = actions[a_idx]
            self.reward_memory[a_idx][index] = rewards[a_idx]
            self.next_obs_memory[a_idx][index] = nobs[a_idx]
            self.done_memory[a_idx][index] = dones[a_idx]
            self.connected_with_memory[a_idx][index] = conns[a_idx]
            self.next_connected_with_memory[a_idx][index] = nconns[a_idx]

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        obs = []
        nobs = []
        action = []
        reward = []
        done = []
        connected_with = []
        next_connected_with = []

        for i in range(self.n_agents):
            obs.append(self.obs_memory[i][batch])
            nobs.append(self.next_obs_memory[i][batch])
            action.append(self.action_memory[i][batch])
            reward.append(self.reward_memory[i][batch])
            done.append(self.done_memory[i][batch])
            connected_with.append(self.connected_with_memory[i][batch])
            next_connected_with.append(self.next_connected_with_memory[i][batch])

        return obs, connected_with, action, reward, nobs, next_connected_with, done
