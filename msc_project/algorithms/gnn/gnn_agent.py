import numpy as np
import torch as T
import torch.nn.functional as F
from msc_project.algorithms.gnn.buffer import GNNReplayBuffer
from msc_project.algorithms.gnn.networks import ActorNetwork, CriticNetwork
from msc_project.utils.noise import GaussianActionNoise
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


class GNNAgent(object):
    def __init__(self, args, policy_info, device) -> None:
        self.device = device

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.lr_actor
        self.beta = args.lr_critic
        self.fc1_dimes = args.hidden_size1
        self.fc2_dims = args.hidden_size2
        self.use_gat = args.use_gat

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
            self.alpha, self.obs_dim, self.fc1_dimes, self.fc2_dims, self.act_dim, "actor", self.device, self.use_gat
        )
        self.critic = CriticNetwork(
            self.beta, self.obs_dim, self.fc1_dimes, self.fc2_dims, self.act_dim, "critic", self.device, self.use_gat
        )

        self.target_actor = ActorNetwork(
            self.alpha,
            self.obs_dim,
            self.fc1_dimes,
            self.fc2_dims,
            self.act_dim,
            "target_actor",
            self.device,
            self.use_gat,
        )
        self.target_critic = CriticNetwork(
            self.beta,
            self.obs_dim,
            self.fc1_dimes,
            self.fc2_dims,
            self.act_dim,
            "target_critic",
            self.device,
            self.use_gat,
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
        fov, who_in_fov = obs["fov"], obs["who_in_fov"]

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
        fovs = obs["fov"]
        comms = obs["who_in_fov"]
        nfovs = nobs["fov"]
        ncomms = nobs["who_in_fov"]

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

        # agent_idx is batch_size long, so 'b' is each batch, for instance from 0 to 127
        for b in range(len(agent_idx)):
            comm = comms[b] # comms are agent specific
            ncomm = ncomms[b]

            edge_index = self._get_edge_index(comm)
            n_edge_index = self._get_edge_index(ncomm)

            # fovs[b] are six fovs for this sample
            # edge_index is the connections between the nodes in the fov seen from this agent.
            # Each graph are constructed based on a single agents fov (see store_transistion)
            data_list.append(Data(x=fovs[b], edge_index=edge_index))
            ndata_list.append(Data(x=nfovs[b], edge_index=n_edge_index))

        # Batch is one graph containg 128 disconnected graphs with nodes that contains fov and are connected based on the edge_index
        # The Batch-structure ensures that the edge_index are handled properly by incrementing them
        batch = Batch.from_data_list(data_list).to(self.device)
        nbatch = Batch.from_data_list(ndata_list).to(self.device)

        # Agent_mask is used to filter out the global knowledge when training
        # However, we are dependent on n_agents during training - which is fine.
        agent_mask = T.zeros_like(batch.batch, dtype=T.bool)
        for i, a in enumerate(agent_idx):
            # Agent mask is 128x6=768 but 1D/flat
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
        with T.no_grad():
            if tau is None:
                tau = self.tau

            actor_params = self.actor.state_dict().items()
            critic_params = self.critic.state_dict().items()
            target_actor_params = self.target_actor.state_dict().items()
            target_critic_params = self.target_critic.state_dict().items()

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
