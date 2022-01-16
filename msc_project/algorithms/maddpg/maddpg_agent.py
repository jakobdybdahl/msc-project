import enum
import torch as T
import torch.nn.functional as F
from .agent import Agent
from .buffer import MultiAgentReplayBuffer
import numpy as np
from msc_project.utils.noise import GaussianActionNoise

# CURRENT: v1 and 64 dims
# TODO: Discerete action space hack?
# TODO: Very high coll penalty and no step penal?
class MADDPGAgent:
    def __init__(self, args, policy_info, device):
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.alpha = args.lr_actor
        self.beta = args.lr_critic
        self.fc1_dims = args.hidden_size1
        self.fc2_dims = args.hidden_size2

        self.obs_space = policy_info["obs_space"]
        self.act_space = policy_info["act_space"]

        self.obs_dim = policy_info["obs_space"].shape[0]
        self.act_dims = policy_info["act_space"].shape[0]

        self.n_agents = policy_info["num_agents"]

        self.agents = []
        self.device = device

        self.memory = MultiAgentReplayBuffer(args.buffer_size, self.obs_dim * self.n_agents, self.obs_dim, 
                        self.act_dims, self.n_agents, batch_size=1024)


        decay = (args.act_noise_std_start - args.act_noise_std_min) / args.act_noise_decay_end_step
        self.noise = GaussianActionNoise(
            mean=0, std=args.act_noise_std_start, decay=decay, min_std=args.act_noise_std_min
        )

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(self.obs_dim, self.obs_dim * self.n_agents,  
                            self.act_dims, self.n_agents, agent_idx, self.noise, decay, fc1=self.fc1_dims, fc2=self.fc2_dims))
        # self.agent = Agent(self.obs_dim, self.obs_dim * self.n_agents,  
        #                     self.act_dims, self.n_agents, 0, self.noise, decay, fc1=self.fc1_dims, fc2=self.fc2_dims)


    def save_models(self, path):
        print('... saving model ...')
        for agent in self.agents:
            agent.save_models(path)

    def load_models(self, path):
        print('... loading model ...')
        for agent in self.agents:
            agent.load_models(path)

    def get_random_actions(self):
        return [self.act_space.sample()[0] for _ in range(self.n_agents)]

    def get_actions(self, raw_obs, explore=True):
        raw_obs = raw_obs["fov"]
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], explore)
            actions.append(action)
        return actions

    def store_transistion(self, obs, actions, rewards, nobs, dones, info):
        obs = obs["fov"]
        nobs = nobs["fov"]
        state = self._obs_list_to_state_vector(obs)
        state_ = self._obs_list_to_state_vector(nobs)

        self.memory.store_transition(obs, state, actions, rewards, nobs, state_, dones)

    def learn(self):
        if not self.memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = self.memory.sample_buffer()

        states = T.tensor(states, dtype=T.float).to(self.device)
        actions = T.tensor(actions, dtype=T.float).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.device)
        dones = T.tensor(dones).to(self.device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(self.device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(self.device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.zero_grad()
        # self.agent.actor.optimizer.zero_grad()
        
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            actor_loss.backward(retain_graph=True)
            
        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.step()
            agent.update_network_parameters()
        # self.agent.actor.optimizer.step()
        # self.agent.update_network_parameters()

    def _obs_list_to_state_vector(self, observation):
        state = np.array([])
        for obs in observation:
            state = np.concatenate([state, obs])
        return state
