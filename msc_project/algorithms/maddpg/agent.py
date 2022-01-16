import torch as T
from .networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx,
                    noise, decay, alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,  name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, name=self.agent_name+'_target_critic')

        self.noise = noise
        self.decay = decay

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, explore):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state)

        if self.noise != None and explore:
            noise = T.tensor(self.noise(mu.shape)).to(self.actor.device)
            mu_prime = T.add(mu, noise)
        else:
            mu_prime = mu

        mu_prime.clamp_(0, 1)  # inplace clamp

        return mu_prime.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self, path):
        self.actor.save_checkpoint(path)
        self.target_actor.save_checkpoint(path)
        self.critic.save_checkpoint(path)
        self.target_critic.save_checkpoint(path)

    def load_models(self, path):
        self.actor.load_checkpoint(path)
        self.target_actor.load_checkpoint(path)
        self.critic.load_checkpoint(path)
        self.target_critic.load_checkpoint(path)
