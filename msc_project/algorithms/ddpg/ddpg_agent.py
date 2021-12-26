import numpy as np
import torch as T
import torch.nn.functional as F
from msc_project.algorithms.ddpg.buffer import ReplayBuffer
from msc_project.algorithms.ddpg.networks import ActorNetwork, CriticNetwork
from msc_project.utils.noise import GaussianActionNoise


class DDPGAgent:
    def __init__(
        self,
        args,
        policy_info,
        device,
    ) -> None:
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

        self.memory = ReplayBuffer(args.buffer_size, self.obs_dim, self.act_dims)

        decay = (args.act_noise_std_start - args.act_noise_std_min) / args.act_noise_decay_end_step
        self.noise = GaussianActionNoise(
            mean=0, std=args.act_noise_std_start, decay=decay, min_std=args.act_noise_std_min
        )

        self.device = device

        self.actor = ActorNetwork(
            self.alpha,
            self.obs_dim,
            self.fc1_dims,
            self.fc2_dims,
            n_actions=self.act_dims,
            name="actor",
            device=device,
        )
        self.critic = CriticNetwork(
            self.beta,
            self.obs_dim,
            self.fc1_dims,
            self.fc2_dims,
            n_actions=self.act_dims,
            name="critic",
            device=device,
        )

        self.target_actor = ActorNetwork(
            self.alpha,
            self.obs_dim,
            self.fc1_dims,
            self.fc2_dims,
            n_actions=self.act_dims,
            name="target_actor",
            device=device,
        )
        self.target_critic = CriticNetwork(
            self.beta,
            self.obs_dim,
            self.fc1_dims,
            self.fc2_dims,
            n_actions=self.act_dims,
            name="target_critic",
            device=device,
        )

    def get_random_actions(self):
        return [self.act_space.sample()[0] for _ in range(self.n_agents)]

    def get_actions(self, obs, explore=True):
        self.actor.eval()

        # TODO how to remove this knowledge to the environment observation?
        obs = obs["fov"]

        obs = T.tensor(obs, dtype=T.float).to(self.device)
        mu = self.actor.forward(obs).to(self.device)

        if self.noise != None and explore:
            noise = T.tensor(self.noise(mu.shape)).to(self.actor.device)
            mu_prime = T.add(mu, noise).to(self.device)
        else:
            mu_prime = mu

        mu_prime.clamp_(0, 1)  # inplace clamp

        self.actor.train()

        return mu_prime.view(-1).cpu().detach().numpy()

    def store_transistion(self, obs, acts, rewards, nobs, dones, info):
        # TODO remove knowledge on the environment
        for i in range(len(obs["fov"])):
            done = dones[i] or not info["cars_on_road"][i]
            fov = obs["fov"][i]
            nfov = nobs["fov"][i]
            self.memory.store_transition(fov, acts[i], rewards[i], nfov, done)

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

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.device)
        actions = T.tensor(actions, dtype=T.float).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

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

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
