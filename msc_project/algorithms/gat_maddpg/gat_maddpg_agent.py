import numpy as np
import random
import torch

from msc_project.utils.noise import GaussianActionNoise

from .buffer import GNNReplayBuffer
from .brain import Brain


class GATMADDPGAgent:
    def __init__(self,
                args,
                policy_info,
                device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                ):

        self.act_space = policy_info["act_space"]
        self.obs_dim = policy_info["obs_space"].shape[0]
        self.act_dims = policy_info["act_space"].shape[0]
        self.n_agents = policy_info["num_agents"]

        decay = (args.act_noise_std_start - args.act_noise_std_min) / args.act_noise_decay_end_step
        self.noise = GaussianActionNoise(
            mean=0, std=args.act_noise_std_start, decay=decay, min_std=args.act_noise_std_min
        )

        self.brain = Brain(
            agent_count=self.n_agents,
            observation_size=self.obs_dim,
            action_size=self.act_dims,
            alpha=args.lr_actor,
            beta=args.lr_critic,
            soft_update_tau=args.tau,
            discount_gamma=args.gamma,
            noise=self.noise,
            device=device
        )

        self.device = device
        self._batch_size = args.batch_size

        # Replay memory
        self.memory = GNNReplayBuffer(
            self.act_dims,
            args.buffer_size,
            self._batch_size,
            device
        )

        self.t_step = 0

    def store_transistion(self, obs, actions, rewards, next_obs, dones, info):
        obs, comms = obs["fov"], obs["who_in_fov"]
        next_obs, next_comms = next_obs["fov"], next_obs["who_in_fov"]

        self.memory.store_transition(obs, actions, rewards, next_obs, dones, comms, next_comms)

    def act_torch(self, fov, who_in_fov, target, explore=True, train=False):
        """Act based on the given batch of observations.
        :param obs: current observation, array of shape == (b, observation_size)
        :param noise: noise factor
        :param train: True for training mode else eval mode
        :return: actions for given state as per current policy.
        """
        actions = [
            self.brain.act(fov[:, i], who_in_fov[i], i, target, explore, train)
            for i in range(self.n_agents)
        ]

        actions = torch.stack(actions).transpose(1, 0)

        return torch.clamp(actions, 0, 1)

    def get_actions(self, obs, explore=True, target=False, ):
        fov, who_in_fov = obs["fov"], obs["who_in_fov"]

        fov = torch.from_numpy(fov).float().\
            to(self.device).unsqueeze(0)

        with torch.no_grad():
            actions = np.vstack([
                a.cpu().numpy()
                for a in self.act_torch(fov, who_in_fov, target, explore)
            ])

        return actions.flatten()

    def get_random_actions(self):
         return [self.act_space.sample()[0] for _ in range(self.n_agents)]

    def save_models(self, path=None):
        self.brain.save_models(path)

    def load_models(self, path=None):
        self.brain.load_models(path)

    def learn(self):
        experiences = self.memory.sample_buffer()
        experiences = self._tensor_experiences(experiences)

        observations, actions, rewards, next_observations, dones, comms, ncomms = experiences

        all_obs = self._flatten(observations)
        all_actions = self._flatten(actions)
        all_next_obs = self._flatten(next_observations)

        all_target_next_actions = self._flatten(self.act_torch(
            next_observations,
            ncomms,
            target=True,
            train=False
        ).contiguous())

        all_local_actions = self.act_torch(
            observations,
            comms,
            target=False,
            train=True
        ).contiguous()

        for i in range(self.n_agents):
            # update critics
            self.brain.update_critic(
                rewards[:, i].unsqueeze(-1), dones[:, i].unsqueeze(-1),
                all_obs, all_actions, all_next_obs, all_target_next_actions,
                comms[:, i], ncomms[:, i]
            )

            # update actors
            all_local_actions_agent = all_local_actions.detach()
            all_local_actions_agent[:, i] = all_local_actions[:, i]
            all_local_actions_agent = self._flatten(all_local_actions_agent)
            self.brain.update_actor(
                all_obs, all_local_actions_agent.detach(), comms[:, i], ncomms[:, i]
            )

            # update targets
            self.brain.update_targets()

    def _tensor_experiences(self, experiences):
        ob, actions, rewards, next_ob, dones, comms, ncomms = \
            [torch.from_numpy(e).float().to(self.device) for e in experiences]
        return ob, actions, rewards, next_ob, dones, comms, ncomms

    @staticmethod
    def _flatten(tensor):
        try:
            b, n_agents, d = tensor.shape
            return tensor.view(b, n_agents * d)
        except:
            return tensor
