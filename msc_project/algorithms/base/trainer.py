from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def __init__(self, args, num_agents, policy, device):
        raise NotImplementedError

    @abstractmethod
    def train_policy_on_batch(self, batch):
        raise NotImplementedError

    @abstractmethod
    def prep_training(self):
        """Sets all networks to training mode."""
        raise NotImplementedError

    @abstractmethod
    def prep_rollout(self):
        """Sets all networks to eval mode."""
        raise NotImplementedError
