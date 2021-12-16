from abc import ABC, abstractmethod


class MLPPolicy(ABC):
    """Abstract MLP policy class"""

    @abstractmethod
    def get_actions(self, obs, t_env, explore):
        raise NotImplementedError

    @abstractmethod
    def get_random_actions(self, obs):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, dir):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, dir):
        raise NotImplementedError
