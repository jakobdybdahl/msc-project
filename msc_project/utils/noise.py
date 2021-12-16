import numpy as np
import torch as T


class GaussianActionNoise:
    def __init__(self, mean=0.0, std=1.0, decay=1e-6, min_std=1e-3):
        self.mean = mean
        self.std = std
        self.decay = decay
        self.min_std = min_std

    def __call__(self, shape=None):
        x = np.random.normal(loc=self.mean, scale=self.std, size=shape)

        new_std = self.std - self.decay

        self.std = new_std if new_std > self.min_std else self.min_std

        return x
