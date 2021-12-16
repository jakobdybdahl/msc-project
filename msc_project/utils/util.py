import gym
import numpy as np
import torch


def get_dim_from_space(space):
    if isinstance(space, gym.spaces.Box):
        dim = space.shape[0]
    else:
        raise Exception("Unknown space: ", type(space))

    return dim


def get_cent_dim_from_spaces(spaces):
    cent_dim = 0
    for space in spaces:
        dim = get_dim_from_space(space)
        cent_dim += dim
    return cent_dim


def polyak_update(params, target_params, tau):
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input
