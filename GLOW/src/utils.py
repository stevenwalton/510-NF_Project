import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def sample(prior,
           n_samples,
           n_levels,
           init_channels,
           init_hw,
           std = 1.):
    samples = []
    n_levels -= 1
    init_channels *= 3
    init_hw = init_hw // 2
    for i in range(n_levels):
        z = std * prior.sample((n_samples,
                                init_channels * 2**i,
                                init_hw // 2**i)).squeeze()
        samples.append(z)
    z = std * prior.sample((n_samples,
                            (init_channels * 2**n_levels)*2,
                            init_hw // 2**n_levels)).squeeze()
    samples.append(z)
    return samples
