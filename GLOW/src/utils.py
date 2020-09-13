import numpy as np
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
    init_channels *= 2
    init_hw = init_hw // 2
    for i in range(n_levels):
        z = std * prior.sample((n_samples,
                                init_channels * 2**i,
                                init_hw // 2**i,
                                init_hw // 2**i)).squeeze()
        samples.append(z)
    z = std * prior.sample((n_samples,
                            (init_channels * 2**n_levels)*2,
                            init_hw // 2**n_levels,
                            init_hw // 2**n_levels)).squeeze()
    samples.append(z)
    return samples

def make_img(model,
             prior,
             n_samples,
             z_stds, 
             n_levels,
             c,
             hw):
    with torch.no_grad():
        samples = []
        for z in z_stds:
            h = sample(prior, n_samples, n_levels, init_channels=c, init_hw=hw, 
                       std=z)
            print(f"h size {np.shape(h)}")
            s, _ = model.backward(h)
            zs, log_det = model(s)
            prior_logprob = sum(prior.log_prob(zz).sum([1,2,3]) for zz in zs)
            log_prob = prior_logprob + log_det
            log_prob /= (np.log(2) * c * hw * hw)
            samples.append(s[log_prob.argsort().flip(0)])
    return torch.cat(samples,0)
