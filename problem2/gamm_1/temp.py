import numpy as np
import torch as th

n = 2
SIGMA = 0.1
MU = 0.2
noise_prev = th.zeros(size=(n,))

noise_mean = th.zeros(size=(n,))
noise_var = SIGMA * th.eye(n=n)


noise = -MU * noise_prev + th.normal(mean=noise_mean, std=noise_var)
dist = th.distributions.MultivariateNormal(noise_mean, noise_var)

print(dist.sample())