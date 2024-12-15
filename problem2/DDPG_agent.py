# Copyright [2024] [KTH Royal Institute of Technology]
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch as th
import torch.nn as nn


class Agent(object):
    """Base agent class

    Args:
        n_actions (int): actions dimensionality

    Attributes:
        n_actions (int): where we store the dimensionality of an action
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, self.action_dim),
            nn.Tanh(),
        )

        self.actor_tar = nn.Sequential(
            nn.Linear(self.state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, self.action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

        self.critic_tar = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

    def forward(self, state: np.ndarray):
        """Performs a forward computation"""
        with th.no_grad():
            x = th.tensor(state)
            action = self.actor.forward(x)
        
        return action.numpy()

    def backward(self):
        """Performs a backward pass on the network"""
        pass


class RandomAgent(Agent):
    """Agent taking actions uniformly at random, child of the class Agent"""

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute a random action in [-1, 1]

        Returns:
            action (np.ndarray): array of float values containing the
                action. The dimensionality is equal to self.n_actions from
                the parent class Agent.
        """
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
