# Copyright [2024] [KTH Royal Institute of Technology]
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


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

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.curr_actor_loss = 0
        self.curr_critic_loss = 0

    def forward(self, state: np.ndarray):
        """Performs a forward computation"""
        with th.no_grad():
            x = th.tensor(state)
            action = self.actor.forward(x)

        return action

    def backward(self, batch, gamma, update_actor=False):
        """Performs a backward pass on the network"""
        states = th.tensor(batch["states"])
        actions = th.tensor(batch["actions"])
        rewards = th.tensor(batch["rewards"])
        next_states = th.tensor(batch["next_states"])
        dones = th.tensor(batch["dones"])

        with th.no_grad():
            next_actions = self.actor_tar.forward(next_states)
            Q_next = self.critic_tar.forward(th.hstack([next_states, next_actions]))
            targets = rewards + (1 - dones) * gamma * Q_next

        Q = self.critic(th.hstack([states, actions]))

        # critic_loss = th.mean((targets - Q)**2)
        # actor_loss = - th.mean(Q**2)

        critic_loss = F.mse_loss(Q, targets)
        self.curr_critic_loss = critic_loss.item()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        if update_actor:
            actions_p = self.actor(states)
            Q = self.critic.forward(th.hstack([states, actions_p]))

            actor_loss = -Q.mean()
            self.curr_actor_loss = actor_loss.item()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
        
        return self.curr_critic_loss, self.curr_actor_loss


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
