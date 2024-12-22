# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

class Actor(th.nn.Module):

    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim

        self.backbone = th.nn.Linear(self.state_dim, 400)

        self.mu_lin1 = th.nn.Linear(400, 200)
        self.mu_lin2 = th.nn.Linear(200, 2)

        self.sigma_lin1 = th.nn.Linear(400, 200)
        self.sigma_lin2 = th.nn.Linear(200, 2)

        self.relu_act = th.nn.ReLU()
        self.tanh_act = th.nn.Tanh()
        self.sig_act = th.nn.Sigmoid()

    def forward(self, x):
        h = self.backbone(x)
        h = self.relu_act(h)

        h1 = self.mu_lin1(h)
        h1 = self.relu_act(h1)
        h1 = self.mu_lin2(h1)
        h1 = self.tanh_act(h1)

        h2 = self.sigma_lin1(h)
        h2 = self.relu_act(h2)
        h2 = self.sigma_lin2(h2)
        h2 = self.sig_act(h2)

        return (h1, h2)

class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, state_dim: int, action_dim: int, eps: float, m: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.EPS = eps
        self.M = m

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

        self.actor = Actor(state_dim=self.state_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.curr_actor_loss = 0
        self.curr_critic_loss = 0

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        x = th.tensor(state)
        mu, var = self.actor.forward(x)

        # cross-variances are zero
        # action_dist = th.distributions.MultivariateNormal(mu, th.diag(var))
        action_dist = th.distributions.Independent(th.distributions.Normal(mu, var), 1)

        action = action_dist.sample()
        action = th.clamp(action, -1.0, 1.0)

        return action


    def mvn_pdf(self, x, mu, cov):
        cov = th.diag(cov)
        D = x.shape[0]
        cov_det = th.det(cov)
        cov_inv = th.inverse(cov)
        
        norm_const = 1.0 / th.sqrt(((2 * th.pi) ** D) * cov_det)
        
        diff = (x - mu).unsqueeze(1)
        exp_term = -0.5 * th.matmul(th.matmul(diff.T, cov_inv), diff)
        
        return norm_const * th.exp(exp_term).squeeze()

    def compute_action_proba(self, states, actions):
        mu, var = self.actor(states)

        pdf = self.mvn_pdf(actions, mu, var)

        return pdf
    
    def compute_action_log_proba(self, states, actions):
        mu, var = self.actor(states)

        action_dist = th.distributions.Independent(th.distributions.Normal(mu, var), 1)

        return action_dist.log_prob(actions)


    def backward(self, batch):
        ''' Performs a backward pass on the network '''
        batch_len = len(batch["states"])
        states = th.tensor(batch["states"])
        actions = th.tensor(batch["actions"])
        rewards = th.tensor(batch["rewards"])
        next_states = th.tensor(batch["next_states"])
        dones = th.tensor(batch["dones"])
        returns = th.tensor(batch["returns"])

        phi = returns - self.critic.forward(states).detach()
        old_proba = th.zeros(size=(batch_len,))


        # for b_ix in range(batch_len):
        #     # old_proba[b_ix] = self.compute_action_proba(states[b_ix], actions[b_ix]).detach()
        #     old_proba[b_ix] = self.compute_action_log_proba(states[b_ix], actions[b_ix]).detach()

        old_proba = self.compute_action_log_proba(states, actions).detach()

        for _ in range(self.M):
            V_pred = self.critic(states)

            critic_loss = F.mse_loss(V_pred, returns)
            self.curr_critic_loss = critic_loss.item()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

            # with th.no_grad():
            # phi = returns - self.critic.forward(states).detach()

            new_proba = th.zeros(size=(batch_len,))

            # for b_ix in range(batch_len):
            #     # new_proba[b_ix] = self.compute_action_proba(states[b_ix], actions[b_ix])
            #     new_proba[b_ix] = self.compute_action_log_proba(states[b_ix], actions[b_ix])

            new_proba = self.compute_action_log_proba(states, actions)

            # ratio = new_proba / old_proba

            ratio = th.exp(new_proba - old_proba)

            clipped_ratio = th.clamp(ratio, 1 - self.EPS, 1 + self.EPS)

            original_returns = ratio * phi
            clipped_returns = clipped_ratio * phi

            actor_loss = -th.minimum(original_returns, clipped_returns).mean()
            self.curr_actor_loss = actor_loss.item()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()


        return self.curr_critic_loss, self.curr_actor_loss


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
