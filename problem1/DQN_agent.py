# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from networks import DQNetwork
from replay_buffer import ReplayMemory, Transition

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class DQNAgent(Agent):
    def __init__(self, n_actions, n_observations, capacity=5000, epsilon_max=0.99, epsilon_min=0.01, 
                 batch_size=32, gamma=0.99, learning_rate=0.001):
        super().__init__(n_actions)
        self.policy_net = DQNetwork(n_actions, n_observations)
        self.target_net = DQNetwork(n_actions, n_observations)
        self.replay_buffer = ReplayMemory(capacity=capacity)
        self.training_mode = True
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        """
        Select an action epsilon-greedily (in training mode)
        """
        if self.training_mode:
            if random.random() < self.epsilon:
                return torch.tensor([[random.randint(0, self.n_actions-1)]])
        return self.policy_net(state).max(1).indices.view(1,1)
    
    def backward(self):
        """
        Perform an optimization step, samples transitions from the replay buffer
        and updates the 
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if not s is None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(size=(self.batch_size,), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        y = (reward_batch + self.gamma * next_state_values)
        loss = self.loss_fn(state_action_values, y.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def memorize(self, state, action, next_state, reward):
        """
        Pushes a transition to the agents replay buffer
        """
        self.replay_buffer.push(state, action, next_state, reward)

    def update_target_net(self):
        """
        Update target net parameters to policy net parameters
        """
        policy_net_state_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_net_state_dict)

    def update_epsilon(self, k, N):
        """
        Updates epsilon (in epsilon-greedy) with exponential decay
        Params:
            k (int): current episode
            N (int): total number of training episodes
        """
        Z = 0.95*N
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon_max*(self.epsilon_min/self.epsilon_max)**((k-1)/(Z-1))
        )

    def set_training_mode(self, mode):
        """
        Set training mode True = Training, False = Inference
        """
        self.training_mode = mode

    def save_model_weights(self, path):
        """
        Saves the model to path
        """
        torch.save(self.policy_net.state_dict(), path)

if __name__ == "__main__":
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    a = torch.tensor([[1,3], [4,3]])
    print(device)
    print(a)
