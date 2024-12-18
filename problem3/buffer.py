from typing import NamedTuple
import numpy as np
import torch as th
import random


class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim, gamma):
        self.gamma = gamma

        self.states_bf = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.actions_bf = np.zeros(shape=(buffer_size, action_dim), dtype=np.float32)
        self.rewards_bf = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.next_states_bf = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.dones_bf = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.returns_bf = np.zeros(shape=(buffer_size, 1), dtype=np.float32)

        self.buffer_size = buffer_size
        self.buffer_ix = 0
        self.curr_size = 0

    def clear(self):
        self.buffer_ix = 0
        self.curr_size = 0

        self.states_bf.fill(0.0)
        self.actions_bf.fill(0.0)
        self.rewards_bf.fill(0.0)
        self.next_states_bf.fill(0.0)
        self.dones_bf.fill(0.0)

    def add(
        self,
        state: np.array,
        action: np.array,
        reward: np.array,
        next_state: np.array,
        done: np.array,
    ):
        self.states_bf[self.buffer_ix] = state
        self.actions_bf[self.buffer_ix] = action
        self.rewards_bf[self.buffer_ix] = reward
        self.next_states_bf[self.buffer_ix] = next_state
        self.dones_bf[self.buffer_ix] = done

        self.buffer_ix = (self.buffer_ix + 1) % self.buffer_size
        if self.curr_size < self.buffer_size:
            self.curr_size += 1

    # this solution did not work, as the gamma^t can approach zero
    # def compute_returns(self):
    #     rewards = self.rewards_bf[:self.curr_size].ravel()

    #     discounts = np.ones_like(rewards) * self.gamma

    #     cumul_discounts = np.cumprod(discounts) / self.gamma

    #     # (g_0, g_1, r_2, ...) * (g^0, g^1, g^2, ...)
    #     # cumsum sums along positive x, so reverse it
    #     returns = np.cumsum((rewards * cumul_discounts)[::-1])[::-1]
    #     # divide by the g^t
    #     returns /= cumul_discounts

    #     self.returns_bf[:self.curr_size] = returns.reshape((-1, 1))

    def compute_returns(self):
        # return for the last step is just the reward
        self.returns_bf[self.curr_size - 1] = self.rewards_bf[self.curr_size - 1]

        # recursive way, returns[ix] = gamma * returns[ix + 1]
        for ix in range(self.curr_size - 2, -1, -1):
            self.returns_bf[ix] = self.rewards_bf[ix] + self.gamma * self.returns_bf[ix + 1]

    def sample(self, batch_size):
        samples_ix = random.choices(range(0, self.curr_size), k=batch_size)

        states = self.states_bf[samples_ix]
        actions = self.actions_bf[samples_ix]
        rewards = self.rewards_bf[samples_ix]
        next_states = self.next_states_bf[samples_ix]
        dones = self.dones_bf[samples_ix]
        returns = self.returns_bf[samples_ix]

        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "returns": returns,
        }

        return batch
