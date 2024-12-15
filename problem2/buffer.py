from typing import NamedTuple
import numpy as np
import torch as th
import random


class BufferSamples(NamedTuple):
    states: th.tensor
    actions: th.tensor
    rewards: th.tensor
    next_states: th.tensor
    dones: th.tensor


class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.states_bf = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.actions_bf = np.zeros(shape=(buffer_size, action_dim), dtype=np.float32)
        self.rewards_bf = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.next_states_bf = np.zeros(shape=(buffer_size, state_dim), dtype=np.float32)
        self.dones_bf = np.zeros(shape=(buffer_size, 1), dtype=np.float32)

        self.buffer_size = buffer_size
        self.buffer_ix = 0
        self.curr_size = 0

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

    def sample(self, batch_size):
        samples_ix = random.choices(range(0, self.curr_size), k=batch_size)

        states = self.states_bf[samples_ix]
        actions = self.actions_bf[samples_ix]
        rewards = self.rewards_bf[samples_ix]
        next_states = self.next_states_bf[samples_ix]
        dones = self.dones_bf[samples_ix]

        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

        return batch
