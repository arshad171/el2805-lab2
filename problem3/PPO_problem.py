# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import RandomAgent, Agent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from buffer import Buffer

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLanderContinuous-v2', render_mode = "human")

env.reset()

# Parameters
N_episodes = 1600               # Number of episodes to run for training
discount_factor = 0.95         # Value of gamma
n_ep_running_average = 50      # Running average of 20 episodes
m = len(env.observation_space.high) # dimensionality of the action
n = len(env.action_space.high) # dimensionality of the action

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []


MAX_BUFFER_SIZE = 30_000
BATCH_SIZE = 64
GAMMA = 0.99
M = 10
EPS = 0.2
# Agent initialization
# agent = RandomAgent(n)
agent = Agent(state_dim=m, action_dim=n, eps=EPS, m=M)
buffer = Buffer(buffer_size=MAX_BUFFER_SIZE, state_dim=m, action_dim=n, gamma=GAMMA)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    buffer.clear()
    while not (done or truncated):
        # Take a random action
        action = agent.forward(state)
        action = action.numpy()

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)

        buffer.add(
            state,
            action,
            reward,
            next_state,
            done or truncated,
        )

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1
    

    buffer.compute_returns()
    batch = buffer.sample(batch_size=BATCH_SIZE)
    agent.backward(batch=batch)

    buffer.clear()
    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Close environment
env.close()

critic = agent.critic
actor = agent.actor

torch.save(critic, "neural-network-3-critic.pth")
torch.save(actor, "neural-network-3-actor.pth")
    
# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
