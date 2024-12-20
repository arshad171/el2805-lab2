# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQNAgent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v2', render_mode = "human")


env.reset()

# Parameters
N_episodes = 300                             # Number of episodes
discount_factor = 0.90                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
capacity = 5000
target_net_update_freq = int(capacity/N_episodes)
agent_cfg = {
    'n_actions': n_actions,
    'n_observations': dim_state,
    'capacity': capacity,
    'epsilon_max': 0.99,
    'epsilon_max': 0.05,
    'batch_size': 32,
    'gamma': discount_factor,
    'learning_rate': 0.001
}

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
# agent = RandomAgent(n_actions)
agent = DQNAgent(**agent_cfg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
step = 0
best_avg_return = float('-inf')
for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_episode_reward = 0.
    t = 0
    while not (done or truncated):
        # Take a random action
        action = agent.forward(state)
        
        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action.item())
        total_episode_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        # Remember transition in replay buffer
        agent.memorize(state, action, next_state, reward)

        # Update state for next iteration
        state = next_state

        # Optimize model
        agent.backward()

        # Update target net
        if step % target_net_update_freq == 0 and t != 0:
            agent.update_target_net()

        t+= 1
        step+=1

    agent.update_epsilon(i, N_episodes)

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    if len(episode_reward_list) >= 50:
        avg_return = sum(episode_reward_list[-50:]) / 50
        if avg_return > 50 and avg_return > best_avg_return:
            agent.save_model_weights("./model_weights/best_model.pt")

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
