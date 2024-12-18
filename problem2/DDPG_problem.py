# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import json
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, Agent
from buffer import Buffer
import warnings
import torch as th
from tqdm import tqdm
from DDPG_soft_updates import soft_updates
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

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLanderContinuous-v2', render_mode = "human")

env.reset()

# Parameters
N_episodes = 300               # Number of episodes to run for training
n_ep_running_average = 50      # Running average of 50 episodes
m = len(env.observation_space.high) # dimensionality of the action
n = len(env.action_space.high) # dimensionality of the action

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
# agent = RandomAgent(n_actions=m)
MAX_BUFFER_SIZE = 30_000
BATCH_SIZE = 64
UPDATE_FREQ = 2
GAMMA = 0.5
TAU = 1e-3
MU = 0.15
SIGMA = 0.2
agent = Agent(state_dim=m, action_dim=n)
buffer = Buffer(buffer_size=MAX_BUFFER_SIZE, state_dim=m, action_dim=n)
# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# initialize buffer

done, truncated = False, False
state = env.reset()[0]

noise_prev = th.zeros(size=(n,))
noise_mean = th.zeros(size=(n,))
noise_var = SIGMA**2 * th.eye(n=n)

noise_dist = th.distributions.MultivariateNormal(noise_mean, noise_var)

print("filling up the buffer")
for _ in tqdm(range(MAX_BUFFER_SIZE)):
    action = agent.forward(state)

    noise = -MU * noise_prev + noise_dist.sample()

    action += noise

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

    if done or truncated:
        state = env.reset()[0]


fig1 = plt.figure(1)
fig2 = plt.figure(2)


actor_losses = []
critic_losses = []
for i in EPISODES:
    # Reset enviroment data
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    eps_actor_losses = []
    eps_critic_losses = []
    while not (done or truncated):
        # Take a random action
        action = agent.forward(state)

        noise = -MU * noise_prev + noise_dist.sample()

        action += noise

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

        polyak_flag = t % UPDATE_FREQ == 0
        # get a batch
        batch = buffer.sample(batch_size=BATCH_SIZE)
        # backward step
        critic_loss, actor_loss = agent.backward(batch=batch, gamma=GAMMA, update_actor=polyak_flag)
        # polyak
        if polyak_flag:
            soft_updates(agent.critic, agent.critic_tar, TAU)
            soft_updates(agent.actor, agent.actor_tar, TAU)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

        eps_critic_losses.append(critic_loss)
        eps_actor_losses.append(actor_loss)

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    critic_losses.append(float(np.mean(eps_critic_losses)))
    actor_losses.append(float(np.mean(eps_actor_losses)))

    plt.figure(1)
    plt.plot(range(len(critic_losses)), critic_losses)
    plt.title("critic")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.show(block=False)

    plt.figure(2)
    plt.plot(range(len(actor_losses)), actor_losses)
    plt.title("actor")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.show(block=False)

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

losses = {
    "critic": critic_losses,
    "actor": actor_losses,
}
json.dump(losses, open("losses.json", "w"))
torch.save(critic, "neural-network-2-critic.pth")
torch.save(actor, "neural-network-2-actor.pth")

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

plt.savefig("output.png")
plt.show()
