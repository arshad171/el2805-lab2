# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import gymnasium as gym
import torch
from tqdm import trange
import warnings
from networks import DQNetwork
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt

def produce_augmented_state_plot():
    
    def construct_state(y, omega):
        state = np.zeros((8,))
        state[1] = y
        state[4] = omega
        return state
    
    # Load model
    try:
        state_dict = torch.load('./model_weights/best_model_170257.pt', weights_only=True, map_location='cpu')
        model = DQNetwork()
        model.load_state_dict(state_dict)
        print('Network model: {}'.format(model))
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    ys = np.linspace(0, 1.5, 100)
    omegas = np.linspace(-np.pi, np.pi, 100)
    X, Y = np.meshgrid(ys, omegas)
    grid_states = np.c_[X.ravel(), Y.ravel()]
    grid_states = np.array([construct_state(*gs) for gs in grid_states])
    grid_states = torch.tensor(grid_states, dtype=torch.float32)
    q_values = model(grid_states).argmax(1).detach().numpy().reshape(X.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, q_values, cmap="viridis")
    ax.set_xlabel("y")
    ax.set_ylabel("$\\omega$")
    ax.set_zlabel("Action")
    plt.show()

def test_q_values():
    def construct_state(y, omega):
        state = np.zeros((8,))
        state[1] = y
        state[4] = omega
        return state
    
    # Load model
    try:
        state_dict = torch.load('./model_weights/best_model_170257.pt', weights_only=True, map_location='cpu')
        model = DQNetwork()
        model.load_state_dict(state_dict)
        print('Network model: {}'.format(model))
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    state = construct_state(1, 4)
    state = torch.tensor(state, dtype=torch.float32)
    print(model(state))

if __name__ == "__main__":
    test_q_values()
    
