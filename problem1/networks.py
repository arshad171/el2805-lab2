import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    
    def __init__(self, n_actions: int=4, n_observations: int=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, n_actions)

    def forward(self, inputs):
        """
        Forward pass through network
        """
        x = F.relu(self.layer1(inputs))
        x = F.relu(self.layer2(x))
        out = self.out_layer(x)
        return out




        