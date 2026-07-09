import torch
from torch import nn

class policyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, s):
        return self.model(s)
    

class baseline(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
    def forward(self, s):
        return self.model(s).squeeze(-1)
    
class DeepQAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = env.observation_space.shape[0], out_channels = 16, kernel_size = 7, stride = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels=16, out_channels = 32, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Flatten(),
            nn.Linear(in_features=1152, out_features=256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n),
        )

    def forward(self, x):
        return self.model(x)