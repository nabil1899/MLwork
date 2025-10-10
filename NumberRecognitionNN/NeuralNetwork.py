import numpy as np



from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
from collections import deque


class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # Increased size for hidden layers
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, states):
        input_list = np.array(states, dtype=np.float32)
        x = torch.from_numpy(input_list).float()
        return self.fc(x)


class DQAgent:

    def __init__(self, state_dim, action_dim,lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        # Models
        self.model = NN(state_dim, action_dim)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()



    def train(self):

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute current Q-values
        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
