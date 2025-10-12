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

            nn.Linear(input_dim, 15),
            nn.Linear(15, output_dim)

        )

    def forward(self, states):
        input_list = np.array(states, dtype=np.float32)
        x = torch.from_numpy(input_list).float()
        return self.fc(x)


class Agent:

    def __init__(self, state_dim, action_dim,lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        # Models
        self.model = NN(state_dim, action_dim)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,states):
        return self.model(states)

    def train(self,state,target):
        self.model.train()
        states = torch.FloatTensor(state)


        # Compute current Q-values
        current = self.model(states)
        target_values=states = torch.FloatTensor(target)
        # current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = self.loss_fn(current, target_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
