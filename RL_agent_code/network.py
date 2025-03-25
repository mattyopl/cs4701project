import torch
from torch import nn
from utils import make_network
import numpy as np

class QNetwork(nn.Module):
    def __init__(self,
                 gamma,
                 state_dim,
                 action_dim,
                 hidden_sizes=[10, 10]):
        super().__init__()
        self.gamma = gamma
        
        # neural net architecture
        self.network = make_network(state_dim, action_dim, hidden_sizes)
    
    def forward(self, states):
        '''Returns the Q values for each action at each state.'''
        qs = self.network(states)
        return qs

    def get_max_q(self, states):
        # TODO: Get the maximum Q values of all states s.
        qs = self.forward(states)  # Get Q values for all actions
        max_q_values, _ = torch.max(qs, dim=1)  # Take max over action dimension
        return max_q_values
    
    def get_action(self, state, eps):
        # TODO: Get the action at a given state according to an epsilon greedy method.
        rand = torch.rand(1).item() # retrieve random value [0, 1) uniform distrib
        qs = self.forward(state) # Get Q-values
        if rand < eps:
          # Exploration: Random action
          distrib = torch.rand(qs.shape[-1]) # uniform distribution over our actions
        else:
          # Exploitation: Follow optimal model prediction
          distrib = qs # Get Q-values

        return torch.argmax(distrib).item()
    
    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):
        # TODO: Get the next Q function targets, as given by the Bellman optimality equation for Q functions.
        max_qs = self.get_max_q(next_states) # Get max Q values for next states
        targets = rewards + self.gamma * max_qs * (1 - dones) # element-wise, if done, then 1-dones[i]=0; Bellman update
        return targets