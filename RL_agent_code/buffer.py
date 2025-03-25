import torch
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(object):
    '''Replay buffer that stores online (s, a, r, s', d) transitions for training.'''
    def __init__(self, maxsize=100000):
        # TODO: Initialize the buffer using the given parameters.
        # HINT: Once the buffer is full, when adding new experience we should not care about very old data.
        self.maxsize=maxsize # keeping track of maxsize
        self.buffer = deque(maxlen=maxsize) #instantiation of deque (allows adding and removing from both ends)
        pass
    
    def __len__(self):
        # TODO: Return the length of the buffer (i.e. the number of transitions).
        return len(self.buffer)
    
    def add_experience(self, state, action, reward, next_state, done):
        # TODO: Add (s, a, r, s', d) to the buffer.
        # HINT: See the transition data type defined at the top of the file for use here.
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        # TODO: Sample 'batch_size' transitions from the buffer.
        # Return a tuple of torch tensors representing the states, actions, rewards, next states, and terminal signals.
        # HINT: Make sure the done signals are floats when you return them.
        batch = random.sample(self.buffer, batch_size)

        # Unpack the batch into states, actions, rewards, next_states, and dones
        states = torch.stack([torch.tensor(transition.state).clone().detach().float() for transition in batch])
        actions = torch.tensor([transition.action for transition in batch], dtype=torch.long)
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float)
        next_states = torch.stack([torch.tensor(transition.next_state).clone().detach().float() for transition in batch])
        dones = torch.tensor([float(transition.done) for transition in batch], dtype=torch.float)


        return states, actions, rewards, next_states, dones