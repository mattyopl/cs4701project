import torch
from torch import optim
from torch.nn import functional as F
from network import QNetwork
import gym
from tqdm import tqdm
from buffer import ReplayBuffer
from itertools import count
import argparse
import numpy as np
import utils

def experiment(args):
    # environment setup
    env = gym.make(args.env)
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # here it is discrete, so we have n here as opposed to the dimension of the action
    
    # network setup
    network = QNetwork(args.gamma, state_dim, action_dim, args.hidden_sizes)
    
    # optimizer setup
    if args.env == 'CartPole-v0':
        optimizer = optim.RMSprop(network.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(network.parameters(), lr=args.lr)
    
    # target setup (if wanted)
    if args.target:
        target_network = QNetwork(args.gamma, state_dim, action_dim, args.hidden_sizes)
        target_network.load_state_dict(network.state_dict())
        target_network.eval()
    
    # buffer setup
    buffer = ReplayBuffer(maxsize=args.max_size)
    
    # training
    for i in tqdm(range(args.num_episodes)):
        # initial observation, cast into a torch tensor
        ob = torch.from_numpy(env.reset()).float()
        
        for t in count():
            with torch.no_grad():
                eps = utils.get_eps(args.eps, i)
                # TODO: Collect the action from the policy.
                action = network.get_action(ob, eps)  # Get action based on epsilon-greedy policy
            # Step the environment, convert everything to torch tensors
            n_ob, rew, done, _ = env.step(action)
            
            action = torch.tensor(action)
            n_ob = torch.from_numpy(n_ob).float()
            rew = torch.tensor([rew])
            
            # TODO: Add new experience to replay buffer.
            buffer.add_experience(ob, action, rew, n_ob, done)
            pass
            
            ob = n_ob
            
            if len(buffer) >= args.batch_size:
                if t % args.learning_freq == 0:
                    # TODO: Sample batch from replay buffer and optimize model via gradient descent.
                    # HINTS:
                    #   If we're using a target Q network (see get_args() for details), make sure to get the targets with the target network.
                    #   Make sure to 'zero_grad' the optimizer before performing gradient descent.
                    
                    # SAMPLE FROM BUFFER HERE
                    states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)
                    
                    
                    # COMPUTE Q VALUES HERE 
                    qs = network(states).gather(1, actions.unsqueeze(1))  # Action values from the Q network


                    # COMPUTE TARGET Q VALUES FROM BATCH HERE
                    if args.target:
                        next_qs = target_network.get_max_q(next_states)
                    else:
                        next_qs = network.get_max_q(next_states)

                    
                    # GRADIENT DESCENT HERE
                    targets = rewards + args.gamma * next_qs * (1 - dones)
                    loss = torch.nn.functional.mse_loss(qs.squeeze(1), targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            
            if done:
                # we are done, so we break out of the for loop here
                # feel free to log anything you want here during training.
                break
            
        
        # TODO: Update target based on args.target_update_freq.
        # See 'utils.py' and argparse for more information.
        if args.target and i % args.target_update_freq == 0:
            utils.update_target(network, target_network, args.tau)

    
    # save final agent
    save_path = args.save_path + '_' + args.env.lower() + '.pt'
    torch.save(network, save_path)
    
def get_args():
    parser = argparse.ArgumentParser(description='Q-Learning')
    
    # Environment args
    parser.add_argument('--env', default='CartPole-v0', help='name of environment')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--eps', type=float, default=0.999, help='epsilon parameter')
    
    # Network args
    parser.add_argument('--hidden_sizes', nargs='+', type=int, help='hidden sizes of Q network')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for Q function optimizer')
    parser.add_argument('--target', action='store_true', help='if we want to use a target network')
    parser.add_argument('--target_update_freq', type=int, default=10, help='how often we update the target network')
    parser.add_argument('--tau', type=float, default=1.0, help='target update parameter')
    
    # Replay buffer args
    parser.add_argument('--max_size', type=int, default=10000, help='max buffer size')
    
    # Training/saving args
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to run during training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--save_path', default='./trained_agent', help='agent save path')
    parser.add_argument('--learning_freq', type=int, default=1, help='how often to update the network after collecting experience')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = get_args()
    experiment(args)
    