from footsies_gym.envs.footsies import FootsiesEnv, FootsiesMove, FOOTSIES_MOVE_ID_TO_INDEX

import time
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
move_list = list(FootsiesMove)
bit_array = torch.tensor([1, 2, 4], dtype=torch.int64, device=device)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNetwork(nn.Module):
    def __init__(self, model_shape):
        super().__init__()

        self.lin1 = nn.Linear(model_shape[0], model_shape[1])
        self.lin2 = nn.Linear(model_shape[1], model_shape[2])
        self.lin3 = nn.Linear(model_shape[2], model_shape[3])

        self.relu    = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.lin1(x))
        out = self.relu(self.lin2(out))
        return self.lin3(out)
    
# Modify this function to add more features
def state_to_tensor(state):
    new_state = torch.tensor([
        state['guard'][0],      state['guard'][1],      
        state['move'][0],       state['move'][1],
        state['move_frame'][0], state['move_frame'][1], 
        move_list[state['move'][0]].in_state(state['move_frame'][0]), 
        move_list[state['move'][1]].in_state(state['move_frame'][1]), 
        state['position'][0], state['position'][1],
        state['position'][1] - state['position'][0]
    ], dtype=torch.float32, device=device)
    return new_state
    
def select_action(state, policy_net, hyper_params):
    sample = random.random()
    eps_threshold = hyper_params['eps_end'] + (hyper_params['eps_start'] - hyper_params['eps_end']) * math.exp(-1. * hyper_params['num_actions'] / hyper_params['eps_decay'])
    hyper_params['num_actions'] += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return nn.Softmax(0)(policy_net(state)).multinomial(1)
    else:
        return torch.randint(8, (1,), device=device)
    
def index_to_action(index):
    return index.broadcast_to(3).bitwise_and(bit_array).eq(bit_array)
    
def optimize_model(policy_net, target_net, optimizer, memory, hyper_params):
    if len(memory) < hyper_params['batch_size']:
        return
    transitions = memory.sample(hyper_params['batch_size'])
    batch = Transition(*zip(*transitions))
    
    state_batch      = torch.stack(batch.state)
    action_batch     = torch.stack(batch.action)
    next_state_batch = torch.stack(batch.next_state)
    reward_batch     = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    expected_state_action_values = (hyper_params['gamma'] * target_net(next_state_batch).max(1).values.reshape((-1, 1))) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
def plot_results(results, title, y_label, baseline, noisy_results=False):
    plt.figure(1)
    r_t = np.array(results, dtype=np.float32)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.plot(np.full(len(r_t), baseline), color='b', label=f'Basline: {baseline}')
    if noisy_results:
        plt.plot(r_t, color='k', label='Noisy Results')
    
    # Take 100 episode averages and plot them too
    MAX_AVG_LENGTH = 100
    r_means  = [r_t[max(0, i - MAX_AVG_LENGTH):i+1].mean(0) for i in range(len(r_t))]
    plt.plot(r_means, color='g', label='Running Average Results')
    plt.plot(np.full(len(r_t), r_means[-1]), color='r', label=f'Final Average: {r_means[-1]}')
    plt.legend()

    if is_ipython:
        display.display(plt.gcf())