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

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.lin1(x))
        out = self.relu(self.lin2(out))
        return self.lin3(out).clamp(-2, 2)
    
# Modify this function to add more features
def state_to_tensor(state, p1_only=True):
    state_p1 = torch.tensor([
        state['guard'][0],      state['guard'][1],      
        state['move'][0],       state['move'][1],
        state['move_frame'][0], state['move_frame'][1], 
        move_list[state['move'][0]].in_state(state['move_frame'][0]), 
        move_list[state['move'][1]].in_state(state['move_frame'][1]), 
        state['position'][0], state['position'][1],
        state['position'][1] - state['position'][0]
    ], dtype=torch.float32, device=device)

    if p1_only:
        return state_p1
    else:
        state_p2 = torch.tensor([
            state['guard'][1],      state['guard'][0],      
            state['move'][1],       state['move'][0],
            state['move_frame'][1], state['move_frame'][0], 
            move_list[state['move'][1]].in_state(state['move_frame'][1]), 
            move_list[state['move'][0]].in_state(state['move_frame'][0]), 
            -state['position'][1], -state['position'][0],
            state['position'][1] - state['position'][0]
        ], dtype=torch.float32, device=device) 
        return state_p1, state_p2
    
def select_action_training(state, policy_net, hyper_params):
    #with torch.no_grad():
    #    print(move_list[int(state.cpu().detach().numpy()[2])].value.startup, policy_net(state).cpu().detach().numpy())
    sample = random.random()
    eps_threshold = hyper_params['eps_end'] + (hyper_params['eps_start'] - hyper_params['eps_end']) * math.exp(-1. * hyper_params['num_actions'] / hyper_params['eps_decay'])
    hyper_params['num_actions'] += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(keepdim=True)
    else:
        return torch.randint(8, (1,), device=device).detach()
    
def select_action(state, policy_net, multinomial=False):
    with torch.no_grad():
        #print(policy_net(state).cpu().detach().numpy(), policy_net(state).add(2).exp().multinomial(1).cpu().detach().numpy())
        return policy_net(state).argmax(keepdim=True) if multinomial else policy_net(state).add(2).exp().multinomial(1)
    
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
    expected_state_action_values = reward_batch + hyper_params['gamma'] * target_net(next_state_batch).max(1).values.reshape((-1, 1))

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def ai_train(env, hyper_params, policy_net, target_net, optimizer, memory, results):
    for episode in range(hyper_params['num_episodes']):
        if episode % 100 == 0:
            print(f'Episode {episode} started.')
        state, _ = env.reset()
        state = state_to_tensor(state)
        for i in count():
            action = select_action_training(state, policy_net, hyper_params)
            next_state, reward, terminated, _, _ = env.step(index_to_action(action))

            next_state = state_to_tensor(next_state)
            reward = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward)

            optimize_model(policy_net, target_net, optimizer, memory, hyper_params)

            # Manual soft update: θ′ ← τθ + (1 − τ)θ′
            policy_net_state_dict = policy_net.state_dict()
            target_net_state_dict = target_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = hyper_params['tau'] * policy_net_state_dict[key] + (1 - hyper_params['tau']) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            state = next_state

            # Record results of episode
            if terminated:
                reward = reward.cpu()[0]
                won = env.delayed_frame_queue[-1].p2Vital == 0
                results['length'].append(i)
                results['win_loss'].append(won)
                results['rewards'].append(reward)
                break

def self_train(env, hyper_params, policy_net, target_net, optimizer, memory_p1, memory_p2, results):
    for episode in range(hyper_params['num_episodes']):
        if episode % 100 == 0:
            print(f'Episode {episode} started.')
        state, _ = env.reset()
        state_p1, state_p2 = state_to_tensor(state, p1_only=False)
        for i in count():
            action_p1 = select_action_training(state_p1, policy_net, hyper_params)
            action_p2 = select_action_training(state_p2, policy_net, hyper_params)

            env.opponent = lambda obs, info: index_to_action(action_p2)
            next_state, reward, terminated, _, _ = env.step(index_to_action(action_p1))

            next_state_p1, next_state_p2 = state_to_tensor(next_state, p1_only=False)
            reward_p1 = torch.tensor([reward], device=device)
            reward_p2 = torch.tensor([-reward], device=device)

            memory_p1.push(state_p1, action_p1, next_state_p1, reward_p1)
            memory_p2.push(state_p2, action_p1, next_state_p2, reward_p2)

            optimize_model(policy_net, target_net, optimizer, memory_p1, hyper_params)
            optimize_model(policy_net, target_net, optimizer, memory_p2, hyper_params)

            # Manual soft update: θ′ ← τθ + (1 − τ)θ′
            policy_net_state_dict = policy_net.state_dict()
            target_net_state_dict = target_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = hyper_params['tau'] * policy_net_state_dict[key] + (1 - hyper_params['tau']) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            state_p1 = next_state_p1
            state_p2 = next_state_p2

            # Record results of episode
            if terminated:
                won = env.delayed_frame_queue[-1].p2Vital == 0
                results['length'].append(i)
                results['win_loss'].append(won)
                results['rewards'].append(reward)
                break

def vs_train(env, hyper_params, policy_net, target_net, optimizer, memory_p1, memory_p2, results):
    for episode in range(hyper_params[0]['num_episodes']):
        if episode % 100 == 0:
            print(f'Episode {episode} started.')
        state, _ = env.reset()
        state_p1, state_p2 = state_to_tensor(state, p1_only=False)
        for i in count():
            action_p1 = select_action_training(state_p1, policy_net[0], hyper_params[0])
            action_p2 = select_action_training(state_p2, policy_net[1], hyper_params[1])

            env.opponent = lambda obs, info: index_to_action(action_p2)
            next_state, reward, terminated, _, _ = env.step(index_to_action(action_p1))

            next_state_p1, next_state_p2 = state_to_tensor(next_state, p1_only=False)
            reward_p1 = torch.tensor([reward], device=device)
            reward_p2 = torch.tensor([-reward], device=device)

            memory_p1[0].push(state_p1, action_p1, next_state_p1, reward_p1)
            memory_p2[0].push(state_p2, action_p2, next_state_p2, reward_p2)
            memory_p1[1].push(state_p1, action_p1, next_state_p1, reward_p1)
            memory_p2[1].push(state_p2, action_p2, next_state_p2, reward_p2)

            optimize_model(policy_net[0], target_net[0], optimizer[0], memory_p1[0], hyper_params[0])
            optimize_model(policy_net[0], target_net[0], optimizer[0], memory_p2[0], hyper_params[0])
            optimize_model(policy_net[1], target_net[1], optimizer[1], memory_p1[1], hyper_params[1])
            optimize_model(policy_net[1], target_net[1], optimizer[1], memory_p2[1], hyper_params[1])

            # Manual soft update: θ′ ← τθ + (1 − τ)θ′
            policy_net_state_dict_p1 = policy_net[0].state_dict()
            target_net_state_dict_p1 = target_net[0].state_dict()
            for key in policy_net_state_dict_p1:
                target_net_state_dict_p1[key] = hyper_params[0]['tau'] * policy_net_state_dict_p1[key] + (1 - hyper_params[0]['tau']) * target_net_state_dict_p1[key]
            target_net[0].load_state_dict(target_net_state_dict_p1)

            policy_net_state_dict_p2 = policy_net[1].state_dict()
            target_net_state_dict_p2 = target_net[1].state_dict()
            for key in policy_net_state_dict_p2:
                target_net_state_dict_p2[key] = hyper_params[1]['tau'] * policy_net_state_dict_p2[key] + (1 - hyper_params[1]['tau']) * target_net_state_dict_p2[key]
            target_net[1].load_state_dict(target_net_state_dict_p2)

            state_p1 = next_state_p1
            state_p2 = next_state_p2

            # Record results of episode
            if terminated:
                won = env.delayed_frame_queue[-1].p2Vital == 0
                results['length'].append(i)
                results['win_loss'].append(won)
                results['rewards'].append(reward)
                break

def play(env, policy_net):
    state, _ = env.reset()
    state = state_to_tensor(state)
    for t in count():
        action = select_action(state, policy_net)
        next_state, reward, terminated, _, _ = env.step(index_to_action(action))

        next_state = state_to_tensor(next_state)
        reward = torch.tensor([reward], device=device)
        state = next_state

        if terminated:
            reward = reward.cpu()[0]
            won = env.delayed_frame_queue[-1].p2Vital == 0
            break
    return won

def self_play(env, policy_net):
    state, _ = env.reset()
    state_p1, state_p2 = state_to_tensor(state, p1_only=False)
    for t in count():
        action = select_action(state_p1, policy_net)
        env.opponent = lambda obs, info: index_to_action(select_action(state_p2, policy_net))
        next_state, reward, terminated, _, _ = env.step(index_to_action(action))

        next_state_p1, next_state_p2 = state_to_tensor(next_state, p1_only=False)
        reward = torch.tensor([reward], device=device)
        state_p1 = next_state_p1
        state_p2 = next_state_p2

        if terminated:
            reward = reward.cpu()[0]
            won = env.delayed_frame_queue[-1].p2Vital == 0
            break
    return won

def vs_play(env, policy_net_p1, policy_net_p2):
    state, _ = env.reset()
    state_p1, state_p2 = state_to_tensor(state, p1_only=False)
    for t in count():
        action_p1 = select_action(state_p1, policy_net_p1)
        action_p2 = select_action(state_p2, policy_net_p2)
        env.opponent = lambda obs, info: index_to_action(action_p2)
        next_state, reward, terminated, _, _ = env.step(index_to_action(action_p1))

        next_state_p1, next_state_p2 = state_to_tensor(next_state, p1_only=False)
        reward = torch.tensor([reward], device=device)
        state_p1 = next_state_p1
        state_p2 = next_state_p2

        if terminated:
            reward = reward.cpu()[0]
            won = env.delayed_frame_queue[-1].p2Vital == 0
            break
    return won
    
def plot_avg_results(results, title, y_label, baseline, avg_length=100, noisy_results=False):
    plt.figure(1)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)

    r_t = np.array(results, dtype=np.float32)
    plt.plot(np.full(len(r_t), baseline), color='b', label=f'Baseline: {baseline}')
    if noisy_results:
        plt.plot(r_t, color='k', label='Noisy Results')
    
    r_means = [r_t[max(0, i+1 - avg_length):i+1].mean() for i in range(len(r_t))]
    plt.plot(r_means, color='g', label='Running Average Results')
    plt.plot(np.full(len(r_t), r_means[-1]), color='r', label=f'Final Average: {r_means[-1]}')
    plt.legend()

    if is_ipython:
        display.display(plt.gcf())

def plot_cummul_results(results, title, y_label, baseline, noisy_results=False):
    plt.figure(1)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)

    r_t = np.array(results, dtype=np.float32)
    plt.plot(baseline * np.arange(len(r_t)), color='b', label=f'Baseline: {baseline}')
    
    r_means = [r_t[0:i+1].sum() for i in range(len(r_t))]
    plt.plot(r_means, color='g', label='Running Cumulative Results')
    plt.legend()

    if is_ipython:
        display.display(plt.gcf())