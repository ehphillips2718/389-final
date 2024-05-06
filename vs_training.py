import utils

def tuning(index_p1, index_p2, env, version=0, episodes=100):
    start_time = utils.time.time()
    model_p1 = utils.torch.load(f'models/model_v{version - 1}_{index_p1[0]}_{index_p1[1]}')
    model_p2 = utils.torch.load(f'models/model_v{version - 1}_{index_p2[0]}_{index_p2[1]}')
    hyper_params_p1 = {
        'model_shape':  model_p1['hyper_params']['model_shape'], # shape of model
        'num_episodes': episodes,                                # number of episodes
        'memory_size':  10000,                                   # size of memory buffer
        'batch_size':   200,                                     # number of transitions sampled from the replay buffer
        'gamma':        0.1 * utils.np.random.random() + 0.9,    # discount factor
        'eps_start':    0.2 * utils.np.random.random() + 0.8,    # starting epsilon
        'eps_end':      0.1 * utils.np.random.random(),          # ending epsilon
        'eps_decay':    utils.np.random.randint(250, 2000),      # rate of exponential decay of epsilon, higher means a slower decay
        'tau':          0.01 * utils.np.random.random(),         # update rate of the target network
        'lr':           1e-3 * utils.np.random.random() + 1e-5,  # learning rate of the ``AdamW`` optimizer
        'num_actions':  0                                        # fixed to 0
    } 
    hyper_params_p2 = {
        'model_shape':  model_p2['hyper_params']['model_shape'], # shape of model
        'num_episodes': episodes,                                # number of episodes
        'memory_size':  10000,                                   # size of memory buffer
        'batch_size':   200,                                     # number of transitions sampled from the replay buffer
        'gamma':        0.1 * utils.np.random.random() + 0.9,    # discount factor
        'eps_start':    0.2 * utils.np.random.random() + 0.8,    # starting epsilon
        'eps_end':      0.1 * utils.np.random.random(),          # ending epsilon
        'eps_decay':    utils.np.random.randint(250, 2000),      # rate of exponential decay of epsilon, higher means a slower decay
        'tau':          0.01 * utils.np.random.random(),         # update rate of the target network
        'lr':           1e-3 * utils.np.random.random() + 1e-5,  # learning rate of the ``AdamW`` optimizer
        'num_actions':  0                                        # fixed to 0
    } 

    # P1 Model
    policy_net_p1 = utils.DQNetwork(hyper_params_p1['model_shape']).to(utils.device)
    policy_net_p1.load_state_dict(model_p1['policy_net'])
    target_net_p1 = utils.DQNetwork(hyper_params_p1['model_shape']).to(utils.device)
    target_net_p1.load_state_dict(policy_net_p1.state_dict())
    optimizer_p1 = utils.optim.AdamW(policy_net_p1.parameters(), lr=hyper_params_p1['lr'], amsgrad=True)

    # P2 Model
    policy_net_p2 = utils.DQNetwork(hyper_params_p2['model_shape']).to(utils.device)
    policy_net_p2.load_state_dict(model_p2['policy_net'])
    target_net_p2 = utils.DQNetwork(hyper_params_p2['model_shape']).to(utils.device)
    target_net_p2.load_state_dict(policy_net_p2.state_dict())
    optimizer_p2 = utils.optim.AdamW(policy_net_p2.parameters(), lr=hyper_params_p2['lr'], amsgrad=True)

    memory_p1_1 = utils.ReplayMemory(hyper_params_p1['memory_size'])
    memory_p1_2 = utils.ReplayMemory(hyper_params_p1['memory_size'])
    memory_p2_1 = utils.ReplayMemory(hyper_params_p2['memory_size'])
    memory_p2_2 = utils.ReplayMemory(hyper_params_p2['memory_size'])
    results_p1 = {'length': [], 'win_loss': [], 'rewards': []}
    utils.vs_train(env, (hyper_params_p1, hyper_params_p2), (policy_net_p1, policy_net_p2), (target_net_p2, target_net_p2), (optimizer_p1, optimizer_p2), (memory_p1_1, memory_p2_1), (memory_p1_2, memory_p2_2), results_p1)
    results_p2 = {'length': results_p1['length'], 'win_loss': [not x for x in results_p1['win_loss']], 'rewards': [-x for x in results_p1['rewards']]}

    model_p1 = {
        'results':      results_p1, 
        'hyper_params': hyper_params_p1, 
        'policy_net':   policy_net_p1.state_dict(), 
        'optimizer':    optimizer_p1.state_dict()
    }
    model_p2 = {
        'results':      results_p2, 
        'hyper_params': hyper_params_p2, 
        'policy_net':   policy_net_p2.state_dict(), 
        'optimizer':    optimizer_p2.state_dict()
    }
    utils.torch.save(model_p1, f'models/model_v{version}_{index_p1[0]}_{index_p1[1]}')
    utils.torch.save(model_p2, f'models/model_v{version}_{index_p2[0]}_{index_p2[1]}')

    end_time = utils.time.time()
    elapsed = end_time - start_time
    r_t = utils.np.array(results_p1["win_loss"], dtype=utils.np.float32)
    avg = r_t[len(r_t) - 100: len(r_t)].mean(0)
    print(f'Model ({index_p1[0]},{index_p1[1]}) vs Model ({index_p2[0]},{index_p2[1]}) completed in {elapsed:.2f} seconds with win percentage {avg:.2f}.')

env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', dense_reward=True, fast_forward=True, opponent=(lambda obs, info: None))
index_p1 = (11, 0)
index_p2 = (17, 6)
tuning(index_p1, index_p2, env, version=0, episodes=100)
env.close()