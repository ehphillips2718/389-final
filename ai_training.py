import utils

def tuning(models, env, version=0, episodes=100):
    load = version > 0
    for i in models:
        start_time = utils.time.time()
        model = utils.torch.load(f'models/model_v{version - 1}_{i}') if load else None
        hyper_params = model['hyper_params'] if load else {
            'model_shape':  (11, utils.np.random.randint(40, 200), utils.np.random.randint(20, 100), 8), # shape of model
            'num_episodes': 2000,                                                                        # number of episodes
            'memory_size':  10000,                                                                       # size of memory buffer
            'batch_size':   200,                                                                         # number of transitions sampled from the replay buffer
            'gamma':        0.1 * utils.np.random.random() + 0.9,                                        # discount factor
            'eps_start':    0.2 * utils.np.random.random() + 0.8,                                        # starting epsilon
            'eps_end':      0.1 * utils.np.random.random(),                                              # ending epsilon
            'eps_decay':    utils.np.random.randint(250, 2000),                                          # rate of exponential decay of epsilon, higher means a slower decay
            'tau':          0.01 * utils.np.random.random(),                                             # update rate of the target network
            'lr':           1e-3 * utils.np.random.random() + 1e-5,                                      # learning rate of the ``AdamW`` optimizer
            'num_actions':  0                                                                            # fixed to 0
        } 
        if load:
            hyper_params['num_episodes'] = episodes

        policy_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
        if load:
            policy_net.load_state_dict(model['policy_net'])
        target_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = utils.optim.AdamW(policy_net.parameters(), lr=hyper_params['lr'], amsgrad=True)
        if load:
            optimizer.load_state_dict(model['optimizer'])
        memory = model['memory'] if load else utils.ReplayMemory(hyper_params['memory_size'])

        results = model['results'] if load else {'length': [], 'win_loss': [], 'rewards': []}
        utils.ai_train(env, hyper_params, policy_net, target_net, optimizer, memory, results)

        model = {
            'results':      results, 
            'hyper_params': hyper_params, 
            'policy_net':   policy_net.state_dict(), 
            'optimizer':    optimizer.state_dict(),
            'memory':       memory
        }
        utils.torch.save(model, f'models/model_v{version}_{i}')

        end_time = utils.time.time()
        elapsed = end_time - start_time
        r_t = utils.np.array(results["win_loss"], dtype=utils.np.float32)
        avg = r_t[len(r_t) - 100: len(r_t)].mean(0)
        print(f'Model {i} completed in {elapsed:.2f} seconds with win percentage {avg:.2f}.')

env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', dense_reward=True, fast_forward=True)
models = [43]
tuning(models, env, version=1, episodes=50000)
env.close()