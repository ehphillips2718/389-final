import utils


env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', dense_reward=True, fast_forward=True)

NUM_MODELS = 100
for i in range(NUM_MODELS):
    start_time = utils.time.time()
    hyper_params = {
        'model_shape':  (11, utils.np.random.randint(40, 200), utils.np.random.randint(20, 100), 8), # shape of model
        'num_episodes': 200,                                                                         # number of episodes
        'memory_size':  utils.np.random.randint(1000, 10000),                                        # size of memory buffer
        'batch_size':   utils.np.random.randint(64, 256),                                            # number of transitions sampled from the replay buffer
        'gamma':        0.1 * utils.np.random.random() + 0.9,                                        # discount factor
        'eps_start':    0.2 * utils.np.random.random() + 0.8,                                        # starting epsilon
        'eps_end':      0.1 * utils.np.random.random(),                                              # ending epsilon
        'eps_decay':    utils.np.random.randint(500, 1000),                                          # rate of exponential decay of epsilon, higher means a slower decay
        'tau':          0.01 * utils.np.random.random(),                                             # update rate of the target network
        'lr':           1e-3 * utils.np.random.random() + 1e-5,                                      # learning rate of the ``AdamW`` optimizer
        'num_actions':  0                                                                            # fixed to 0
    }
    policy_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
    target_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = utils.optim.AdamW(policy_net.parameters(), lr=hyper_params['lr'], amsgrad=True) # amsgrad?
    memory = utils.ReplayMemory(hyper_params['memory_size'])

    results = {'length': [], 'win_loss': [], 'rewards': []}
    for episode in range(hyper_params['num_episodes']):
        state, _ = env.reset()
        state = utils.state_to_tensor(state)
        for t in utils.count():
            action = utils.select_action(state, policy_net, hyper_params)
            next_state, reward, terminated, _, _ = env.step(utils.index_to_action(action))

            next_state = utils.state_to_tensor(next_state)
            reward = utils.torch.tensor([reward], device=utils.device)
            memory.push(state, action, next_state, reward)
            state = next_state

            utils.optimize_model(policy_net, target_net, optimizer, memory, hyper_params)

            # Manual soft update: θ′ ← τθ + (1 − τ)θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = hyper_params['tau'] * policy_net_state_dict[key] + (1 - hyper_params['tau']) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            # Record results of episode
            if terminated:
                reward = reward.cpu()[0]
                won = next_state.cpu()[2] != 0
                results['length'].append(t)
                results['win_loss'].append(won)
                results['rewards'].append(reward)
                #print(f"Episode: {episode}, Length: {t}, Win/Loss: {won}, Reward: {reward}")
                break

    model = {
        'results':      results, 
        'hyper_params': hyper_params, 
        'policy_net':   policy_net.state_dict(), 
        'target_net':   target_net.state_dict(), 
        'optimizer':    optimizer.state_dict()
    }
    utils.torch.save(model, f'models/model_{i}')

    end_time = utils.time.time()
    elapsed = end_time - start_time
    r_t = utils.np.array(results["win_loss"], dtype=utils.np.float32)
    avg = r_t[len(r_t) - 100: len(r_t)].mean(0)
    print(f'Model {i} completed in {elapsed:.2f} seconds with win/loss {avg:.2f}.\n{hyper_params = }')

env.close()