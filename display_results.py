import utils

MODELS = range(200)
best_models = []
best_hyper_params = {
    'model_shape':  [],
    'num_episodes': [],
    'memory_size':  [],
    'batch_size':   [],
    'gamma':        [],
    'eps_start':    [],
    'eps_end':      [],
    'eps_decay':    [],
    'tau':          [],
    'lr':           [],
    'num_actions':  [],
}
for i in MODELS:
    model = utils.torch.load(f'models/model_{i}')
    results      = model['results']
    hyper_params = model['hyper_params']

    policy_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
    policy_net.load_state_dict(model['policy_net'])

    # Baseline 240 = (at least) 4 seconds
    #utils.plot_results(results['length'], f'Model {i} Episode Length', 'Length', 240)
    #utils.plt.show()
    
    # Baseline 0.5 = 50/50 win/loss
    utils.plot_results(results['win_loss'], f'Model {i} Win/Loss', 'Win/Loss', 0.5)
    utils.plt.show()
    
    # Baseline 0 = 0 reward
    #utils.plot_results(results['rewards'], f'Model {i} Rewards', 'Rewards', 0)
    #utils.plt.show()

    r_t = utils.np.array(results["win_loss"], dtype=utils.np.float32)
    avg = r_t[len(r_t) - 100: len(r_t)].mean()
    if avg > 0.70:
        best_models.append(i)
        for key, value in hyper_params.items():
            best_hyper_params[key].append(value)

print(best_models)

for i in range(4):
    best_hyper_params[f'model_shape_{i}'] = [shape[i] for shape in best_hyper_params['model_shape']]

for key, value in best_hyper_params.items():
    print(key, min(value), max(value))

