import utils

NUM_MODELS = 100
for i in range(NUM_MODELS):
    model = utils.torch.load(f'models/model_{i}')
    results      = model['results']
    hyper_params = model['hyper_params']

    policy_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
    target_net = utils.DQNetwork(hyper_params['model_shape']).to(utils.device)
    policy_net.load_state_dict(model['policy_net'])
    target_net.load_state_dict(model['target_net'])
    policy_net.eval() # set to eval mode (no more training)
    target_net.eval() # set to eval mode (no more training)

    # Baseline 240 = (at least) 4 seconds
    #utils.plot_results(results['length'], f'Model {i} Episode Length', 'Length', 240)
    #utils.plt.show()
    
    # Baseline 0.5 = 50/50 win/loss
    utils.plot_results(results['win_loss'], f'Model {i} Win/Loss', 'Win/Loss', 0.5)
    utils.plt.show()
    
    # Baseline 0 = 0 reward
    #utils.plot_results(results['rewards'], f'Model {i} Rewards', 'Rewards', 0)
    #utils.plt.show()
