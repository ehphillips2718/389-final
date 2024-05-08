import utils

def training_results(models):
    for indices in models:
        model = utils.torch.load(f'models/model_v{indices[0]}_{indices[1]}_{indices[2]}')
        results = model['results']
    
        # Baseline 240 = (at least) 4 seconds
        utils.plot_avg_results(results['length'], f'Model {indices} Episode Length', 'Length', 240)
        utils.plt.show()
        
        # Baseline 0.5 = 50% win percentage
        utils.plot_avg_results(results['win_loss'], f'Model {indices} Win Percentage', 'Win Percentage', 0.5)
        utils.plt.show()
    
        # Baseline 0.5 = 50% win percentage
        utils.plot_cummul_results(results['win_loss'], f'Model {indices} Cumulative Wins', 'Cumulative Wins', 0.5)
        utils.plt.show()
        
        # Baseline 0 = 0 reward
        utils.plot_avg_results(results['rewards'], f'Model {indices} Rewards', 'Rewards', 0)
        utils.plt.show()

def test(env, policy_net):
    results = {'length': [], 'win_loss': [], 'rewards': [], 'actions': []}
    for t in range(10000):
        length, reward, won, actions = utils.play(env, policy_net)
        results['length'].append(length)
        results['rewards'].append(reward)
        results['win_loss'].append(won)
        results['actions'] += actions
    return results

def test_results(env, models):
    for indices in models:
        model = utils.torch.load(f'models/model_v{indices[0]}_{indices[1]}_{indices[2]}')
        policy_net = utils.DQNetwork(model['hyper_params']['model_shape']).to(utils.device)
        policy_net.load_state_dict(model['policy_net'])
        results = test(env, policy_net)
    
        # Baseline 240 = (at least) 4 seconds
        utils.plot_avg_results(results['length'], f'Model {indices} Episode Length', 'Length', 240)
        utils.plt.show()
        
        # Baseline 0.5 = 50% win percentage
        utils.plot_avg_results(results['win_loss'], f'Model {indices} Win Percentage', 'Win Percentage', 0.5)
        utils.plt.show()
    
        # Baseline 0.5 = 50% win percentage
        utils.plot_cummul_results(results['win_loss'], f'Model {indices} Cumulative Wins', 'Cumulative Wins', 0.5)
        utils.plt.show()
        
        # Baseline 0 = 0 reward
        utils.plot_avg_results(results['rewards'], f'Model {indices} Rewards', 'Rewards', 0)
        utils.plt.show()

        action_counts = [results['actions'].count(i) for i in range(8)]

        utils.plt.bar(utils.np.arange(len(action_counts)), action_counts)
        utils.plt.title("Action Frequency")
        utils.plt.xlabel("Action #")
        utils.plt.ylabel("Frequency")
        utils.plt.show()

MODELS = [(11, 0, 11), (11, 0, 6), (13, 17, 6), (11, 0, 15), (13, 11, 0), (11, 0, 5)]
MODELS = [(11, 0, 6)]

env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', dense_reward=True, fast_forward=True)
models = [(11, 0, 6)]
test_results(env, models)
env.close()