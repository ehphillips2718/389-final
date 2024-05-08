import utils

#MODELS = [(7, 0, 15), (8, 0, 15)]
#best_models = []
#for i, j, k in MODELS:
#    print(f'Model ({i},{j},{k})')
#    model = utils.torch.load(f'models/model_v{i}_{j}_{k}')
#    results      = model['results']
#    #hyper_params = model['hyper_params']
#    print(utils.np.array(results["win_loss"], dtype=utils.np.float32).mean(0))
#
#    # Baseline 240 = (at least) 4 seconds
#    #utils.plot_avg_results(results['length'], f'Model {i} Episode Length', 'Length', 240)
#    #utils.plt.show()
#    
#    # Baseline 0.5 = 50% win percentage
#    #utils.plot_avg_results(results['win_loss'], f'Model {i} Win Percentage', 'Win Percentage', 0.5)
#    #utils.plt.show()
#
#    # Baseline 0.5 = 50% win percentage
#    #utils.plot_cummul_results(results['win_loss'], f'Model {i} Cumulative Wins', 'Cumulative Wins', 0.6)
#    #utils.plt.show()
#    
#    # Baseline 0 = 0 reward
#    #utils.plot_avg_results(results['rewards'], f'Model {i} Rewards', 'Rewards', 0)
#    #utils.plt.show()
#
#    #r_t = utils.np.array(results["win_loss"], dtype=utils.np.float32)
#    #avg = r_t[len(r_t) - 100: len(r_t)].mean()
#    #best_models.append((i, avg))
##print(sorted(best_models, key=lambda x: x[1], reverse=True))

MODELS = [(11, 0, 11), (11, 0, 6), (13, 17, 6), (11, 0, 15), (13, 11, 0), (11, 0, 5)]
MODELS = [(11, 0, 5)]
#env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', render_mode='human', dense_reward=True, fast_forward=False, vs_player=True)
#env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', render_mode='human', dense_reward=True, fast_forward=False, opponent=(lambda obs, info: None))
env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', dense_reward=True, fast_forward=True)
best_models = []
for model in MODELS:
    start_time = utils.time.time()
    model_p1 = utils.torch.load(f'models/model_v{model[0]}_{model[1]}_{model[2]}')
    policy_net_p1 = utils.DQNetwork(model_p1['hyper_params']['model_shape']).to(utils.device)
    policy_net_p1.load_state_dict(model_p1['policy_net'])
    #utils.time.sleep(2)
    win_loss = []
    for t in range(10000):
        if t % 100 == 0: 
            print(f"Game {t} started.")
        win_loss.append(utils.play(env, policy_net_p1))
    win_rate = utils.np.mean(win_loss)
    best_models.append((model, win_rate))
    end_time = utils.time.time()
    print(f'Model {model} completed in {(end_time - start_time):.2f} seconds with win percentage {win_rate}.')
env.close()
print(sorted(best_models, key=lambda x: x[1], reverse=True))