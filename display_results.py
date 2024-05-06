import utils


#best_models = []
#for i, j in MODELS:
#    print(f'Model {i}')
#    model = utils.torch.load(f'models/model_v2_{i}_{j}')
#    results      = model['results']
#    hyper_params = model['hyper_params']
#    print(hyper_params)
#
#    # Baseline 240 = (at least) 4 seconds
#    utils.plot_avg_results(results['length'], f'Model {i} Episode Length', 'Length', 240)
#    utils.plt.show()
#    
#    # Baseline 0.5 = 50% win percentage
#    utils.plot_avg_results(results['win_loss'], f'Model {i} Win Percentage', 'Win Percentage', 0.5)
#    utils.plt.show()
#
#    # Baseline 0.5 = 50% win percentage
#    #utils.plot_cummul_results(results['win_loss'], f'Model {i} Cumulative Wins', 'Cumulative Wins', 0.6)
#    #utils.plt.show()
#    
#    # Baseline 0 = 0 reward
#    utils.plot_avg_results(results['rewards'], f'Model {i} Rewards', 'Rewards', 0)
#    utils.plt.show()
#
#    r_t = utils.np.array(results["win_loss"], dtype=utils.np.float32)
#    avg = r_t[len(r_t) - 100: len(r_t)].mean()
#    best_models.append((i, avg))
#print(sorted(best_models, key=lambda x: x[1], reverse=True))

MODELS = [(0, 5), (0, 6)]
env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', render_mode='human', dense_reward=True, fast_forward=False, vs_player=True)
#env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', render_mode='human', dense_reward=True, fast_forward=False, opponent=(lambda obs, info: None))
#env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', dense_reward=True, fast_forward=True)
best_models = []
for i, j in MODELS:
    start_time = utils.time.time()
    model = utils.torch.load(f'models/model_v1_{i}_{j}')
    policy_net = utils.DQNetwork(model['hyper_params']['model_shape']).to(utils.device)
    policy_net.load_state_dict(model['policy_net'])
    utils.time.sleep(2)
    win_loss = []
    for k in range(5):
        #if k % 100 == 0: 
        #    print(f"Game {j} started.")
        win_loss.append(utils.play(env, policy_net))
    win_rate = utils.np.mean(win_loss)
    best_models.append((i, win_rate))
    end_time = utils.time.time()

    print(f'Model {i} completed in {(end_time - start_time):.2f} seconds with win percentage {win_rate}.')
env.close()
print(sorted(best_models, key=lambda x: x[1], reverse=True))