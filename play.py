import utils
import sys

def demo_cpu(model):
    model = utils.torch.load(f'models/model_v{model[0]}_{model[1]}_{model[2]}')
    policy_net = utils.DQNetwork(model['hyper_params']['model_shape']).to(utils.device)
    policy_net.load_state_dict(model['policy_net'])

    try:
        env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', render_mode='human', fast_forward=False, vs_player=False)
        for _ in utils.count():
            utils.play(env, policy_net)
    except KeyboardInterrupt:
        env.close()
    
def demo_vs(model):
    model = utils.torch.load(f'models/model_v{model[0]}_{model[1]}_{model[2]}')
    policy_net = utils.DQNetwork(model['hyper_params']['model_shape']).to(utils.device)
    policy_net.load_state_dict(model['policy_net'])

    try:
        env = utils.FootsiesEnv(frame_delay=16, sync_mode='synced_blocking', render_mode='human', fast_forward=False, vs_player=True)
        for _ in utils.count():
            utils.play(env, policy_net)
    except KeyboardInterrupt:
        env.close()

VS_PLAYER = True
model = (11, 0, 6)
if VS_PLAYER:
    demo_vs(model)
else:
    demo_cpu(model)
