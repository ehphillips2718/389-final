from gymnasium.envs.registration import register

register(
    id="FootsiesEnv-v0",
    entry_point="footsies_gym.envs.footsies:FootsiesEnv",
    nondeterministic=True,
)
