import gymnasium as gym
from gymnasium import spaces
from ..moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE
from ..envs.footsies import FootsiesEnv


class FootsiesNormalized(gym.ObservationWrapper):
    """Normalizes all observation space variables. Wrapper should be applied to the base FOOTSIES environment before any other observation wrapper

    Move frame durations will be between `0` and `1`, inclusive. `0` indicates the start of the move, while `1` indicates the end of it.
    The guard information of each player can also be set to be between `0` and `1`.
    """

    def __init__(self, env, normalize_guard: bool = True):
        super().__init__(env)

        if not isinstance(env, FootsiesEnv):
            raise ValueError("FootsiesNormalized wrapper should be applied to the base FOOTSIES environment")
        
        self.normalize_guard = normalize_guard

        self.observation_space: spaces.Dict = env.observation_space
        if self.normalize_guard:
            self.observation_space.spaces["guard"] = spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.observation_space.spaces["move_frame"] = spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.observation_space.spaces["position"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def observation(self, obs: dict) -> dict:
        obs = obs.copy()
        if self.normalize_guard:
            obs["guard"] = (obs["guard"][0] / 3.0, obs["guard"][1] / 3.0)
        obs["position"] = (obs["position"][0] / 4.6, obs["position"][1] / 4.6)
        obs["move_frame"] = (
            obs["move_frame"][0]
            / FOOTSIES_MOVE_INDEX_TO_MOVE[int(obs["move"][0])].value.duration,
            obs["move_frame"][1]
            / FOOTSIES_MOVE_INDEX_TO_MOVE[int(obs["move"][1])].value.duration
        )

        return obs

    @staticmethod
    def undo(obs: dict, normalized_guard: bool = True) -> dict:
        obs = obs.copy()
        if normalized_guard:
            obs["guard"] = (obs["guard"][0] * 3.0, obs["guard"][1] * 3.0)
        obs["position"] = (obs["position"][0] * 4.6, obs["position"][1] * 4.6)
        obs["move_frame"] = (
            obs["move_frame"][0]
            * FOOTSIES_MOVE_INDEX_TO_MOVE[int(obs["move"][0])].value.duration,
            obs["move_frame"][1]
            * FOOTSIES_MOVE_INDEX_TO_MOVE[int(obs["move"][1])].value.duration
        )

        return obs
