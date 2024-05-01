import gymnasium as gym
from ..moves import FOOTSIES_MOVE_INDEX_TO_MOVE, FootsiesMove


class FootsiesStatistics(gym.Wrapper):
    """Collect statistics on the FOOTSIES environment. The environment that this wrapper receives should not be wrapped by observation wrappers"""

    def __init__(self, env):
        super().__init__(env)
        self._special_moves_per_episode = []
        self._special_moves_from_neutral_per_episode = []
        self._special_moves_per_episode_counter = 0
        self._special_moves_from_neutral_per_episode_counter = 0
        self._prev_p1_move = None  # used to make sure special moves are only counted when they are performed, and not every time step they are active

    def _get_p1_move(self, obs) -> FootsiesMove:
        p1_move_index = obs["move"][0]
        return FOOTSIES_MOVE_INDEX_TO_MOVE[p1_move_index]

    def reset(self, *, seed: int = None, options: dict = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_p1_move = self._get_p1_move(obs)

        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        p1_move = self._get_p1_move(next_obs)
        if self._prev_p1_move != p1_move and p1_move in {
            FootsiesMove.B_SPECIAL,
            FootsiesMove.N_SPECIAL,
        }:
            self._special_moves_per_episode_counter += 1

            if self._prev_p1_move not in {
                FootsiesMove.B_ATTACK,
                FootsiesMove.N_ATTACK,
            }:
                self._special_moves_from_neutral_per_episode_counter += 1
        
        self._prev_p1_move = p1_move

        if terminated or truncated:
            self._special_moves_per_episode.append(
                self._special_moves_per_episode_counter
            )
            self._special_moves_per_episode_counter = 0

        return next_obs, reward, terminated, truncated, info

    @property
    def metric_special_moves_per_episode(self):
        return self._special_moves_per_episode

    @property
    def metric_special_moves_from_neutral_per_episode(self):
        return self._special_moves_from_neutral_per_episode
    
    def report(self):
        total_episodes = len(self.metric_special_moves_per_episode)
        total_special_moves = sum(self.metric_special_moves_per_episode)
        total_special_moves_from_neutral = sum(self.metric_special_moves_from_neutral_per_episode)

        print("Report")
        print(" Special moves")
        print(f"  Average: {total_special_moves / total_episodes}")
        print(f"  Total: {total_special_moves}")
        print(" Special moves")
        print(f"  Average: {total_special_moves_from_neutral / total_episodes}")
        print(f"  Total: {total_special_moves_from_neutral}")