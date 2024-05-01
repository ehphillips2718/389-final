import gymnasium as gym
from gymnasium import spaces
from ..moves import FOOTSIES_MOVE_INDEX_TO_MOVE, FootsiesMove


class FootsiesFrameSkipped(gym.Wrapper):
    """
    Skip environment time steps on which the agent can't act (such as when performing a non-instantaneous move or getting hit).
    Player 1's current move ID and progress will be removed from the observations

    Should be applied on top of any other FOOTSIES observation wrapper, but not other observation wrappers
    """

    def __init__(self, env):
        super().__init__(env)

        # Assumed to be a dictionary, since 'env' should be either FootsiesEnv or one of FOOTSIES's observation wrappers
        wrapped_observation_space = env.observation_space

        move_frame_low = wrapped_observation_space["move_frame"].low[1]
        move_frame_high = wrapped_observation_space["move_frame"].high[1]
        self.observation_space = spaces.Dict(
            {
                "guard": wrapped_observation_space["guard"],
                "move": wrapped_observation_space["move"],
                "move_frame": spaces.Box(
                    low=move_frame_low, high=move_frame_high, shape=(1,)
                ),
                "position": wrapped_observation_space["position"],
            }
        )

        # At the moment the agent is hit or hits the opponent, if frame skip is enabled,
        # then the agent will not receive the reward immediately, and so we should accumulate it
        self._frame_skip_retained_reward = 0.0

    def _frame_skip_obs(self, state_dict: dict) -> dict:
        """From the extracted observation data, transform it in case we are using frame skipping"""
        return {
            "guard": state_dict["guard"],
            "move": state_dict["move"],
            "move_frame": state_dict["move_frame"][1],
            "position": state_dict["position"],
        }

    def _is_obs_skippable(self, state_dict: dict) -> bool:
        """From the extracted observation data, check whether the observation is skippable, i.e. the agent can't act on it"""
        p1_move = FOOTSIES_MOVE_INDEX_TO_MOVE[state_dict["move"][0]]
        p2_move = FOOTSIES_MOVE_INDEX_TO_MOVE[state_dict["move"][1]]
        hit_guard_moves = {FootsiesMove.DAMAGE, FootsiesMove.GUARD_STAND, FootsiesMove.GUARD_CROUCH, FootsiesMove.GUARD_M, FootsiesMove.GUARD_BREAK}

        return (
            # player 1 is in the middle of a move (that hasn't hit the opponent yet!). We are assuming the move's progress is simplified (always 0 for instantaneous actions)
            (state_dict["move_frame"][0] != 0.0 and (p2_move not in hit_guard_moves))
            # player 1 is being hit
            or p1_move == FootsiesMove.DAMAGE
        )

    def reset(self, *, seed, options):
        obs, info = self.env.reset(seed=seed, options=options)

        # We assume there is no need for frame skipping on the first state
        return self._frame_skip_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Skip irrelevant environment steps until relevant state or termination/truncation
        skipped = False
        self._frame_skip_retained_reward += reward
        if self._is_obs_skippable(obs) and not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.step((False, False, False))
            self._frame_skip_retained_reward += reward
            skipped = True

        reward = self._frame_skip_retained_reward
        obs = self._frame_skip_obs(obs) if not skipped else obs
        self._frame_skip_retained_reward = 0

        return obs, reward, terminated, truncated, info
