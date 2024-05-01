from footsies_gym.wrappers.normalization import FootsiesNormalized
from gymnasium.spaces.utils import unflatten
from gymnasium.spaces import Space
import numpy as np


def get_dict_obs_from_vector_obs(
    vector_obs: np.ndarray,
    flattened: bool = True,
    unflattenend_observation_space: Space = None,
    normalized: bool = True,
    normalized_guard: bool = True,
) -> dict:
    """
    Convert a FOOTSIES observation from a transformed version (with observation wrappers) into the original version.
    Doesn't work on observations that had frame skipping
    """

    dict_obs = {}

    if flattened:
        if unflattenend_observation_space is None:
            raise ValueError(
                "if argument vector_obs is flattened, then the unflattened observation space needs to be provided"
            )
        dict_obs = unflatten(unflattenend_observation_space, vector_obs)

    # If not flattened, we assume it's a dictionary
    elif isinstance(vector_obs, dict):
        dict_obs = vector_obs

    else:
        raise ValueError(
            f"if argument vector_obs is not flattened, it's assumed to be a dictionary (actual type: {type(vector_obs).__name__})"
        )

    if normalized:
        dict_obs = FootsiesNormalized.undo(dict_obs, normalized_guard=normalized_guard)

    return dict_obs
