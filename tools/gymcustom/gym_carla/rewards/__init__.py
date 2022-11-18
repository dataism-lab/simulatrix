from typing import Dict, Callable

import numpy as np

from tools.gymcustom.gym_carla.rewards.basic_reward import reward_function
from tools.gymcustom.gym_carla.rewards.deepracer_reward_v2 import reward_function as deepracer_reward_v2
from tools.gymcustom.gym_carla.rewards.no_zigzag_reward import Reward as NoZigZag
from tools.gymcustom.gym_carla.rewards.two_d_reward import reward_function as two_d_reward


def reward_factory(reward_name: str, waypoints: np.ndarray = np.zeros(0)) -> Callable[[Dict], float]:
    if reward_name == 'basic':
        return reward_function
    if reward_name == 'no_zigzag':
        # every call to factory will make a new reward object
        no_zigzag = NoZigZag(penalize_slow=False)
        return no_zigzag.reward_function
    if reward_name == 'no_zigzag_speed':
        no_zigzag = NoZigZag(penalize_slow=True)
        return no_zigzag.reward_function
    if reward_name == 'from_2d':
        return two_d_reward
    if reward_name == 'deepracer_reward_v2':
        return deepracer_reward_v2
    raise NotImplementedError(f"Not found any reward function for '{reward_name}'")
