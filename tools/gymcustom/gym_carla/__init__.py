from typing import Dict

import gym
from gym.envs.registration import register
from gym.wrappers import TransformObservation, NormalizeObservation
from tools.gymcustom.gym_carla.envs.carla_core import kill_all_servers
from tools.gymcustom.gym_carla.envs.wrappers.metrics_wrapper import MetricsWrapper
from tools.common.exceptions import GymCarlaException

register(
    id='carla-racer-v2',
    entry_point='tools.gymcustom.gym_carla.envs:CarlaEnv'
)

register(
    id='carla-camera-v1',
    entry_point='tools.gymcustom.gym_carla.envs:CarlaFrontCameraEnv'
)

ALLOWED_ENV_TYPES = ['carla-racer', 'carla-racer-minmax', 'carla-racer-norm', 'carla-racer-metrics',
                     'carla-camera']


def env_factory(config: Dict, env_type='carla-racer'):
    assert env_type in ALLOWED_ENV_TYPES, f"invalid env_type='{env_type}', should be one of {ALLOWED_ENV_TYPES}"
    if '-racer' in env_type:
        env = gym.make('carla-racer-v2', config=config)
        if env_type == ALLOWED_ENV_TYPES[0]:
            return env

        if env_type == ALLOWED_ENV_TYPES[1]:
            min_th = env.observation_space.low
            max_th = env.observation_space.high
            diff = max_th - min_th
            env = TransformObservation(env, lambda obs: (obs - min_th) / (diff + 1e-7))
            return env

        if env_type == ALLOWED_ENV_TYPES[2]:
            env = NormalizeObservation(env)
            return env

        if env_type == ALLOWED_ENV_TYPES[3]:
            env = MetricsWrapper(env)
            env.set_compute_metrics(True)
            return env

    elif '-camera' in env_type:
        env = gym.make('carla-camera-v1', config=config)
        if env_type == ALLOWED_ENV_TYPES[4]:
            return env

    raise GymCarlaException(f'No such env_type="{env_type}", allowed: {ALLOWED_ENV_TYPES}')
