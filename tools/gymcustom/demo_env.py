# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

import sys
import time
import traceback
from pathlib import Path
from pprint import pprint

# noinspection PyUnresolvedReferences
import numpy as np
from absl import flags, app
# noinspection PyUnresolvedReferences
from gym.wrappers import Monitor

# noinspection PyUnresolvedReferences
from tools.gymcustom.gym_carla import env_factory, kill_all_servers

sys.path.append(str(Path(__file__).parent))

flags.DEFINE_string('host', default='localhost', help="Simulator host url")
flags.DEFINE_string('track', '/Game/map_package/Maps/expedition_loop_bordered/expedition_loop_bordered', 'track name')
flags.DEFINE_boolean('pause', False, 'Execute ub synchronous mode')

FLAGS = flags.FLAGS
RENDER = True


def _prepare_env():
    host = FLAGS.host
    print('host:', host)
    # parameters for the gym_carla environment
    config = {
        'reward_name': 'basic',

        "sensor": {
            'angles_front': [6, 15, 30, 45, 60, 75, 86],
            # 'angles_front': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
            #                  50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88],
            'bew_height_meters': 84,  # bird-eye render heights in meters
            'bew_width_meters': 37.5,  # bird-eye render heights in meters
            'camera_size': 256,
            'out_lane_threshold': 5.,  # threshold for out of lane
            'min_crash_impulse': 120,
            'crop_type': 'front_rear',  # or 'front''
        },
        'action': {
            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        },
        "carla": {
            "host": "127.0.0.1",  # Client host
            "timeout": 10.0,  # Timeout of the client
            "timestep": 1. / 25,  # time interval between two frames or 1./FPS
            'track': FLAGS.track,  # which town to simulate
            "retries_on_error": 10,  # Number of tries to connect to the client
            "enable_map_assets": False,  # enable / disable all town assets except for the road
            "enable_rendering": RENDER,  # enable / disable camera images
            "avoid_server_init": True,  # Create or use an existing server
            "port": 2000,  # default port, used in case of avoid_server_init is False
            "quality_level": "Low"
        },

        "spawn_points": [
            # "-25.518145,80.065155,29.364782",  # x,y,yaw
            # "32.551285,94.990494,-63.094921",
            # "43.0,59.7,-137.353",
            # "-50.5695686340332,49.29917526245117,131.55929565429688",
            # "29.6,25.5,-224.999"
        ],
        'spawn_mode': 'random',  # [random, consequent]
        'n_spawn_points': 30,
        'consequent_mode_step': 3,
        'spawn_off_center': True,

        'augmentations': {
            'throttle_noise_std': 0.01,
            'brake_noise_std': 0.01,
            'steering_noise_std': 0.01,
            'randomize_fps': False,
            'randomize_fps_every': 100,
            'low_friction_enable': False,
            'low_friction_n': 2,
            'low_friction_multiplier': 0.7,
            'low_friction_areas_size': 100,
            'weather_change_enable': False,
            'weather_update_freq': 10,
            'weather_speed_factor': 1
        }
    }

    env = env_factory(config, env_type='carla-racer-minmax')
    # env = env_factory(config, env_type='carla-racer')
    # env = env_factory(config, env_type='carla-camera')
    return env


def demo(*args, **kwargs):
    env = _prepare_env()

    print(env.observation_space)
    try:

        # env = Monitor(env, './video', force=True, write_upon_reset=True)
        obs = env.reset()
        i = 1
        prev_pause_i = i - 1
        s = time.time()
        obs_min = None
        obs_max = None
        while True:
            # forward action
            # action = env.action_space.sample()
            action = [2.1, 0.05]

            obs, r, done, info = env.step(action)
            if obs_min is None:
                obs_min = obs
                obs_max = obs
            else:
                obs_min = np.minimum(obs_min, obs)
                obs_max = np.maximum(obs_max, obs)

            # print(obs)
            if RENDER:
                env.render()

            if FLAGS.pause and ((i % 200 == 0) or done):
                del env.ego_state['waypoints']
                pprint(env.ego_state)
                print('Rollout length =', i - prev_pause_i)
                input("Press Enter to continue...\n")
                prev_pause_i = i
            if i % 200 == 0:
                print('-------------')
                print('obs_min:', obs_min.tolist())
                print('obs_max:', obs_max.tolist())
                call_fps = 1 / (time.time() - s)
                env_fps = 1. / env.core.timestep
                print('Calls per second:', call_fps, 'while env FPS is', env_fps, f'({call_fps / env_fps})X')

            i += 1
            s = time.time()

            if done:
                print('=== steps', env.time_step, 'reset_count', env.reset_step, 'speed', obs[-2])
                obs = env.reset()

    except KeyboardInterrupt:  # pragma: no branch
        pass
    except Exception:
        traceback.print_exc()
    finally:
        env.close()
        kill_all_servers()


if __name__ == "__main__":
    app.run(demo)
