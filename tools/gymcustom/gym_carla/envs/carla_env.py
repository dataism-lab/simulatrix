# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

from __future__ import division

import logging
import math
import os
from collections import deque
from typing import Dict, Union, Tuple, Callable

import carla
import gym
import numpy as np
from carla_birdeye_view import (BirdViewProducer, BirdViewCropType)
from carla_birdeye_view.mask import PixelDimensions
from cv2 import cv2 as cv
from gym import spaces
from gym.utils import seeding

from tools.bew import virtual_lidar as vl
from tools.common.exceptions import GymCarlaException
from tools.common.utils import join_dicts
from tools.gymcustom.gym_carla.envs import misc
from tools.gymcustom.gym_carla.envs.carla_core import CarlaCore, global_sensor_list
from tools.gymcustom.gym_carla.rewards import reward_factory

logger = logging.getLogger(__name__)

BASE_ENV_CONFIG = dict(
    sensor=dict(
        angles_front=[6, 15, 30, 45, 60, 75, 86],
        bew_height_meters=84,  # bird-eye render heights in meters
        bew_width_meters=37.5,  # bird-eye render heights in meters
        camera_size=256,
        out_lane_threshold=5.,  # threshold for out of lane
        min_crash_impulse=120,
        crop_type='front_rear',  # or 'front'
    ),
    action=dict(
        discrete=False,  # whether to use discrete control space
        discrete_acc=[-3.0, 0.0, 3.0],  # discrete value of accelerations
        discrete_steer=[-0.2, 0.0, 0.2],  # discrete value of steering angles
        continuous_accel_range=[-3.0, 3.0],  # continuous acceleration range
        continuous_steer_range=[-0.3, 0.3],  # continuous steering angle range
    ),
    max_time_episode=1000000,  # maximum timesteps per episode

    reward_name='basic',
)

TRACK_BW_MAPPING = {
    '/Game/map_package/Maps/expedition_loop_bordered/expedition_loop_bordered': 'birdview_v2_cache/map_package/Maps/expedition_loop_bordered',
    'map_package/Maps/expedition_loop_bordered/expedition_loop_bordered': 'birdview_v2_cache/map_package/Maps/expedition_loop_bordered',
}


class CarlaEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'video.frames_per_second': 20}

    def __init__(self, config: Dict):
        config = join_dicts(BASE_ENV_CONFIG, config)

        # Core init
        self.core = CarlaCore(core_config=config['carla'], exp_config=config)
        self.world = self.core.world

        # env parameters
        self.max_time_episode = config['max_time_episode']
        self.ego_state = None

        # action parameters
        cfg_action = config['action']
        self.discrete = cfg_action['discrete']
        self.discrete_act = [cfg_action['discrete_acc'], cfg_action['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(np.array([cfg_action['continuous_accel_range'][0],
                                                     cfg_action['continuous_steer_range'][0]], dtype=np.float32),
                                           np.array([cfg_action['continuous_accel_range'][1],
                                                     cfg_action['continuous_steer_range'][1]], dtype=np.float32),
                                           dtype=np.float32)  # acc, steer

        # Record the time step, total steps and resetting steps
        self.time_step = 0
        self.total_step = 0
        self.reset_step = 0

        self._prepare_sensors(config['sensor'])

        # Restore for other envs
        # mask_classes = 9
        # observation_space_dict = {
        #     'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        #     'birdeye': spaces.Box(low=0, high=1, shape=(mask_classes, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.uint8),
        #     'edge_distance': spaces.Box(low=0, high=200, shape=(self.vlidar.rays_num,), dtype=np.uint8),
        #     'state': spaces.Box(np.array([-10, -1, -5, 0], dtype=np.float32),
        #                         np.array([12, 1.5, 30, 1], dtype=np.float32), dtype=np.float32)
        # }
        # self.observation_space = spaces.Dict(observation_space_dict)

        self.prepare_observation_space()

        self._obs: Dict[str, np.array] = dict()

        # Reward function
        self._reward_funct: Callable[[Dict], float] = reward_factory(config['reward_name'])

        # Eval related variables
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0

        self._do_core_state_info = False

    def prepare_observation_space(self):
        # Last 4 items of obs are 'distance_from_center', 'delta_yaw', 'speed', 'speed_longitudinal'
        self.observation_space = spaces.Box(np.array([0] * self.vlidar.rays_num + [0, 0, -1, -1], dtype=np.float32),
                                            np.array([200] * self.vlidar.rays_num + [5.2, 180, 12, 12],
                                                     dtype=np.float32),
                                            shape=(self.vlidar.rays_num + 4,), dtype=np.float32)

    def enable_state_info(self):
        self._do_core_state_info = True

    def _prepare_sensors(self, config: Dict) -> None:
        self.camera_size = int(config['camera_size'])
        pixel_per_meter = 4
        self.bew_height = int(pixel_per_meter * config['bew_height_meters'])
        self.bew_width = int(pixel_per_meter * config['bew_width_meters'])
        self.out_lane_threshold = config['out_lane_threshold']
        self.min_crash_impulse = config['min_crash_impulse']

        ###############
        # SENSOR set-up
        ###############

        # Collision sensor
        self.collision_hist = deque()  # The collision history
        self.collision_last_n = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Camera sensor
        self.camera_img = np.zeros((self.camera_size, self.camera_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.camera_size))
        self.camera_bp.set_attribute('image_size_y', str(self.camera_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', str(self.core.timestep))

        # Add bew sensor
        if config['crop_type'] == 'front_rear':
            crop_type = BirdViewCropType.FRONT_AND_REAR_AREA
        else:
            crop_type = BirdViewCropType.FRONT_AREA_ONLY

        bw_directory: str = TRACK_BW_MAPPING[self.core.track_name]
        os.makedirs(bw_directory, exist_ok=True)
        self.bew_producer = BirdViewProducer(
            self.core.client,
            PixelDimensions(width=self.bew_width, height=self.bew_height),
            pixels_per_meter=4,
            crop_type=crop_type
        )
        # Add edge distance sensor
        self.vlidar = vl.VirtualLiDAR(angles_front=config.get('angles_front', []))

    def reset(self, **kwargs):
        self.core.reset()

        ###############
        # SENSORS reset
        ###############

        # 1. Add collision sensor
        collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.core.ego)
        collision_sensor.listen(lambda event: get_collision_hist(event))
        global_sensor_list.append(collision_sensor)

        self.collision_hist = deque()

        def get_collision_hist(event):
            # Avoid spawn impulse
            if self.time_step < 40:
                return

            impulse = event.normal_impulse
            # TODO reveal the object type at which we crashed
            # event.other_actor
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            if intensity > self.min_crash_impulse:
                self.collision_hist.append(intensity)
            # store only one most recent
            if len(self.collision_hist) > self.collision_last_n:
                self.collision_hist.popleft()

        # 2. Add camera sensor
        self.camera_bp.set_attribute('sensor_tick', str(self.core.timestep))
        camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.core.ego)

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        camera_sensor.listen(lambda data: get_camera_img(data))
        global_sensor_list.append(camera_sensor)

        # Update timestamps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.core.set_synchronous_mode(True)

        self.update_state()
        return self.get_obs()

    def update_state(self) -> None:
        state: Dict = self.core.get_state()
        state["is_crashed"] = len(self.collision_hist) > 0
        state['step'] = self.time_step
        self.ego_state = state

    def step(self, action: Union[int, Tuple[float]]):
        """
        time_step - current rollout step index
        reset_step - the rollout index
        total_step - averall sum of time_step
        """

        # Calculate acceleration and steering
        acc, steer = self.action_to_acc_steer(action)

        self.core.apply_action(acc, steer)

        if math.isnan(acc):
            raise GymCarlaException(f'Got NaN in acceleration value while steer={steer}')

        self.update_state()
        if math.isnan(self.ego_state['x']):
            raise GymCarlaException(f'Got NaN in ego_state while action={action}')

        # state information
        if self._do_core_state_info:
            avoid_keys = ['waypoints', 'track_length', 'closest_waypoints']
            info = {k: self.ego_state[k] for k in self.ego_state if k not in avoid_keys}
        else:
            info = dict()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        # is_done: bool = self._terminal() | self._truncated()
        is_done: bool = self._terminal()
        return self.get_obs(), self._reward_funct(self.ego_state), is_done, info

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        bw_masks = self._obs['birdeye']
        camera = self._obs['camera']
        bw_rgb = BirdViewProducer.as_rgb(bw_masks)

        if 'edge_distance' in self._obs:
            distances = self._obs['edge_distance']
            bw_rgb_distances = self.vlidar.vlidar_rgb(bw_rgb, bw_masks, distances)
            bw_rgb = bw_rgb_distances

        bw_rgb = cv.resize(bw_rgb, dsize=(self.camera_size, self.camera_size),
                                     interpolation=cv.INTER_AREA)
        display = np.concatenate((bw_rgb, camera), axis=1).astype(np.uint8)
        if mode == 'human':
            display = cv.cvtColor(display, cv.COLOR_RGB2BGR)
            cv.imshow('Observation', display)
            _ = cv.waitKey(10) & 0xFF
            return
        elif mode == 'rgb_array':
            return np.array(display)
        else:
            return super().render(mode=mode)  # just raise an exception

    def close(self):
        self.core.clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb',
                                    'vehicle.*',
                                    'controller.ai.walker', 'walker.*'])
        cv.destroyAllWindows()

    def action_to_acc_steer(self, action) -> (float, float):
        if self.discrete:
            acc = self.discrete_act[0][action // self.n_steer]
            steer = self.discrete_act[1][action % self.n_steer]
        else:
            # ToDo limit actions in order avoid some strange states
            acc = action[0]
            steer = action[1]

        return acc, steer

    def get_obs(self):
        # BEW sensor and Edge distance sensor
        bw_masks: np.ndarray = self.bew_producer.produce(agent_vehicle=self.core.ego)
        distances = self.vlidar(bw_masks)

        # Display camera image
        camera = cv.resize(self.camera_img, dsize=(self.camera_size, self.camera_size), interpolation=cv.INTER_AREA)

        # State observation
        state = np.array([self.ego_state['distance_from_center'],
                          abs(self.ego_state['delta_yaw']),
                          self.ego_state['speed'],
                          self.ego_state['speed_longitudinal']])

        self._obs = {
            'camera': camera.astype(np.uint8),
            'birdeye': bw_masks.astype(np.uint8),
            'edge_distance': distances.astype(np.uint8),
            'state': state.astype(np.float32),
        }

        return np.concatenate([self._obs['edge_distance'], self._obs['state']]).astype(np.float32)

    def _truncated(self):
        """Calculate whether to truncate the current episode."""
        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True
        return False

    def _terminal(self):
        """Calculate whether to terminate the current episode."""

        # If collides
        if self.ego_state['is_crashed']:
            return True

        if self.ego_state['is_stuck']:
            return True

        dis = self.ego_state['distance_from_center']
        # If out of lane
        if dis > self.out_lane_threshold:
            return True

        return False
