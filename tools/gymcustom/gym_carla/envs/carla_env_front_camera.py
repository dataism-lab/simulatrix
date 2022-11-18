# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

from __future__ import division

from typing import Dict

import numpy as np
from cv2 import cv2 as cv
from gym import spaces

from tools.gymcustom.gym_carla.envs.carla_env import CarlaEnv


class CarlaFrontCameraEnv(CarlaEnv):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.vlidar = None

    def prepare_observation_space(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera_size, self.camera_size, 3),
                                            dtype=np.uint8)

    def get_obs(self):
        # BEW sensor and Edge distance sensor
        bw_masks: np.ndarray = self.bew_producer.produce(agent_vehicle=self.core.ego)

        # Display camera image
        camera = cv.resize(self.camera_img, dsize=(self.camera_size, self.camera_size), interpolation=cv.INTER_AREA)

        self._obs = {
            'camera': camera.astype(np.uint8),
            'birdeye': bw_masks.astype(np.uint8)
        }

        return self._obs['camera']
