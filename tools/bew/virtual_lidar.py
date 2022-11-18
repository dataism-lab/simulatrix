# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

from __future__ import annotations

import logging
import math
import os
import time
from math import sqrt
from pathlib import Path
from typing import Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
from carla_birdeye_view import BirdViewMasks, BirdViewCropType
from cv2 import cv2 as cv

from tools.common import get_abs_resourcepath
from tools.common.exceptions import BewParamsException

PointType = Tuple[Union[float, int], Union[float, int]]
RayType = Tuple[PointType, PointType]

mask_types = [
    # BirdViewMasks.PEDESTRIANS,
    # BirdViewMasks.RED_LIGHTS,
    # BirdViewMasks.YELLOW_LIGHTS,
    # BirdViewMasks.GREEN_LIGHTS,
    BirdViewMasks.AGENT,
    BirdViewMasks.CENTERLINES,
    BirdViewMasks.LANES,
    BirdViewMasks.ROAD
]
ORANGE = (252, 175, 62)
BLUE = (0, 164, 255)
DIM_GRAY = (105, 105, 105)
_root_fld = Path(__file__).absolute().parent.parent.parent.as_posix()

logger = logging.getLogger(__name__)


#######
# Utils
#######

def _load_bew_example(fpath_no_ext: str) -> Tuple[np.ndarray, np.ndarray]:
    """load image and masks"""
    bw_masks = np.load(fpath_no_ext + '.npy')
    bw_bgr = cv.imread(fpath_no_ext + '.png')
    return bw_bgr, bw_masks


def _show_bgr(bw_bgr: np.ndarray) -> None:
    plt.axis('off')
    plt.imshow(cv.cvtColor(bw_bgr, cv.COLOR_BGR2RGB))
    plt.show()


def _apply_mask(bw_bgr: np.ndarray, bw_masks: np.ndarray, mask_type: BirdViewMasks) -> np.ndarray:
    mask = bw_masks[mask_type.numerator]
    idx = (mask == 0)

    bw_bgr_copy = bw_bgr.copy()
    bw_bgr_copy[idx] = 0
    return bw_bgr_copy


def show_all_masks(bw_bgr: np.ndarray, bw_masks: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(mask_types), figsize=(12, 9))
    for i, mask_type in enumerate(mask_types):
        bwg_plot = _apply_mask(bw_bgr, bw_masks, mask_type)
        axes[i].axis('off')
        axes[i].set_title(mask_type.name)
        axes[i].imshow(cv.cvtColor(bwg_plot, cv.COLOR_BGR2RGB))

    plt.show()


def ray_distance(bw_road_mask: np.ndarray, start: PointType, vector: PointType) -> float:
    """
    :param bw_road_mask: birds-eye view mask of a road.
    :param start: start point for the ray.
    :param vector: a direction from starting point.
    :return: return distance from starting point to a road edge in pixels count.
    """

    length = 0
    ray_end = np.array(start, dtype=np.float32)
    vector_length = sqrt(vector[0] ** 2 + vector[1] ** 2)
    unit_vector = np.array(vector, dtype=np.float32) / vector_length

    # loop condition checks pixel_color ~= road_color, and image index boundaries
    y, x = start
    y, x = int(y), int(x)
    while 0 <= y < bw_road_mask.shape[0] and 0 <= x < bw_road_mask.shape[1] and bw_road_mask[y][x] == 1:
        ray_end += unit_vector
        length += 1
        y = int(ray_end[0])
        x = int(ray_end[1])
    return length


#####################
# Edge Distnace LiDAR
#####################

class VirtualLiDAR:
    _type_example_mapping = {BirdViewCropType.FRONT_AND_REAR_AREA: get_abs_resourcepath('bew_types/free_way'),
                             BirdViewCropType.FRONT_AREA_ONLY: get_abs_resourcepath('bew_types/lear_by_cheating')}

    def __init__(self, bew_type: BirdViewCropType = BirdViewCropType.FRONT_AND_REAR_AREA,
                 angles_front: Tuple[int] = None,
                 y_directs: Tuple[int] = None):
        self._car_front, self.shape_yx = self._lidar_position(bew_type)
        self.rays, self.angles = self._prepare_rays(angles_front, y_directs)
        assert self.angles and len(self.angles) == len(self.rays)

    @property
    def rays_num(self):
        return len(self.rays)

    def _lidar_position(self, bew_type: BirdViewCropType) -> Tuple[PointType, Tuple[int, int]]:
        """
        Reveal Virtual LiDAR position on the image as (y, x)
        """
        bw_bgr, bw_masks = _load_bew_example(self._type_example_mapping[bew_type])
        b = bw_masks[BirdViewMasks.AGENT]
        ys, xs = np.where(b == 1)
        car_y = min(ys)
        car_x = int(np.median(xs))
        logger.debug(f'Virtual LiDAR position is y={car_y}, x={car_x}')
        car_front = (float(car_y), float(car_x))
        shape_yx = (bw_bgr.shape[0], bw_bgr.shape[1])
        return car_front, shape_yx

    def _prepare_rays(self, angles_front, y_directs) -> Tuple[List[RayType], List[int]]:
        """
        a ray is a tuple of starting point and unit direction vector
        """
        if not angles_front:
            angles_front = (6, 15, 30, 45, 60, 75, 86)
            logger.info(f'angles front uses default values: {angles_front}')
        elif any(x < 0 or x > 90 for x in angles_front):
            raise BewParamsException(
                f"angles_front param should contain values in range [0,90], but got: {angles_front}")

        if not y_directs:
            y_directs = (-1, 1)
            logger.info(f'y directions uses default values: {y_directs}')
        elif any(y != -1 and y != 1 for y in y_directs):
            raise BewParamsException(
                f"y_directs param should contain one of [-1, 1] of both, but got: {y_directs}")

        rays = []
        angles = []
        for y_direct in y_directs:
            ray = (self._car_front, (y_direct, 0))
            rays.append(ray)
            initial_angle = 0 if y_direct == -1 else 180
            angles.append(initial_angle)
            for angle in angles_front:
                tangents = math.tan(angle * math.pi / 180)
                l_ray = (self._car_front, (y_direct, tangents))
                r_ray = (self._car_front, (y_direct, -tangents))
                angles += [initial_angle + angle, initial_angle - angle]
                rays += [l_ray, r_ray]

        return rays, angles

    def __call__(self, bw_masks: np.ndarray) -> np.array:
        bw_road_mask = bw_masks[BirdViewMasks.ROAD]
        distances = []
        for start_point, unit_vector in self.rays:
            ray_dist = ray_distance(bw_road_mask, start_point, unit_vector)
            distances.append(ray_dist if ray_dist > 0 else 0)
        return np.array(distances, dtype=float)

    def create_edge_points_mask(self, distances: np.array, p_ratio=2) -> np.ndarray:
        edge_mask = np.zeros(self.shape_yx)
        max_y, max_x = self.shape_yx

        car_y, car_x = self._car_front
        for angle, dist in zip(self.angles, distances):

            x_delta = dist * math.sin(math.radians(angle))
            # reflection
            if angle > 90:
                x_delta = -x_delta
            x = car_x + x_delta
            x = int(x)
            if x < 0:
                x = 1
            elif x >= max_x:
                x = max_x - 1

            y_delta = dist * math.cos(math.radians(angle))
            y = car_y - y_delta
            y = int(y)
            if y < 0:
                y = 1
            elif y >= max_y:
                y = max_y - 1

            # print(angle, dist, '|delta:', y_delta, x_delta, '|y,x:', y, x)

            edge_mask[max(y - p_ratio, 0):y + p_ratio, max(x - p_ratio, x):x + p_ratio] = 1
        return edge_mask

    def vlidar_rgb(self, bw_bgr: np.ndarray, bw_masks: np.ndarray, distances: np.array) -> np.ndarray:

        edge_mask = self.create_edge_points_mask(distances)

        # combine agent mask and edge mask in RGB
        bw_dpoints = _apply_mask(bw_bgr, bw_masks, BirdViewMasks.AGENT)
        idx = (edge_mask == 1)
        bw_dpoints[idx] = BLUE

        bw_out = np.zeros(bw_dpoints.shape, dtype=np.uint8)
        # road
        road_mask = bw_masks[BirdViewMasks.ROAD]
        idx = (road_mask == 1)
        bw_out[idx] = DIM_GRAY

        # combine road RGB with agent and edge points
        idx = (bw_dpoints != 0)
        bw_out[idx] = bw_dpoints[idx]
        return bw_out


if __name__ == '__main__':
    vlidar = VirtualLiDAR()

    examples_fld = _root_fld + '/notebooks/out'
    fnames = sorted(x.split('.')[0] for x in os.listdir(examples_fld) if x.endswith('.npy'))

    while True:
        for fname in fnames:
            ex_fpath = f'{examples_fld}/{fname}'
            bw_bgr, bw_masks = _load_bew_example(ex_fpath)

            # _show_bgr(bw_bgr)
            # show_all_masks(bw_bgr, bw_masks)

            distances: np.array = vlidar(bw_masks)

            # print('angle', 'dist')
            # pprint(list(zip(vlidar.angles, distances)))

            bw_rgb_out = vlidar.vlidar_rgb(bw_bgr, bw_masks, distances)
            cv.imshow("BirdView RGB", bw_rgb_out)
            # Play next frames without having to wait for the key
            key = cv.waitKey(10) & 0xFF
            time.sleep(0.3)

    cv.destroyAllWindows()
