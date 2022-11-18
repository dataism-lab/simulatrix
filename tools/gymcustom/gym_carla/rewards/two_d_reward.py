from typing import Dict

import numpy as np


class RewardCoefs:
    off_track_end_penalty = -500
    idle = -1
    close_to_center_coef = 1
    progress_coef = 100
    low_steering_angle_coef = 3
    general_speed_coef = 0.05
    waypoint_angle_coef = 10
    waypoint_angle_skip = 3
    square_waypoint_angle = True
    SLOW_SPEED = 15
    HIGH_SPEED = 50


def reward_function(state: Dict, verbose=False) -> float:
    speed: float = state['speed']
    steer: float = _to_rad(state['steering_angle'])
    is_offtrack = state['is_offtrack']
    waypoint_yaw = _to_rad(state['delta_yaw'])

    if is_offtrack:
        return RewardCoefs.off_track_end_penalty

    reward = RewardCoefs.idle

    prog = state['progress'] * RewardCoefs.progress_coef
    reward += prog
    if verbose:
        print(f"progress: {prog}")

    # slow speed is very bad
    if speed < RewardCoefs.SLOW_SPEED:
        reward -= RewardCoefs.SLOW_SPEED - speed
        if verbose:
            print(f"slow speed penalty: {-(RewardCoefs.SLOW_SPEED - speed)}")

    high_speed = (speed - RewardCoefs.HIGH_SPEED) * RewardCoefs.general_speed_coef
    reward += high_speed
    if verbose:
        print(f"high speed: {high_speed}")

    # low steering angle is good
    steering = (0.42 - abs(steer)) * RewardCoefs.low_steering_angle_coef
    reward += steering
    if verbose:
        print(f"steering: {steering}")

    # reduce reward based on relative distance from center
    dist = state['distance_from_center']
    center = (dist / (state['track_width'] / 2)) * RewardCoefs.close_to_center_coef
    reward -= center
    if verbose:
        print(f"center dist: {-center}")

    # car should aim at a forward waypoint
    if RewardCoefs.square_waypoint_angle:
        waypoint_yaw = waypoint_yaw ** 2
    wp_angle_penalty = waypoint_yaw * RewardCoefs.waypoint_angle_coef
    reward -= wp_angle_penalty
    if verbose:
        print(f"waypoint angle: {-wp_angle_penalty}")
    return reward


def _to_rad(angle: float) -> float:
    return angle * np.pi / 180
