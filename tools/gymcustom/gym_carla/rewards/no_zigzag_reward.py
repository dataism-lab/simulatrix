from typing import Dict


class Reward:
    def __init__(
            self,
            penalize_slow: bool, speed_threshold=0.5,
            penalize_steering=True, steering_threshold=15
    ):
        self.penalize_slow = penalize_slow
        self.speed_threshold = speed_threshold
        self.penalize_steering = penalize_steering
        self.steering_threshold = steering_threshold

    def reward_function(
            self,
            params: Dict,
    ) -> float:
        """
        Example that penalizes steering, which helps mitigate zig-zag behaviors.

        Maybe adding waypoint angle will also help.
        """

        # Calculate 3 marks that are farther and father away from the center line
        marker_1 = 0.1 * params['track_width']
        marker_2 = 0.25 * params['track_width']
        marker_3 = 0.5 * params['track_width']

        # Give higher reward if the car is closer to center line and vice versa
        if params['distance_from_center'] <= marker_1:
            reward = 1
        elif params['distance_from_center'] <= marker_2:
            reward = 0.5
        elif params['distance_from_center'] <= marker_3:
            reward = 0.1
        else:
            reward = 1e-3  # likely crashed/ close to off track

        # Penalize reward if the car is steering too much
        if self.penalize_steering:
            if abs(params[
                       'steering_angle']) > self.steering_threshold:  # Only need the absolute steering angle
                reward *= 0.5

        # penalize reward for the car taking slow actions
        if self.penalize_slow:
            if params['speed'] < self.speed_threshold:
                reward *= 0.5

        return float(reward)
