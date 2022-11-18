import gym
import numpy as np

from tools.gymcustom.gym_carla.envs.carla_core import CarlaCore
from tools.gymcustom.gym_carla.envs.misc import is_on_straight
from tools.gymcustom.gym_carla.envs.wrappers.metrics.acceleration import AccelerationTracker
from tools.gymcustom.gym_carla.envs.wrappers.metrics.aggregate import AggregatedMetrics
from tools.gymcustom.gym_carla.envs.wrappers.metrics.steering import SteeringTracker


class MetricsWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, compute_metrics=False):
        # Call the parent constructor, so we can access self.env later
        super(MetricsWrapper, self).__init__(env)

        self.steering_metric: SteeringTracker = SteeringTracker()
        self.acceleration_metric: AccelerationTracker = AccelerationTracker()
        self.aggregator: AggregatedMetrics = AggregatedMetrics()
        self.compute_metrics = compute_metrics

    def step(self, action):
        """
        Does not change the action, observation, reward, or done.
        Add 'computing metrics' to info.
        Records metrics iff self.compute_metrics.
        """
        obs, reward, done, info = self.env.step(action)
        info['computing metrics'] = self.compute_metrics
        if self.compute_metrics:
            car_pos = np.array([obs['x'], obs['y']])
            acc, steer = self.unwrapped.action_to_acc_steer(action)
            throttle, brake = CarlaCore.acc_to_throttle_break(acc)

            waypoints = obs['waypoints']
            prev_wp = waypoints[obs['closest_waypoints'][0]]
            curr_wp = waypoints[obs['current_wp_index']]
            next_wp = waypoints[obs['closest_waypoints'][1]]

            is_straight = is_on_straight(prev_wp, curr_wp, next_wp)

            self.steering_metric.record(car_pos, obs['steering_angle'], obs['distance_from_center'], is_straight)
            self.acceleration_metric.record(throttle, brake, obs['speed'], is_straight)

            if done:
                self.reset_metrics()
        return obs, reward, done, info

    def reset_metrics(self):
        """
        Updates the aggregator with the last episode and resets the metric trackers.
        """
        episode_length = self.steering_metric.len

        self.aggregator.update(self.steering_metric.calculate(), episode_length)
        self.aggregator.update(self.acceleration_metric.calculate(), episode_length)
        self.aggregator.update({"ep length": episode_length}, 1)

        self.steering_metric = SteeringTracker()
        self.acceleration_metric = AccelerationTracker()

    def set_compute_metrics(self, value: bool):
        """
        To start metric computation, set self.compute_metrics to True.
        """
        self.compute_metrics = value

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        return self.aggregator
