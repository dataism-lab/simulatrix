from typing import List

import numpy as np


class SteeringTracker:
    """
    Focuses on steering metrics, and going straight on the straight.
    One instance should be used only for 1 episode.
    """
    def __init__(self):
        self.len = 0
        self.positions: List[np.ndarray] = []
        self.steering: List[float] = []
        self.center_dist: List[float] = []
        self.straight_segments: List[bool] = []
        # a hyperparameter for the metric: how much steering is considered negligible
        # self.straight_steering_threshold = 0.07
        self.extremum_precision_relative = 0.15
        self.interval_precision_relative = 0.15

    def record(self, car_pos: np.ndarray, steering_angle, center_dist, is_straight_segment: bool):
        """

        :param car_pos:
        :param steering_angle:
        :param center_dist:
        :param is_straight_segment:
        :return:
        """
        self.positions.append(car_pos)
        self.steering.append(steering_angle)
        self.center_dist.append(center_dist)
        self.straight_segments.append(is_straight_segment)
        self.len += 1

    def calculate(self) -> dict:
        """
        Produces a number of metrics related to wiggling.
        :return: the dictionary with metrics
        """
        metrics = dict()
        metrics["avg absolute steering"] = self._avg_absolute(self.steering)
        straight_steering = []
        for i in range(self.len):
            if self.straight_segments[i]:
                straight_steering.append(self.steering[i])
        metrics["avg steering on straight"] = self._avg_absolute(straight_steering)
        metrics["avg center distance"] = self._avg_absolute(self.center_dist)
        metrics["cyclic steering"] = self.cyclic_steering() / len(self.steering)
        return metrics

    @staticmethod
    def _avg_absolute(series):
        if len(series) == 0:
            return 0
        ssum = 0.0
        for e in series:
            ssum += abs(e)
        return ssum / len(series)

    def cyclic_steering(self):
        """
        Calculates the number of times steering angle goes approximately like this
        A -> B            (here score = 0)
        A -> B -> A       (score = 1)
        A -> B -> A -> B  (score = 2)
        where A is a local maximum, B is a local minimum, (or vice versa)
        and intervals between each neighbouring A and B are equal
        :return: the total score
        """
        ans = 0
        extremums = self._extremums(self.steering)
        last_max = 0  # index
        last_min = 0
        last_interval = 0
        for i in range(self.len):
            if extremums[i] == 1:
                precision = self.extremum_precision_relative * self.steering[i]
                new_interval = i - last_min
                if abs(self.steering[last_max] - self.steering[i]) < precision:
                    # extremum's value is close to the last corresponding one
                    interval_precision = new_interval * self.interval_precision_relative
                    if abs(new_interval - last_interval) < interval_precision:
                        # interval from last extremum is similar to last one
                        ans += 1
                last_max = i
                last_interval = new_interval
            elif extremums[i] == -1:
                precision = self.extremum_precision_relative * self.steering[i]
                new_interval = i - last_max
                if abs(self.steering[last_min] - self.steering[i]) < precision:
                    interval_precision = new_interval * self.interval_precision_relative
                    if abs(new_interval - last_interval) < interval_precision:
                        ans += 1
                last_min = i
                last_interval = new_interval
        return ans

    @staticmethod
    def _extremums(series):
        ans = [0]
        for i in range(1, len(series) - 1):
            if series[i - 1] > series[i] <= series[i + 1]:
                ans.append(-1)
            elif series[i - 1] < series[i] >= series[i + 1]:
                ans.append(1)
            else:
                ans.append(0)
        ans.append(0)
        return ans
