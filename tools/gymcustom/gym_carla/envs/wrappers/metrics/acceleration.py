from typing import List


class AccelerationTracker:
    """
    Focuses on brake and throttle metrics.
    One instance should be used only for 1 episode.
    """
    def __init__(self):
        self.len = 0
        self.speeds: List[float] = []
        self.throttles: List[float] = []
        self.brakes: List[float] = []
        self.straight_segments: List[bool] = []
        # throttle and brake below this value are treated as 0
        self.significant_pedal_application = 0.1

    def record(self, throttle, brake, speed, is_straight_segment: bool):
        self.throttles.append(throttle)
        self.brakes.append(brake)
        self.speeds.append(speed)
        self.straight_segments.append(is_straight_segment)
        self.len += 1

    @staticmethod
    def _avg(series):
        if len(series) == 0:
            return 0
        ssum = 0.0
        for e in series:
            ssum += e
        return ssum / len(series)

    def throttle_with_brake(self):
        """
        How long both pedals are pressed simultaneously.
        Note: maybe add to reward?
        :return: amount of steps when it happened
        """
        cnt = 0
        for i in range(self.len):
            if self.throttles[i] > self.significant_pedal_application and \
                    self.brakes[i] > self.significant_pedal_application:
                cnt += 1
        return cnt

    def slowing_straight(self):
        """
        Determines cases when a car slows down on a straight part of the track,
        only to speed up again before any turn comes up.
        If throttle_with_brake is high, this will also be high.
        :return: amount of cases
        """
        straight_cnt = 0
        ans_cnt = 0
        braked_straight_num = -1
        for i in range(self.len):
            if self.straight_segments[i]:
                if self.brakes[i] > self.significant_pedal_application:
                    # the current straight has braking
                    braked_straight_num = straight_cnt
                if self.throttles[i] > self.significant_pedal_application:
                    # the current straight has acceleration
                    if braked_straight_num == straight_cnt:
                        # the car already braked, so the cases counter goes up
                        ans_cnt += 1
                        # if the throttle is now applied continuously, without braking,
                        # the counter should not increase anymore.
                        # so we assume we start a new straight here
                        straight_cnt += 1
            else:
                # there is a turn, so we need to change the index of the straight
                straight_cnt += 1
        return ans_cnt

    def calculate(self) -> dict:
        """
        Produces a number of metrics related to brake and throttle.
        :return: the dictionary with metrics
        """
        metrics = dict()
        metrics["avg speed"] = self._avg(self.speeds)
        metrics["throttle with brake"] = self.throttle_with_brake() / self.len
        metrics["brake into throttle on straight"] = self.slowing_straight() / self.len
        return metrics
