from typing import Dict

from tools.common.average_meter import AverageMeter


class AggregatedMetrics:
    """
    Merges metric results across multiple episodes.
    """
    def __init__(self):
        self.dict: Dict[str, AverageMeter] = dict()

    def update(self, additional: Dict[str, float], run_length: int):
        for key, value in additional.items():
            if key not in self.dict:
                self.dict[key] = AverageMeter()
            self.dict[key].update(value, run_length)

    def __str__(self):
        ans = ""
        for key, value in self.dict.items():
            ans += f"{key}: {value.avg}\n"
        return ans
