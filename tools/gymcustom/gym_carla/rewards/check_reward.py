import json
import os
import sys
from pathlib import Path

import numpy as np

from tools.gymcustom.gym_carla.rewards import reward_factory

_root_fld = Path(__file__).parent.as_posix()

if __name__ == "__main__":
    # reward_name = sys.argv[1]
    reward_name = "two_d_reward"
    param_fpath = os.path.join(_root_fld, 'exparams.json')

    with open(param_fpath, 'r') as f:
        params = json.load(f)

    # my_module = importlib.import_module(reward_name)
    # res = my_module.reward_function(params)

    func = reward_factory('no zigzag', np.zeros(0))
    res = func(params)

    func2 = reward_factory('no zigzag', np.zeros(0))
    print(func is func2)

    print('res', res)
    if res:
        sys.exit(0)
