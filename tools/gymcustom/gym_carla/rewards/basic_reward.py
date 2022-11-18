from typing import Dict


def reward_function(state: Dict) -> float:
    # reward for speed tracking
    speed: float = state['speed']
    r_speed = speed

    # reward for steering:
    steer: float = state['steering_angle']
    r_steer: float = -steer ** 2

    # reward for out of lane
    dis = state['distance_from_center']
    is_offtrack = state['is_offtrack']
    is_crashed = state['is_crashed']
    is_stuck = state['is_stuck']

    # reward for collision
    r_collision = 0
    if is_stuck or is_crashed:
        r_collision = -1

    r_out = 1/(dis + 1e-5)
    if is_offtrack or is_crashed or is_stuck:
        r_out = -1

    # longitudinal speed
    lspeed_lon = state['speed_longitudinal_waypoint']

    # cost for too fast
    r_fast = lspeed_lon
    if lspeed_lon > 4:
        r_fast = 20

    # cost for lateral acceleration
    steer = state['steering_angle']
    r_lat = - abs(steer) * lspeed_lon ** 2

    r = 20 * r_speed + 200 * r_collision + 1 * lspeed_lon + 10 * r_fast + 10 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

    r = r / 100.

    return r
