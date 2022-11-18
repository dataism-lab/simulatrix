import collections
import os
import queue
from pathlib import Path

import carla
import numpy as np

_root_fld = Path(__file__).absolute().parent.parent.parent.as_posix()


def get_abs_resourcepath(relative_fpath: str) -> str:
    return os.path.join(_root_fld, 'resources', relative_fpath)


def relative_coordinate_1(target_data, camera_rgb, K):
    # import transforms3d
    # import math
    """
    Convert 3D coordinate of vehicles in carla to image coordinate using pinhole equation
    https://github.com/carla-simulator/carla/discussions/5229
    camera rgb of ego vehicle, also remember to change camera_rgb and K matrix according to left, right and back camera
    """
    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
    world_points = np.array(
        [target_data.get_location().x, target_data.get_location().y, target_data.get_location().z, 1]).T

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_points)
    point_in_camera_coords = np.array([sensor_points[1], sensor_points[2] * -1, sensor_points[0]]).T

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([points_2d[0] / points_2d[2], points_2d[1] / points_2d[2], points_2d[2]])

    return points_2d[0], points_2d[1] - 10


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


